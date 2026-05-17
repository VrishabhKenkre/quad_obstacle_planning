"""
train_diffusion.py -- BC training of the diffusion student on the
planner_dataset_v1.npz dataset, plus 10-seed evaluation.

Pipeline:
  1. Load Phase 1 dataset.
  2. Build sliding windows of length WINDOW=8 within each rollout
     (variant_id + contiguous step_indices).
  3. Fit LinearNormalizer on the dataset arrays.
  4. Train DiffusionUnetLowdimPolicy via DDPM noise prediction, with
     EMA on the weights (decay 0.999, ramp).
  5. Switch to DDIM @ 8 steps for inference. Eval on 10-seed obstacle
     course, save results/diffusion_distill_10seed.json.

Usage:
    python distillation/train_diffusion.py            # full run
    python distillation/train_diffusion.py --epochs 30   # short
    python distillation/train_diffusion.py --eval-only --model data/diffusion_student_v1_ema.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / 'planning'))
sys.path.insert(0, str(_HERE.parent / 'src'))
sys.path.insert(0, str(_HERE.parent / 'external' / 'diffusion_policy'))

from voxelize import VoxelMap
from quad_env import CrazyflieEnv
from obstacle_course import make_obstacles, obstacle_field_value

from collect_planner_data import make_observation, OBS_DIM, ACT_DIM, START, GOAL
from diffusion_student import (
    build_diffusion_policy, make_inference_scheduler,
    build_normalizer_from_arrays, DEFAULT_HORIZON)
from diffusion_policy.model.diffusion.ema_model import EMAModel


SEEDS = [42, 7, 13, 99, 256, 128, 314, 2024, 777, 1337]


# ---- Windowed dataset ----------------------------------------------------

class PlannerWindowDataset(Dataset):
    """Sliding-window dataset: each item is (obs(1, Do), action(window, Da))
    drawn from a single rollout. The window starts at any step i such that
    i .. i+window-1 are all in the same rollout."""

    def __init__(self, obs: np.ndarray, act: np.ndarray,
                 variant_ids: np.ndarray, step_indices: np.ndarray,
                 window: int = DEFAULT_HORIZON):
        self.obs = obs.astype(np.float32)
        self.act = act.astype(np.float32)
        self.window = int(window)
        n = obs.shape[0]
        # A valid start i requires variant_ids[i+window-1] == variant_ids[i]
        # AND step_indices[i+window-1] - step_indices[i] == window - 1.
        if n < window:
            self.starts = np.array([], dtype=np.int64)
            return
        same_var = (variant_ids[:n - window + 1] ==
                    variant_ids[window - 1:])
        contig = (step_indices[window - 1:] -
                  step_indices[:n - window + 1]) == (window - 1)
        self.starts = np.flatnonzero(same_var & contig).astype(np.int64)

    def __len__(self):
        return int(len(self.starts))

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        return dict(
            obs=self.obs[s:s+1],                      # (1, Do)
            action=self.act[s:s + self.window],       # (window, Da)
        )


def _collate(batch):
    obs = torch.from_numpy(np.stack([b['obs'] for b in batch], axis=0))
    act = torch.from_numpy(np.stack([b['action'] for b in batch], axis=0))
    return dict(obs=obs, action=act)


# ---- Training ------------------------------------------------------------

def train_diffusion(policy, dataset: PlannerWindowDataset, device,
                    epochs: int = 100, batch_size: int = 256, lr: float = 1e-4,
                    val_frac: float = 0.05, ema_decay: float = 0.999,
                    verbose: bool = True):
    """Standard DDPM noise-prediction BC training with EMA."""
    import copy
    n_total = len(dataset)
    val_n = max(int(n_total * val_frac), 256)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_total)
    val_idx = perm[:val_n]; tr_idx = perm[val_n:]

    tr_set = torch.utils.data.Subset(dataset, tr_idx.tolist())
    va_set = torch.utils.data.Subset(dataset, val_idx.tolist())
    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True,
                           num_workers=4, persistent_workers=True,
                           collate_fn=_collate, pin_memory=True)
    va_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False,
                           num_workers=2, persistent_workers=True,
                           collate_fn=_collate, pin_memory=True)

    opt = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-6)

    ema_policy = copy.deepcopy(policy).to(device).eval()
    ema = EMAModel(ema_policy, update_after_step=0,
                   inv_gamma=1.0, power=2/3,
                   min_value=0.0, max_value=ema_decay)

    tr_curve, va_curve = [], []
    best_val = float('inf')
    best_ema_state = None
    for ep in range(epochs):
        policy.train()
        ep_loss = 0.0; ep_n = 0
        t0 = time.perf_counter()
        for batch in tr_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = policy.compute_loss(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            ema.step(policy)
            ep_loss += float(loss.item()) * batch['obs'].shape[0]
            ep_n += batch['obs'].shape[0]
        tr_loss = ep_loss / max(ep_n, 1)
        tr_curve.append(tr_loss)

        # Validation: use EMA model for cleaner signal
        ema_policy.eval()
        with torch.no_grad():
            v_loss = 0.0; v_n = 0
            for batch in va_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                vloss = ema_policy.compute_loss(batch)
                v_loss += float(vloss.item()) * batch['obs'].shape[0]
                v_n += batch['obs'].shape[0]
            v_loss = v_loss / max(v_n, 1)
        va_curve.append(v_loss)
        dt = time.perf_counter() - t0
        if v_loss < best_val:
            best_val = v_loss
            best_ema_state = {k: v.detach().clone() for k, v in
                              ema_policy.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}: tr={tr_loss:.4f} ema_val={v_loss:.4f} "
                  f"({dt:.1f}s, ema_decay={ema.decay:.4f})")

    if best_ema_state is not None:
        ema_policy.load_state_dict(best_ema_state)
    return dict(policy=policy, ema_policy=ema_policy,
                tr_curve=tr_curve, va_curve=va_curve,
                best_val=best_val)


# ---- 10-seed eval --------------------------------------------------------

def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _sem(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std() / np.sqrt(len(arr)))


def run_diffusion_seed(policy, seed: int, device, dt_ctrl: float = 0.02,
                       T_max: float = 10.0,
                       diffusion_seed: int | None = None,
                       multi_K: int = 1) -> dict:
    obstacles = make_obstacles(seed=seed)
    vm = VoxelMap()
    vm.from_obstacle_field(obstacles)
    vm.compute_esdf()
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=dt_ctrl)
    state_mj = env.reset(pos=START)
    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0

    if diffusion_seed is not None:
        torch.manual_seed(int(diffusion_seed))
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(int(diffusion_seed))

    # Lazy-imports for multi-sample safety scorer to avoid widening
    # this module's hard deps when multi_K==1.
    if multi_K and multi_K > 1:
        sys.path.insert(0, str(_HERE))
        from eval_decision_points import (
            _predict_traj_linearized, _QP)
        u_hover_ms = np.array([_QP.hover_thrust, 0.0, 0.0, 0.0])

    n_steps = int(T_max / dt_ctrl)
    xs = [state_mj.copy()]
    field_vals = [obstacle_field_value(state_mj[0:3], obstacles)]
    inf_times = []
    policy.eval()
    for i in range(n_steps):
        obs = make_observation(state_mj, GOAL, vm)
        t0 = time.perf_counter()
        with torch.no_grad():
            if multi_K and multi_K > 1:
                obs_t = torch.from_numpy(obs).to(device).reshape(1, 1, OBS_DIM)
                obs_t = obs_t.expand(int(multi_K), 1, OBS_DIM).contiguous()
                result = policy.predict_action({'obs': obs_t})
                actions_k = result['action'].cpu().numpy()
                best_k = 0; best_score = -float('inf')
                for k in range(int(multi_K)):
                    traj = _predict_traj_linearized(
                        state_mj, actions_k[k], u_mid, u_half,
                        u_hover_ms, dt_ctrl)
                    sdfs = [vm.query_esdf(p) for p in traj]
                    score = float(min(sdfs))
                    if score > best_score:
                        best_score = score; best_k = k
                a = actions_k[best_k, 0]
            else:
                obs_t = torch.from_numpy(obs).to(device).reshape(1, 1, OBS_DIM)
                result = policy.predict_action({'obs': obs_t})
                a = result['action'][0, 0].cpu().numpy()
        torch.cuda.synchronize() if device.type == 'cuda' else None
        inf_times.append(time.perf_counter() - t0)
        u = u_mid + u_half * a
        u = np.clip(u, env.u_min, env.u_max)
        state_mj = env.step(u)
        xs.append(state_mj.copy())
        field_vals.append(obstacle_field_value(state_mj[0:3], obstacles))

    xs = np.asarray(xs); field_vals = np.asarray(field_vals)
    final_pos = xs[-1, 0:3]
    goal_err_mm = float(np.linalg.norm(final_pos - GOAL) * 1000)
    path_len_m = float(np.sum(np.linalg.norm(np.diff(xs[:, 0:3], axis=0), axis=1)))
    duration_s = n_steps * dt_ctrl
    return dict(
        goal_err_mm=goal_err_mm,
        max_field=float(np.max(field_vals)),
        mean_field=float(np.mean(field_vals)),
        path_len_m=path_len_m,
        mean_speed_mps=path_len_m / max(duration_s, 1e-6),
        median_inference_us=float(np.median(inf_times) * 1e6),
        mean_inference_us=float(np.mean(inf_times) * 1e6),
        n_steps=int(n_steps),
    )


def run_10seed_eval(policy, save_path: str, device, verbose: bool = True,
                    multi_K: int = 1) -> dict:
    rows = []
    detail = {}
    for s in SEEDS:
        if verbose:
            print(f"[diffusion eval] seed {s}")
        r = run_diffusion_seed(policy, s, device,
                               diffusion_seed=1234 + int(s),
                               multi_K=multi_K)
        rows.append(r); detail[str(s)] = r
        if verbose:
            print(f"  goal_err={r['goal_err_mm']:.0f} mm, "
                  f"max_field={r['max_field']:.3f}, "
                  f"speed={r['mean_speed_mps']:.2f} m/s, "
                  f"inference={r['median_inference_us']:.0f} us")

    keys = ['goal_err_mm', 'max_field', 'mean_field', 'path_len_m',
            'mean_speed_mps', 'median_inference_us', 'mean_inference_us']
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows]
        m, sd = _mean_std(vals)
        agg[k] = dict(mean=m, std=sd, sem=_sem(vals), values=vals)
    canonical = dict(
        goal_err_mm=agg['goal_err_mm']['mean'],
        goal_std=agg['goal_err_mm']['std'],
        max_field=agg['max_field']['mean'],
        field_std=agg['max_field']['std'],
        teacher_dev_mm=None,
        dev_std=None,
        path_len_m=agg['path_len_m']['mean'],
        path_std=agg['path_len_m']['std'],
        speed_us=agg['median_inference_us']['mean'],
    )
    out = dict(
        metadata=dict(
            n_seeds=len(SEEDS),
            obstacle_seeds=SEEDS,
            task='8-obstacle Gaussian course, start=(-1.5,-1.5,1) goal=(1.5,1.5,1)',
            controller='Diffusion_BC_planner',
            obs_dim=OBS_DIM,
            net='DiffusionUnetLowdimPolicy (ConditionalUnet1D)',
        ),
        obstacle_course=dict(Diffusion_BC_planner=canonical),
        aggregate=agg, per_seed=detail,
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))
    if verbose:
        print(f"\n[diffusion eval] -> {save_path}")
        print(f"  mean goal_err: {canonical['goal_err_mm']:.0f} +/- "
              f"{canonical['goal_std']:.0f} mm")
        print(f"  mean max_field: {canonical['max_field']:.3f}")
        print(f"  mean inference: {canonical['speed_us']:.0f} us")
    return out


# ---- Main ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='data/planner_dataset_v1.npz')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--window', type=int, default=DEFAULT_HORIZON)
    parser.add_argument('--inference-steps', type=int, default=8)
    parser.add_argument('--down-dims', type=str, default='128,256,512')
    parser.add_argument('--diffusion-embed-dim', type=int, default=128)
    parser.add_argument('--model-out', type=str,
                        default='data/diffusion_student_v1_ema.pt')
    parser.add_argument('--eval-out', type=str,
                        default='results/diffusion_distill_10seed.json')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--multi-K', type=int, default=1,
                        help='K-sample safety filter at inference; '
                             '1 means single sample (original behaviour)')
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[diffusion] device={device}")
    down_dims = tuple(int(x) for x in args.down_dims.split(','))

    # Load dataset
    data_path = _HERE.parent / args.data
    data = np.load(data_path)
    obs = data['observations']; act = data['actions']
    variant_ids = data['variant_ids']; step_indices = data['step_indices']
    print(f"[diffusion] dataset {obs.shape[0]:,} samples, "
          f"{int(np.unique(variant_ids).size):,} rollouts")

    # Build normalizer from full dataset (used regardless of train/eval).
    normalizer = build_normalizer_from_arrays(obs, act)

    # Build policy.
    policy = build_diffusion_policy(
        obs_dim=OBS_DIM, action_dim=ACT_DIM, horizon=args.window,
        n_obs_steps=1, n_action_steps=args.window,
        down_dims=down_dims,
        diffusion_step_embed_dim=args.diffusion_embed_dim,
        num_inference_steps=args.inference_steps,
    )
    policy.set_normalizer(normalizer)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[diffusion] policy params: {n_params:,}")

    if not args.eval_only:
        ds = PlannerWindowDataset(obs, act, variant_ids, step_indices,
                                  window=args.window)
        print(f"[diffusion] sliding windows (W={args.window}): {len(ds):,}")
        policy = policy.to(device)
        log = train_diffusion(policy, ds, device,
                              epochs=args.epochs,
                              batch_size=args.batch_size,
                              lr=args.lr)
        ema_policy = log['ema_policy']
        out_path = _HERE.parent / args.model_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ema_policy.state_dict(), out_path)
        print(f"[diffusion] saved EMA -> {out_path}")
        # Also save the raw policy for comparison/debugging.
        torch.save(policy.state_dict(),
                   str(out_path).replace('_ema.pt', '_raw.pt'))

        # Training plot
        try:
            plot_path = _HERE.parent / 'results' / 'diffusion_training.png'
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            xs = np.arange(len(log['tr_curve']))
            ax.plot(xs, log['tr_curve'], label='train', lw=1.5)
            ax.plot(xs, log['va_curve'], label='EMA val', lw=1.5,
                    linestyle='--')
            ax.set_xlabel('epoch'); ax.set_ylabel('DDPM noise MSE')
            ax.set_yscale('log'); ax.grid(True, alpha=0.3); ax.legend()
            ax.set_title('Diffusion BC training')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"[diffusion] plot -> {plot_path}")
        except Exception as e:
            print(f"[diffusion] plot skipped: {e}")
    else:
        # Load existing model
        out_path = _HERE.parent / args.model_out
        ema_policy = policy
        ema_policy.load_state_dict(torch.load(out_path, map_location='cpu'))
        ema_policy = ema_policy.to(device)
        print(f"[diffusion] loaded -> {out_path}")

    # Switch to DDIM at inference for speed
    ema_policy.noise_scheduler = make_inference_scheduler(
        num_inference_steps=args.inference_steps)
    ema_policy.num_inference_steps = args.inference_steps

    run_10seed_eval(ema_policy, str(_HERE.parent / args.eval_out), device,
                    multi_K=args.multi_K)


if __name__ == '__main__':
    main()
