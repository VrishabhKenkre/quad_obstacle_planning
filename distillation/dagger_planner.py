"""
dagger_planner.py -- DAgger+DART loop with the hierarchical planner as the
teacher. Reuses Phase 1's plan_once + NMPC tracker to relabel student-
visited states.

Pipeline per iter:
    1. roll out CURRENT student in the env (DART noise sigma * beta_t)
    2. at every visited state, query the planner-teacher for the
       same-step expert action a*_t
    3. append (obs(s_t), a*_t) to the dataset
    4. retrain (warm-start the previous weights) for N_epochs

Adaptive noise schedule: the DART paper uses alpha adapted so the
trajectory deviation has bounded total-variation distance. We approximate
that by halving the noise scale every iteration once the validation MSE
stops improving.

Saves the final policy to data/mlp_student_v1.pt and a training-curve
plot to results/mlp_distill_training.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / 'planning'))
sys.path.insert(0, str(_HERE.parent / 'src'))

from voxelize import VoxelMap
from min_snap import smooth_waypoints
from nonlinear_mpc import SE3_NMPC, rotors_to_mujoco
from quad_env import CrazyflieEnv
from obstacle_course import make_obstacles
from hierarchical_ctrl import _mujoco_state_to_nmpc_state

from randomize_astar import randomized_astar_paths
from collect_planner_data import (make_observation, take_ref_window,
                                  PROBE_OFFSETS, OBS_DIM, ACT_DIM,
                                  START, GOAL, DART_SIGMA)
from mlp_student import MLPStudent


# ---- BC training ---------------------------------------------------------

def train_bc(model: nn.Module, obs: np.ndarray, act: np.ndarray,
             device: torch.device,
             epochs: int = 60, batch_size: int = 256,
             lr: float = 1e-3, val_frac: float = 0.05,
             verbose: bool = True) -> dict:
    """Plain BC training with a held-out validation split. Returns
    a dict with loss curves and the final val loss."""
    n = obs.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_val = max(int(val_frac * n), 256)
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]

    Xtr = torch.from_numpy(obs[tr_idx]).to(device)
    Ytr = torch.from_numpy(act[tr_idx]).to(device)
    Xva = torch.from_numpy(obs[val_idx]).to(device)
    Yva = torch.from_numpy(act[val_idx]).to(device)

    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size,
                        shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    tr_losses = []
    val_losses = []
    grad_norms = []
    best_val = float('inf'); best_state = None
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0; ep_n = 0
        ep_grad = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = mse(pred, yb)
            opt.zero_grad()
            loss.backward()
            gnorm = float(sum((p.grad.detach().norm().item()**2
                               for p in model.parameters()
                               if p.grad is not None)) ** 0.5)
            opt.step()
            ep_loss += float(loss.item()) * xb.shape[0]
            ep_n += xb.shape[0]
            ep_grad += gnorm
        tr_losses.append(ep_loss / max(ep_n, 1))
        grad_norms.append(ep_grad / max(len(loader), 1))
        model.eval()
        with torch.no_grad():
            vloss = float(mse(model(Xva), Yva).item())
        val_losses.append(vloss)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in
                          model.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}: tr={tr_losses[-1]:.4f} "
                  f"val={vloss:.4f} grad_norm~{grad_norms[-1]:.2f}")
    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return dict(tr_losses=tr_losses, val_losses=val_losses,
                grad_norms=grad_norms, best_val=best_val)


# ---- Teacher rollout to relabel student visits ---------------------------

def _build_voxel_for_seed(seed: int) -> Tuple[list, VoxelMap]:
    obstacles = make_obstacles(seed=seed)
    vm = VoxelMap()
    vm.from_obstacle_field(obstacles)
    vm.compute_esdf()
    return obstacles, vm


def _build_ref_for_seed(seed: int, vm: VoxelMap,
                        rng: np.random.Generator,
                        nmpc_dt: float) -> Optional[np.ndarray]:
    paths = randomized_astar_paths(
        START, GOAL, vm, k=1, safety_margin=0.15,
        h_noise=0.15, edge_noise=0.15,
        seed=int(rng.integers(0, 2**31 - 1)))
    if not paths:
        return None
    try:
        ref, _ = smooth_waypoints(paths[0], target_dt=nmpc_dt,
                                  target_avg_speed=0.8, return_meta=True)
    except Exception:
        return None
    return ref


def dagger_rollout(model: nn.Module, env: CrazyflieEnv, nmpc: SE3_NMPC,
                   vm: VoxelMap, ref: np.ndarray,
                   device: torch.device,
                   beta_dart: float,
                   rng: np.random.Generator,
                   nmpc_N: int = 15, nmpc_dt: float = 0.02,
                   T_max: float = 10.0) -> dict:
    """Roll out the STUDENT, label every visited state with the planner's
    NMPC, return (obs, expert_action) pairs.

    Notes:
      * the obs uses the SAME SDF-probe layout as Phase 1 collection
      * the action stepped in the env is the student's noisy action so
        the trajectory drifts off-policy (DART data augmentation)
      * the recorded label is the PLANNER's clean action at the visited
        state, NMPC-tracking the SAME plan
    """
    nmpc.prev_X = None; nmpc.prev_U = None
    state_mj = env.reset(pos=ref[0:3, 0])
    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0

    obss = []
    acts = []
    n_steps = min(int(T_max / nmpc_dt), ref.shape[1])
    aborted = False
    model.eval()
    for i in range(n_steps):
        obs = make_observation(state_mj, GOAL, vm)
        rp_win, rv_win = take_ref_window(ref, i, nmpc_N)
        x13 = _mujoco_state_to_nmpc_state(state_mj)
        # Teacher action (NMPC).
        u_rotors, info = nmpc.solve(x13, rp_win, rv_win)
        u_mj = rotors_to_mujoco(u_rotors)
        expert_a = np.clip((u_mj - u_mid) / u_half, -1.0, 1.0).astype(np.float32)
        # Student action.
        with torch.no_grad():
            inp = torch.from_numpy(obs).to(device).unsqueeze(0)
            student_a = model(inp).cpu().numpy()[0]
        # DART noise on the action that's APPLIED to the env, so the
        # trajectory drifts off-distribution. We always SAVE the teacher
        # label at the visited state.
        noise = rng.normal(0.0, beta_dart * DART_SIGMA).astype(np.float32)
        a_apply = np.clip(student_a + noise, -1.0, 1.0)
        u_apply = u_mid + u_half * a_apply
        u_apply = np.clip(u_apply, env.u_min, env.u_max)
        obss.append(obs)
        acts.append(expert_a)
        state_mj = env.step(u_apply)
        if (np.linalg.norm(state_mj[0:3] - ref[0:3, min(i, ref.shape[1]-1)])
                > 2.0):
            aborted = True
            break
    return dict(observations=np.stack(obss).astype(np.float32),
                actions=np.stack(acts).astype(np.float32),
                n_steps=len(obss), aborted=aborted)


# ---- The whole pipeline --------------------------------------------------

def run_dagger(
        seed_dataset_path: str = 'data/planner_dataset_v1.npz',
        seed_pool_size: int = 200,
        seed_offset: int = 1000,
        n_iters: int = 5,
        episodes_per_iter: int = 40,
        bc_epochs: int = 80,
        dagger_epochs: int = 40,
        out_model: str = 'data/mlp_student_v1.pt',
        device: str | None = None,
        verbose: bool = True,
) -> dict:
    """End-to-end MLP+DAgger+DART training. Initial dataset = the Phase 1
    collection; each iter adds `episodes_per_iter` planner-labelled
    student rollouts on randomly-drawn training seeds.

    The student is trained from scratch on the seed dataset (BC step),
    then iteratively refined via DAgger with on-policy noise (DART).
    """
    device = (torch.device(device) if device is not None
              else torch.device('cuda' if torch.cuda.is_available()
                                 else 'cpu'))
    print(f"[dagger] device={device}")

    # ---- 0. load seed dataset ----------------------------------
    seed_path = _HERE.parent / seed_dataset_path
    data = np.load(seed_path)
    obs = data['observations'].astype(np.float32)
    act = data['actions'].astype(np.float32)
    print(f"[dagger] seed dataset: {obs.shape[0]:,} samples from {seed_path}")

    # ---- 1. model ----------------------------------------------
    model = MLPStudent().to(device)
    print(f"[dagger] MLPStudent params: {model.n_params():,}")

    # ---- 2. BC pretrain ----------------------------------------
    print(f"\n[dagger] BC pretrain on seed dataset ({bc_epochs} epochs)")
    bc_log = train_bc(model, obs, act, device, epochs=bc_epochs,
                      verbose=verbose)
    print(f"[dagger] BC done: best_val={bc_log['best_val']:.4f}")

    # ---- 3. DAgger iters ---------------------------------------
    eval_seeds_excluded = {42, 7, 13, 99, 256, 128, 314, 2024, 777, 1337}
    pool = [s for s in range(seed_offset, seed_offset + seed_pool_size * 2)
            if s not in eval_seeds_excluded][:seed_pool_size]
    rng = np.random.default_rng(0xDA66E2)
    dagger_logs = []
    nmpc_cache = {}
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=0.02)

    for it in range(n_iters):
        beta = max(0.25, 1.0 - 0.15 * it)  # mild decay schedule
        print(f"\n[dagger] iter {it+1}/{n_iters} "
              f"({episodes_per_iter} eps, beta={beta:.2f})")
        new_obs = []
        new_act = []
        ep_seeds = rng.choice(pool, size=episodes_per_iter, replace=False)
        t_iter = time.perf_counter()
        for ep_i, s in enumerate(ep_seeds):
            obstacles, vm = _build_voxel_for_seed(int(s))
            if int(s) not in nmpc_cache:
                nmpc_cache[int(s)] = SE3_NMPC(
                    N=15, dt=0.02, obstacles=obstacles,
                    q_pos=300, q_vel=10, q_quat=20, q_omega=0.1,
                    r_thrust=1e3, w_obs=800.0)
            nmpc = nmpc_cache[int(s)]
            ref = _build_ref_for_seed(int(s), vm, rng, 0.02)
            if ref is None:
                continue
            ro = dagger_rollout(model, env, nmpc, vm, ref, device,
                                beta_dart=beta, rng=rng)
            new_obs.append(ro['observations'])
            new_act.append(ro['actions'])
        if not new_obs:
            print("  no new samples this iter, skipping retrain")
            continue
        new_obs = np.concatenate(new_obs, axis=0)
        new_act = np.concatenate(new_act, axis=0)
        obs = np.concatenate([obs, new_obs], axis=0)
        act = np.concatenate([act, new_act], axis=0)
        t_collect = time.perf_counter() - t_iter
        print(f"  collected {new_obs.shape[0]:,} new samples "
              f"({t_collect:.1f}s); total dataset {obs.shape[0]:,}")
        log = train_bc(model, obs, act, device, epochs=dagger_epochs,
                       verbose=verbose)
        log['n_new'] = int(new_obs.shape[0])
        log['n_total'] = int(obs.shape[0])
        log['beta_dart'] = float(beta)
        dagger_logs.append(log)
        print(f"  retrain done: val={log['best_val']:.4f}")
        # Free NMPC cache occasionally to keep RAM bounded.
        if len(nmpc_cache) > 50:
            nmpc_cache.clear()

    # ---- 4. save ----------------------------------------------
    out_path = _HERE.parent / out_model
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"\n[dagger] saved -> {out_path}")

    # Training-curve plot.
    try:
        plot_path = _HERE.parent / 'results' / 'mlp_distill_training.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        cur = 0
        for tag, log in [('BC', bc_log)] + [
                (f'D{i+1}', dagger_logs[i]) for i in range(len(dagger_logs))]:
            tr = np.array(log['tr_losses']); val = np.array(log['val_losses'])
            xs = np.arange(cur, cur + len(tr))
            ax.plot(xs, tr, label=f'{tag} tr', lw=1.0)
            ax.plot(xs, val, label=f'{tag} val', lw=1.0, linestyle='--')
            cur += len(tr)
        ax.set_xlabel('cumulative epoch'); ax.set_ylabel('MSE loss')
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        ax.set_title('MLP+DAgger training curves')

        ax = axes[1]
        cur = 0
        for tag, log in [('BC', bc_log)] + [
                (f'D{i+1}', dagger_logs[i]) for i in range(len(dagger_logs))]:
            g = np.array(log['grad_norms'])
            xs = np.arange(cur, cur + len(g))
            ax.plot(xs, g, label=tag, lw=1.0)
            cur += len(g)
        ax.set_xlabel('cumulative epoch'); ax.set_ylabel('grad norm')
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        ax.set_title('Gradient norm')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[dagger] training plot -> {plot_path}")
    except Exception as e:
        print(f"[dagger] plot skipped: {e}")

    return dict(model=model, bc_log=bc_log, dagger_logs=dagger_logs,
                model_path=str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--eps-per-iter', type=int, default=40)
    parser.add_argument('--bc-epochs', type=int, default=80)
    parser.add_argument('--dagger-epochs', type=int, default=40)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='small smoke run')
    args = parser.parse_args()
    if args.quick:
        args.iters = 2
        args.eps_per_iter = 8
        args.bc_epochs = 20
        args.dagger_epochs = 15
    run_dagger(n_iters=args.iters,
               episodes_per_iter=args.eps_per_iter,
               bc_epochs=args.bc_epochs,
               dagger_epochs=args.dagger_epochs,
               device=args.device,
               verbose=True)


if __name__ == '__main__':
    main()
