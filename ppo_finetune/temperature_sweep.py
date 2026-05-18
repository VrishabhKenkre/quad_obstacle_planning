"""
ppo_finetune/temperature_sweep.py -- T ablation for phase-2 AWR.

From the iter-5 checkpoint of the main phase-2 run, train three branches
in parallel (sequential here) with T in {0.1, 0.3, 0.5} for 5
iterations each. Compare:
  - mean_return trend
  - mean goal_err trend
  - max_field trends (mean / p95)
  - param drift
and recommend the cleanest-trending T for the phase-2 main loop.

Each branch saves:
  data/diffusion_v2_ppo_phase2_T{T}_iter{1..5}.pt
  results/temperature_sweep_T{T}.json

Plus a combined results/temperature_sweep.json with the recommendation.

Usage:
  python -m ppo_finetune.temperature_sweep \\
      --start-checkpoint data/diffusion_v2_ppo_phase2_iter5.pt \\
      --temperatures 0.1 0.3 0.5 --n-iters 5
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from ppo_finetune.train_ppo import load_diffusion_policy
from ppo_finetune.rollout import collect_rollouts
from ppo_finetune.advantage import estimate_advantages
from ppo_finetune.awr_update import awr_step, _param_l2, param_drift_relative
from ppo_finetune.reward import compute_episode_undiscounted_return
from distillation.diffusion_student import DEFAULT_HORIZON


def run_branch(start_checkpoint: str, data_path: str, T: float,
               n_iters: int, device: torch.device,
               n_rollouts: int = 50, horizon: int = 500,
               awr_steps: int = 100, awr_lr: float = 1e-6,
               value_epochs: int = 10,
               checkpoint_dir: str = 'data',
               checkpoint_prefix: str = 'diffusion_v2_ppo_phase2',
               start_iter: int = 5) -> dict:
    """Train one temperature branch for `n_iters` AWR iterations."""
    policy = load_diffusion_policy(start_checkpoint, data_path, device)
    init_snapshot = copy.deepcopy(policy).to(device).eval()
    for p in init_snapshot.parameters():
        p.requires_grad_(False)
    value_net = None
    rows = []
    t0 = time.time()
    for it in range(1, n_iters + 1):
        episodes = collect_rollouts(
            policy, n_episodes=n_rollouts, horizon=horizon, device=device,
            rng_seed=10_000 + int(T * 100) + it, verbose=False)
        adv = estimate_advantages(
            episodes, device, gamma=0.99, lam=0.95,
            value_epochs=value_epochs, value_net=value_net, verbose=False)
        value_net = adv['value_net']
        actions_all = np.concatenate([e.actions for e in episodes], axis=0)
        upd = awr_step(
            policy, adv['obs'], actions_all, adv['advantages_norm'],
            adv['episode_ends'], device,
            n_grad_steps=awr_steps, batch_size=64, lr=awr_lr,
            temperature=float(T), weight_clip=5.0,
            window=DEFAULT_HORIZON, init_policy=init_snapshot,
            verbose=False)
        ep_returns = [compute_episode_undiscounted_return(e.rewards)
                      for e in episodes]
        row = dict(
            iteration=int(it),
            T=float(T),
            mean_return=float(np.mean(ep_returns)),
            mean_goal_err_mm=float(np.mean([e.goal_err_mm for e in episodes])),
            mean_max_field=float(np.mean([e.max_field for e in episodes])),
            p95_max_field=float(np.percentile(
                [e.max_field for e in episodes], 95)),
            param_drift_relative=float(param_drift_relative(policy, init_snapshot)),
            param_drift_l2=float(_param_l2(policy, init_snapshot)),
        )
        rows.append(row)
        print(f"  [T={T}] iter {it}: return={row['mean_return']:.1f}  "
              f"goal={row['mean_goal_err_mm']:.0f}mm  "
              f"max_field mean/p95={row['mean_max_field']:.3f}/"
              f"{row['p95_max_field']:.3f}  drift_rel="
              f"{row['param_drift_relative']:.4f}")

        ckpt_path = Path(_ROOT / checkpoint_dir
                         / f'{checkpoint_prefix}_T{T:.2f}_iter{start_iter+it}.pt')
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), ckpt_path)
    return dict(T=float(T), iters=rows,
                wallclock_seconds=float(time.time() - t0))


def _trend_score(rows: list) -> float:
    """Monotonic-improvement score: slope of mean_return over iterations.
    Higher (more positive) = cleaner upward trend.
    """
    returns = np.asarray([r['mean_return'] for r in rows], dtype=np.float64)
    if returns.size < 2:
        return 0.0
    xs = np.arange(returns.size, dtype=np.float64)
    slope = float(np.polyfit(xs, returns, 1)[0])
    return slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-checkpoint', type=str,
                    default='data/diffusion_v2_ppo_phase2_iter5.pt')
    ap.add_argument('--data', type=str,
                    default='data/planner_dataset_v2.npz')
    ap.add_argument('--temperatures', type=float, nargs='+',
                    default=[0.1, 0.3, 0.5])
    ap.add_argument('--n-iters', type=int, default=5)
    ap.add_argument('--n-rollouts', type=int, default=50)
    ap.add_argument('--horizon', type=int, default=500)
    ap.add_argument('--awr-steps', type=int, default=100)
    ap.add_argument('--awr-lr', type=float, default=1e-6)
    ap.add_argument('--value-epochs', type=int, default=10)
    ap.add_argument('--out', type=str,
                    default='results/temperature_sweep.json')
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[T-sweep] device={device}, T={args.temperatures}, "
          f"n_iters={args.n_iters}")

    all_branches = []
    for T in args.temperatures:
        print(f"\n[T-sweep] branch T={T}")
        br = run_branch(args.start_checkpoint, args.data, T,
                        args.n_iters, device,
                        n_rollouts=args.n_rollouts, horizon=args.horizon,
                        awr_steps=args.awr_steps, awr_lr=args.awr_lr,
                        value_epochs=args.value_epochs)
        all_branches.append(br)
        # Write per-branch JSON for safety
        with open(_ROOT / args.out.replace('.json', f'_T{T:.2f}.json'),
                  'w') as f:
            json.dump(br, f, indent=2)

    # Rank by trend score
    scored = [(b['T'], _trend_score(b['iters']), b) for b in all_branches]
    scored.sort(key=lambda x: -x[1])   # highest slope first
    best_T = scored[0][0]
    reason = (f"T={best_T} has the highest mean_return slope across "
              f"{args.n_iters} iters (slope={scored[0][1]:.2f}/iter)")
    print(f"\n[T-sweep] recommended T = {best_T}  ({reason})")

    out_path = Path(_ROOT / args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(dict(
            temperatures=args.temperatures,
            recommended_T=float(best_T),
            recommended_reason=reason,
            branches=all_branches,
            trend_scores={str(t): s for t, s, _ in scored},
        ), f, indent=2)
    print(f"[T-sweep] -> {out_path}")


if __name__ == '__main__':
    main()
