"""
k_sweep.py -- evaluate the v2 diffusion student at multiple K values for
the multi-sample safety filter, on both the 10-seed decision-point suite
and the 10-seed random obstacle course. Saves a combined JSON and a
Pareto plot (safety vs latency).

Reuses the rollout primitives from `distillation/eval_decision_points.py`
so the K=1 / K=3 numbers in the existing v2 single / v2 multi3 evals are
reproduced exactly when the seeds and torch RNG match.

Usage:
  python distillation/k_sweep.py --Ks 1 3 8 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'external' / 'diffusion_policy'))

from voxelize import VoxelMap
from obstacle_course import make_obstacles

from collect_planner_data import OBS_DIM, ACT_DIM
from randomize_astar import decision_point_layout
from diffusion_student import (build_diffusion_policy,
                                make_inference_scheduler,
                                build_normalizer_from_arrays,
                                DEFAULT_HORIZON)
from eval_decision_points import (rollout_diffusion,
                                   rollout_diffusion_multisample)


RANDOM_SEEDS = [42, 7, 13, 99, 256, 128, 314, 2024, 777, 1337]
DP_SEEDS = list(range(10))


def _stats(values: list, keys=('mean', 'median', 'p95', 'std')) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    out = {}
    if 'mean' in keys:
        out['mean'] = float(arr.mean())
    if 'median' in keys:
        out['median'] = float(np.median(arr))
    if 'p95' in keys:
        out['p95'] = float(np.percentile(arr, 95))
    if 'std' in keys:
        out['std'] = float(arr.std())
    return out


def run_dp_eval(policy, K: int, device, seeds, T_max: float = 10.0) -> dict:
    """Run the diffusion student on the 10 dp seeds at the given K."""
    per_seed_rows = []
    for s in seeds:
        obstacles, _lb, _rb = decision_point_layout(seed=s)
        vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
        if K and K > 1:
            r = rollout_diffusion_multisample(
                policy, vm, obstacles, device,
                T_max=T_max,
                diffusion_seed=1234 + int(s),
                K=int(K))
        else:
            r = rollout_diffusion(
                policy, vm, obstacles, device,
                T_max=T_max,
                diffusion_seed=1234 + int(s))
        per_seed_rows.append({k: v for k, v in r.items() if k != 'xs'})
        print(f"    dp seed {s}: K={K} goal={r['goal_err_mm']:.0f}mm "
              f"max_field={r['max_field']:.3f} "
              f"inf={r['median_inference_us']/1000:.1f}ms")
    return per_seed_rows


def run_random_eval(policy, K: int, device, seeds,
                    T_max: float = 10.0) -> dict:
    """Run the diffusion student on the 10 random obstacle-course seeds."""
    per_seed_rows = []
    for s in seeds:
        obstacles = make_obstacles(seed=int(s))
        vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
        if K and K > 1:
            r = rollout_diffusion_multisample(
                policy, vm, obstacles, device,
                T_max=T_max,
                diffusion_seed=1234 + int(s),
                K=int(K))
        else:
            r = rollout_diffusion(
                policy, vm, obstacles, device,
                T_max=T_max,
                diffusion_seed=1234 + int(s))
        per_seed_rows.append({k: v for k, v in r.items() if k != 'xs'})
        print(f"    random seed {s}: K={K} goal={r['goal_err_mm']:.0f}mm "
              f"max_field={r['max_field']:.3f} "
              f"inf={r['median_inference_us']/1000:.1f}ms")
    return per_seed_rows


def plot_k_sweep(K_list, dp_rows, random_rows, out_path: Path):
    """Two-panel figure: (left) dp max_field median+p95 vs K (log),
    (right) inference latency per timestep (ms) vs K (log).
    """
    Ks = np.asarray(K_list)
    dp_median = np.asarray([r['dp_max_field']['median'] for r in dp_rows])
    dp_p95 = np.asarray([r['dp_max_field']['p95'] for r in dp_rows])
    dp_latency = np.asarray([r['dp_inference_ms']['median'] for r in dp_rows])
    rd_latency = np.asarray([
        r.get('random_inference_ms', {}).get('median', np.nan) for r in dp_rows])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=120)
    axes[0].plot(Ks, dp_median, 'o-', color='#4c78a8', lw=2,
                 markersize=8, label='median max_field')
    axes[0].plot(Ks, dp_p95, 's--', color='#e45756', lw=2,
                 markersize=8, label='p95 max_field')
    axes[0].axhline(0.5, color='gray', linestyle=':', alpha=0.6,
                    label='target p95 < 0.5')
    axes[0].axhline(1.0, color='black', linestyle=':', alpha=0.4,
                    label='inside obstacle (max_field $\\geq$ 1)')
    axes[0].set_xscale('log')
    axes[0].set_xticks(Ks)
    axes[0].set_xticklabels([str(int(k)) for k in Ks])
    axes[0].set_xlabel('K (samples per control step)')
    axes[0].set_ylabel('decision-point max_field')
    axes[0].set_title('Safety vs K')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc='upper right')

    axes[1].plot(Ks, dp_latency, 'o-', color='#54a24b', lw=2,
                 markersize=8, label='dp seeds')
    if not np.all(np.isnan(rd_latency)):
        axes[1].plot(Ks, rd_latency, 's--', color='#f58518', lw=2,
                     markersize=8, label='random seeds')
    axes[1].set_xscale('log')
    axes[1].set_xticks(Ks)
    axes[1].set_xticklabels([str(int(k)) for k in Ks])
    axes[1].set_xlabel('K (samples per control step)')
    axes[1].set_ylabel('median inference latency per step (ms)')
    axes[1].set_title('Latency vs K')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc='upper left')

    fig.suptitle('Pareto frontier: K-sample safety filter on diffusion student v2',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ks', type=int, nargs='+', default=[1, 3, 8, 16])
    ap.add_argument('--model', type=str,
                    default='data/diffusion_student_v2_ema.pt')
    ap.add_argument('--data', type=str,
                    default='data/planner_dataset_v2.npz')
    ap.add_argument('--dp-out', type=str,
                    default='results/decision_point_eval_v2_kSweep.json')
    ap.add_argument('--random-out', type=str,
                    default='results/diffusion_distill_v2_kSweep.json')
    ap.add_argument('--plot', type=str, default='results/k_sweep.png')
    ap.add_argument('--t-max', type=float, default=10.0)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--random-only-best-K', action='store_true', default=True,
                    help='Run random eval only at the best-K choice (default '
                         'true) instead of at every K')
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[k-sweep] device={device}, Ks={args.Ks}")

    data = np.load(_ROOT / args.data)
    norm = build_normalizer_from_arrays(data['observations'], data['actions'])
    policy = build_diffusion_policy(
        obs_dim=OBS_DIM, action_dim=ACT_DIM,
        horizon=DEFAULT_HORIZON, n_obs_steps=1,
        n_action_steps=DEFAULT_HORIZON,
        down_dims=(128, 256, 512),
        diffusion_step_embed_dim=128,
        num_inference_steps=8,
    )
    policy.set_normalizer(norm)
    policy.load_state_dict(torch.load(_ROOT / args.model, map_location='cpu'))
    policy = policy.to(device).eval()
    policy.noise_scheduler = make_inference_scheduler(num_inference_steps=8)
    policy.num_inference_steps = 8

    sweep_rows = []
    dp_per_seed_full = {}
    for K in args.Ks:
        t0 = time.time()
        print(f"\n[k-sweep] K={K} -- dp eval (10 seeds)")
        per_seed = run_dp_eval(policy, K, device, DP_SEEDS, T_max=args.t_max)
        dt = time.time() - t0
        dp_per_seed_full[str(K)] = per_seed
        goal_vals = [r['goal_err_mm'] for r in per_seed]
        field_vals = [r['max_field'] for r in per_seed]
        lat_vals_us = [r['median_inference_us'] for r in per_seed]
        row = dict(
            K=int(K),
            dp_goal_err_mm=_stats(goal_vals),
            dp_max_field=_stats(field_vals),
            dp_inference_ms=_stats([v / 1000.0 for v in lat_vals_us]),
            wall_seconds=float(dt),
        )
        sweep_rows.append(row)
        print(f"[k-sweep] K={K} done in {dt:.1f}s: "
              f"goal={row['dp_goal_err_mm']['mean']:.0f}+/-"
              f"{row['dp_goal_err_mm']['std']:.0f}mm  "
              f"max_field mean/median/p95 = "
              f"{row['dp_max_field']['mean']:.3f} / "
              f"{row['dp_max_field']['median']:.3f} / "
              f"{row['dp_max_field']['p95']:.3f}  "
              f"latency {row['dp_inference_ms']['median']:.1f} ms")

    # Decide recommended K
    candidate = None
    for r in sweep_rows:
        if r['dp_max_field']['p95'] < 0.5:
            candidate = r['K']
            reason = (f"K={candidate} is the smallest K with dp max_field "
                      f"p95 < 0.5 (p95={r['dp_max_field']['p95']:.3f})")
            break
    if candidate is None:
        # Fall back: pick the K with the lowest p95 max_field
        best = min(sweep_rows, key=lambda r: r['dp_max_field']['p95'])
        candidate = best['K']
        reason = (f"K={candidate} is the lowest-p95 K available "
                  f"(p95={best['dp_max_field']['p95']:.3f}); no K in the "
                  f"sweep brought p95 < 0.5, so we report the safest "
                  f"reachable Pareto point.")
    print(f"\n[k-sweep] recommended K = {candidate}  ({reason})")

    # Run random eval at recommended K (and at K=3 as reference baseline)
    print(f"\n[k-sweep] random eval at recommended K = {candidate}")
    random_rows_per_K = {}
    for K_rand in sorted({3, candidate}):
        print(f"\n[k-sweep] random eval K={K_rand}")
        per_seed = run_random_eval(policy, K_rand, device, RANDOM_SEEDS,
                                   T_max=args.t_max)
        random_rows_per_K[str(K_rand)] = per_seed
        # Patch the corresponding sweep row with random latency
        for row in sweep_rows:
            if row['K'] == K_rand:
                row['random_goal_err_mm'] = _stats(
                    [r['goal_err_mm'] for r in per_seed])
                row['random_max_field'] = _stats(
                    [r['max_field'] for r in per_seed])
                row['random_inference_ms'] = _stats(
                    [r['median_inference_us'] / 1000.0 for r in per_seed])
                print(f"  random K={K_rand}: "
                      f"goal={row['random_goal_err_mm']['mean']:.0f}+/-"
                      f"{row['random_goal_err_mm']['std']:.0f}mm  "
                      f"max_field mean/median/p95 = "
                      f"{row['random_max_field']['mean']:.3f} / "
                      f"{row['random_max_field']['median']:.3f} / "
                      f"{row['random_max_field']['p95']:.3f}  "
                      f"latency {row['random_inference_ms']['median']:.1f} ms")

    # Save sweep JSON (dp)
    dp_out = dict(
        K=[int(r['K']) for r in sweep_rows],
        dp_goal_err_mm=[r['dp_goal_err_mm']['mean'] for r in sweep_rows],
        dp_goal_err_std_mm=[r['dp_goal_err_mm']['std'] for r in sweep_rows],
        dp_max_field_mean=[r['dp_max_field']['mean'] for r in sweep_rows],
        dp_max_field_median=[r['dp_max_field']['median'] for r in sweep_rows],
        dp_max_field_p95=[r['dp_max_field']['p95'] for r in sweep_rows],
        inference_latency_ms=[r['dp_inference_ms']['median']
                              for r in sweep_rows],
        recommended_K=int(candidate),
        recommended_K_reason=reason,
        per_seed=dp_per_seed_full,
        sweep_rows=sweep_rows,
    )
    dp_out_path = _ROOT / args.dp_out
    dp_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dp_out_path, 'w') as f:
        json.dump(dp_out, f, indent=2)
    print(f"\n  -> {dp_out_path}")

    # Save random JSON
    random_out = dict(
        Ks_evaluated=[int(k) for k in sorted(random_rows_per_K.keys(),
                                              key=int)],
        recommended_K=int(candidate),
        per_K={str(k): {
            'goal_err_mm': _stats(
                [r['goal_err_mm'] for r in random_rows_per_K[str(k)]]),
            'max_field': _stats(
                [r['max_field'] for r in random_rows_per_K[str(k)]]),
            'inference_ms': _stats(
                [r['median_inference_us'] / 1000.0
                 for r in random_rows_per_K[str(k)]]),
            'per_seed': random_rows_per_K[str(k)],
        } for k in random_rows_per_K},
    )
    rd_out_path = _ROOT / args.random_out
    with open(rd_out_path, 'w') as f:
        json.dump(random_out, f, indent=2)
    print(f"  -> {rd_out_path}")

    plot_k_sweep(args.Ks, sweep_rows, random_rows_per_K, _ROOT / args.plot)


if __name__ == '__main__':
    main()
