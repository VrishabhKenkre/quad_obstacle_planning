"""
hybrid/eval_hybrid.py -- 10 random + 10 dp eval harness for the hybrid
controller, in the same metric format as eval_planner.py.

Random seeds: [42, 7, 13, 99, 256, 128, 314, 2024, 777, 1337]
dp seeds:     range(10)  (same as every previous dp eval)

Per-rollout metrics: goal_err_mm, max_field, tracking_error
(mean/max/p95), path_length, efficiency, effective inference latency.

Outputs:
  results/hybrid_10seed.json
  results/hybrid_dp.json

Usage:
  python -m hybrid.eval_hybrid --tracker data/mlp_tracker_v1.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'distillation'))
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))

from obstacle_course import make_obstacles
from randomize_astar import decision_point_layout

from hybrid.mlp_tracker import MLPTracker
from hybrid.hybrid_ctrl import run_hybrid, START, GOAL

RANDOM_SEEDS = [42, 7, 13, 99, 256, 128, 314, 2024, 777, 1337]
DP_SEEDS = list(range(10))


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return dict(mean=float(arr.mean()), std=float(arr.std()),
                median=float(np.median(arr)),
                p95=float(np.percentile(arr, 95)),
                values=[float(v) for v in arr])


def eval_layout(tracker, layout: str, seeds, dt_ctrl: float,
                dt_sim: float, t_max: float) -> dict:
    per_seed = {}
    for s in seeds:
        if layout == 'random':
            obstacles = make_obstacles(seed=int(s))
            safety_margin = 0.15
        else:
            obstacles, _, _ = decision_point_layout(seed=int(s))
            safety_margin = 0.30
        t0 = time.time()
        try:
            r = run_hybrid(obstacles, tracker, safety_margin=safety_margin,
                           dt_ctrl=dt_ctrl, dt_sim=dt_sim, t_max=t_max,
                           return_trajectory=False)
        except Exception as e:
            print(f"  {layout} seed {s}: FAILED ({e})")
            continue
        per_seed[str(s)] = r
        print(f"  {layout} seed {s:>5}: goal={r['goal_err_mm']:.0f}mm  "
              f"max_field={r['max_field']:.3f}  "
              f"track_err mean/max={r['tracking_error_mean_mm']:.1f}/"
              f"{r['tracking_error_max_mm']:.1f}mm  "
              f"eff_latency={r['effective_latency_us']:.0f}us  "
              f"({time.time()-t0:.1f}s)")
    keys = ('goal_err_mm', 'max_field', 'mean_field', 'path_len_m',
            'efficiency', 'tracking_error_mean_mm', 'tracking_error_max_mm',
            'tracking_error_p95_mm', 'median_tracker_inference_us',
            'amortised_planning_us', 'effective_latency_us')
    rows = list(per_seed.values())
    agg = {k: _stats([r[k] for r in rows]) for k in keys if rows}
    return dict(metadata=dict(seeds=list(seeds), layout=layout,
                              dt_ctrl=dt_ctrl, controller='Hybrid'),
                aggregate=agg, per_seed=per_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', type=str, default='data/mlp_tracker_v1.pt')
    ap.add_argument('--dt-ctrl', type=float, default=0.005,
                    help='control period (s); 0.005 = 200 Hz')
    ap.add_argument('--dt-sim', type=float, default=0.001)
    ap.add_argument('--t-max', type=float, default=10.0)
    ap.add_argument('--random-out', type=str,
                    default='results/hybrid_10seed.json')
    ap.add_argument('--dp-out', type=str, default='results/hybrid_dp.json')
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    tracker = MLPTracker(args.tracker, device=args.device)
    print(f"[eval-hybrid] tracker {args.tracker} "
          f"({tracker.n_params:,} params), dt_ctrl={args.dt_ctrl} "
          f"({1.0/args.dt_ctrl:.0f} Hz)")

    print(f"\n[eval-hybrid] === random 10-seed ===")
    rand = eval_layout(tracker, 'random', RANDOM_SEEDS,
                       args.dt_ctrl, args.dt_sim, args.t_max)
    Path(_ROOT / args.random_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rand, open(_ROOT / args.random_out, 'w'), indent=2)
    print(f"[eval-hybrid] -> {args.random_out}")

    print(f"\n[eval-hybrid] === dp 10-seed ===")
    dp = eval_layout(tracker, 'dp', DP_SEEDS,
                     args.dt_ctrl, args.dt_sim, args.t_max)
    json.dump(dp, open(_ROOT / args.dp_out, 'w'), indent=2)
    print(f"[eval-hybrid] -> {args.dp_out}")

    for label, store in (('random', rand), ('dp', dp)):
        a = store['aggregate']
        if not a:
            continue
        print(f"\n[eval-hybrid] {label} summary:")
        print(f"  goal_err     {a['goal_err_mm']['mean']:.0f} +/- "
              f"{a['goal_err_mm']['std']:.0f} mm")
        print(f"  max_field    mean {a['max_field']['mean']:.3f}  "
              f"p95 {a['max_field']['p95']:.3f}")
        print(f"  tracking err mean {a['tracking_error_mean_mm']['mean']:.1f} mm  "
              f"max {a['tracking_error_max_mm']['mean']:.1f} mm")
        print(f"  eff latency  {a['effective_latency_us']['mean']:.0f} us "
              f"({a['median_tracker_inference_us']['mean']:.0f} us tracker "
              f"+ {a['amortised_planning_us']['mean']:.0f} us amortised plan)")


if __name__ == '__main__':
    main()
