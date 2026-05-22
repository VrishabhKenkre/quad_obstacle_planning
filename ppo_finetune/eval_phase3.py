"""
ppo_finetune/eval_phase3.py -- phase-3 eval driver.

For each saved checkpoint in `--checkpoint-pattern`, runs the 10-seed
random and 10-seed dp evals (re-uses ppo_finetune.eval_finetuned
primitives). Then:

  - Picks the BEST checkpoint by combined score
        score = mean_goal_err_mm + 100 * dp_p95_max_field
  - Re-runs that best checkpoint, additionally recording per-rollout
    position trajectories.
  - Computes trajectory_efficiency =
        straight_line_distance(start, goal) / actual_path_length
    per rollout, and reports per-seed + aggregate.

Outputs:
  results/diffusion_v2_ppo_phase3_10seed.json (best ckpt random)
  results/diffusion_v2_ppo_phase3_dp.json     (best ckpt dp)
  results/diffusion_v2_ppo_phase3_checkpoint_sweep.json
        (per-checkpoint score table + chosen best ckpt path)
  results/diffusion_v2_ppo_phase3_trajectory_efficiency.json

Usage:
  python -m ppo_finetune.eval_phase3
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'distillation'))
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'external' / 'diffusion_policy'))

from voxelize import VoxelMap
from collect_planner_data import START, GOAL
from train_diffusion import SEEDS as RANDOM_EVAL_SEEDS, run_diffusion_seed
from eval_decision_points import (rollout_diffusion, DEFAULT_DP_SEEDS)
from obstacle_course import make_obstacles
from randomize_astar import decision_point_layout
from ppo_finetune.train_ppo import load_diffusion_policy
from ppo_finetune.eval_finetuned import _stats, eval_random, eval_dp


def find_checkpoints(pattern: str) -> list:
    matches = sorted(glob.glob(str(_ROOT / pattern)))
    rows = []
    for path in matches:
        m = re.search(r'iter(\d+)\.pt$', path)
        if m is None:
            continue
        rows.append((int(m.group(1)), path))
    rows.sort(key=lambda x: x[0])
    return rows


def quick_eval(policy, device, t_max: float = 10.0) -> dict:
    """One-pass random + dp eval; light wrapper around eval_finetuned."""
    print(f"  -> random 10-seed eval")
    rand = eval_random(policy, device, t_max=t_max)
    print(f"  -> dp 10-seed eval")
    dp = eval_dp(policy, device, t_max=t_max)
    return dict(random=rand, dp=dp)


def trajectory_efficiency_for_seed(policy, seed: int, layout: str,
                                    device, t_max: float = 10.0) -> dict:
    """Run one rollout that records the position trajectory; compute
    straight-line / actual-path ratio."""
    if layout == 'random':
        r = run_diffusion_seed(policy, seed, device,
                               diffusion_seed=1234 + int(seed),
                               multi_K=1, T_max=t_max)
        # run_diffusion_seed doesn't return xs by default. Re-roll with the
        # eval_decision_points primitive that does.
        obstacles = make_obstacles(seed=int(seed))
    else:
        obstacles, _, _ = decision_point_layout(seed=int(seed))
    vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
    r = rollout_diffusion(policy, vm, obstacles, device,
                          T_max=t_max, diffusion_seed=1234 + int(seed))
    xs = r['xs']                  # (N, 3) world positions
    seg = np.linalg.norm(np.diff(xs, axis=0), axis=1)
    path_len = float(seg.sum())
    straight = float(np.linalg.norm(np.asarray(GOAL) - np.asarray(START)))
    eff = straight / max(path_len, 1e-9)
    return dict(seed=int(seed), layout=str(layout),
                path_length_m=path_len,
                straight_line_m=straight,
                trajectory_efficiency=float(eff),
                goal_err_mm=float(r['goal_err_mm']),
                max_field=float(r['max_field']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-pattern', type=str,
                    default='data/diffusion_v2_ppo_phase3_iter*.pt')
    ap.add_argument('--data', type=str,
                    default='data/planner_dataset_v2.npz')
    ap.add_argument('--random-out', type=str,
                    default='results/diffusion_v2_ppo_phase3_10seed.json')
    ap.add_argument('--dp-out', type=str,
                    default='results/diffusion_v2_ppo_phase3_dp.json')
    ap.add_argument('--sweep-out', type=str,
                    default='results/diffusion_v2_ppo_phase3_checkpoint_sweep.json')
    ap.add_argument('--eff-out', type=str,
                    default='results/diffusion_v2_ppo_phase3_trajectory_efficiency.json')
    ap.add_argument('--t-max', type=float, default=10.0)
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[phase3-eval] device={device}")

    ckpts = find_checkpoints(args.checkpoint_pattern)
    print(f"[phase3-eval] {len(ckpts)} checkpoints to evaluate: "
          f"iters {[it for it, _ in ckpts]}")

    sweep = []
    best = None
    best_score = float('inf')
    best_eval = None
    for it, path in ckpts:
        print(f"\n[phase3-eval] === iter{it} ({Path(path).name}) ===")
        policy = load_diffusion_policy(path, args.data, device)
        ev = quick_eval(policy, device, t_max=args.t_max)
        rand_goal = ev['random']['aggregate']['goal_err_mm']['mean']
        rand_p95 = ev['random']['aggregate']['max_field']['p95']
        dp_goal = ev['dp']['aggregate']['goal_err_mm']['mean']
        dp_p95 = ev['dp']['aggregate']['max_field']['p95']
        score = float(0.5 * rand_goal + 0.5 * dp_goal + 100.0 * dp_p95)
        row = dict(iter=int(it), checkpoint=str(path),
                   random_goal_mm=float(rand_goal),
                   random_p95_max_field=float(rand_p95),
                   dp_goal_mm=float(dp_goal),
                   dp_p95_max_field=float(dp_p95),
                   score=score)
        sweep.append(row)
        print(f"  iter{it} score={score:.1f}  "
              f"(rand_goal={rand_goal:.0f}mm p95={rand_p95:.3f}; "
              f"dp_goal={dp_goal:.0f}mm p95={dp_p95:.3f})")
        if score < best_score:
            best_score = score
            best = (it, path)
            best_eval = ev

    print(f"\n[phase3-eval] BEST: iter{best[0]} score={best_score:.1f}")
    print(f"  -> {best[1]}")
    Path(_ROOT / args.sweep_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(
        checkpoints=sweep,
        best_iter=int(best[0]),
        best_checkpoint=str(best[1]),
        best_score=float(best_score),
        scoring_rule='0.5*rand_goal + 0.5*dp_goal + 100*dp_p95_max_field',
    ), open(_ROOT / args.sweep_out, 'w'), indent=2,
        default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"[phase3-eval] sweep -> {args.sweep_out}")

    # Save the best ckpt's eval as the canonical phase-3 result.
    with open(_ROOT / args.random_out, 'w') as f:
        json.dump(best_eval['random'], f, indent=2,
                  default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))
    with open(_ROOT / args.dp_out, 'w') as f:
        json.dump(best_eval['dp'], f, indent=2,
                  default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"[phase3-eval] best-ckpt random eval -> {args.random_out}")
    print(f"[phase3-eval] best-ckpt dp eval    -> {args.dp_out}")

    # Trajectory efficiency on the best ckpt -- all 20 eval seeds.
    print(f"\n[phase3-eval] computing trajectory_efficiency on best ckpt")
    policy = load_diffusion_policy(best[1], args.data, device)
    rows = []
    for s in RANDOM_EVAL_SEEDS:
        e = trajectory_efficiency_for_seed(policy, s, 'random',
                                            device, t_max=args.t_max)
        rows.append(e)
        print(f"  random seed {s}: path={e['path_length_m']:.2f}m "
              f"eff={e['trajectory_efficiency']:.3f} "
              f"goal={e['goal_err_mm']:.0f}mm")
    for s in DEFAULT_DP_SEEDS:
        e = trajectory_efficiency_for_seed(policy, s, 'dp',
                                            device, t_max=args.t_max)
        rows.append(e)
        print(f"  dp seed {s}: path={e['path_length_m']:.2f}m "
              f"eff={e['trajectory_efficiency']:.3f} "
              f"goal={e['goal_err_mm']:.0f}mm")
    eff_arr = np.asarray([r['trajectory_efficiency'] for r in rows])
    with open(_ROOT / args.eff_out, 'w') as f:
        json.dump(dict(
            per_seed=rows,
            aggregate=dict(
                mean=float(eff_arr.mean()),
                median=float(np.median(eff_arr)),
                p5=float(np.percentile(eff_arr, 5)),
                p95=float(np.percentile(eff_arr, 95)),
            ),
            best_checkpoint=str(best[1]),
        ), f, indent=2)
    print(f"\n[phase3-eval] trajectory efficiency -> {args.eff_out}")
    print(f"  aggregate mean trajectory_efficiency = {eff_arr.mean():.3f}")
    print(f"  aggregate median trajectory_efficiency = {np.median(eff_arr):.3f}")


if __name__ == '__main__':
    main()
