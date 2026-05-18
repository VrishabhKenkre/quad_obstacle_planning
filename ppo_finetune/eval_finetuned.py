"""
ppo_finetune/eval_finetuned.py -- evaluate a fine-tuned diffusion
checkpoint on the 10 random and 10 dp seeds used by the BC baseline.

Reuses the existing single-seed rollout primitives so the output JSONs
are bit-for-bit comparable to the BC baseline files. K=1 (single
sample); the multi-sample safety filter is orthogonal to fine-tuning
and is left for phase 2.
"""
from __future__ import annotations

import argparse
import json
import sys
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
from obstacle_course import make_obstacles
from collect_planner_data import OBS_DIM, ACT_DIM
from diffusion_student import (build_diffusion_policy, make_inference_scheduler,
                                build_normalizer_from_arrays, DEFAULT_HORIZON)
from train_diffusion import SEEDS as RANDOM_EVAL_SEEDS
from eval_decision_points import (rollout_diffusion, DEFAULT_DP_SEEDS)
from randomize_astar import decision_point_layout

from ppo_finetune.train_ppo import load_diffusion_policy
from ppo_finetune.rollout import rollout_one_episode  # not used here but kept for symmetry


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return dict(mean=float(arr.mean()), std=float(arr.std()),
                median=float(np.median(arr)),
                p95=float(np.percentile(arr, 95)),
                values=[float(v) for v in arr])


def eval_random(policy, device, t_max: float = 10.0) -> dict:
    from train_diffusion import run_diffusion_seed
    rows = []
    detail = {}
    for s in RANDOM_EVAL_SEEDS:
        r = run_diffusion_seed(policy, s, device,
                               diffusion_seed=1234 + int(s), multi_K=1,
                               T_max=t_max)
        rows.append(r); detail[str(s)] = r
        print(f"    random seed {s}: goal={r['goal_err_mm']:.0f}mm  "
              f"max_field={r['max_field']:.3f}  "
              f"inf={r['median_inference_us']/1000:.1f}ms")
    keys = ['goal_err_mm', 'max_field', 'mean_field', 'path_len_m',
            'mean_speed_mps', 'median_inference_us']
    agg = {k: _stats([r[k] for r in rows]) for k in keys}
    return dict(metadata=dict(seeds=RANDOM_EVAL_SEEDS,
                              T_max=t_max,
                              controller='Diffusion_v2_PPO_phase1'),
                aggregate=agg, per_seed=detail)


def eval_dp(policy, device, t_max: float = 10.0) -> dict:
    rows = []
    detail = {}
    for s in DEFAULT_DP_SEEDS:
        obstacles, _, _ = decision_point_layout(seed=int(s))
        vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
        r = rollout_diffusion(policy, vm, obstacles, device,
                              T_max=t_max, diffusion_seed=1234 + int(s))
        r = {k: v for k, v in r.items() if k != 'xs'}
        rows.append(r); detail[str(s)] = r
        print(f"    dp seed {s}: goal={r['goal_err_mm']:.0f}mm  "
              f"max_field={r['max_field']:.3f}  "
              f"inf={r['median_inference_us']/1000:.1f}ms")
    keys = ['goal_err_mm', 'max_field', 'mean_field', 'path_len_m',
            'mean_speed_mps', 'median_inference_us']
    agg = {k: _stats([r[k] for r in rows]) for k in keys}
    return dict(metadata=dict(seeds=DEFAULT_DP_SEEDS,
                              T_max=t_max,
                              controller='Diffusion_v2_PPO_phase1'),
                aggregate=agg, per_seed=detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str,
                    default='data/diffusion_v2_ppo_phase2_iter15.pt')
    ap.add_argument('--data', type=str,
                    default='data/planner_dataset_v2.npz')
    ap.add_argument('--random-out', type=str,
                    default='results/diffusion_v2_ppo_phase2_10seed.json')
    ap.add_argument('--dp-out', type=str,
                    default='results/diffusion_v2_ppo_phase2_dp.json')
    ap.add_argument('--t-max', type=float, default=10.0)
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[ppo-eval] device={device}")
    policy = load_diffusion_policy(args.checkpoint, args.data, device)
    print(f"[ppo-eval] loaded checkpoint: {args.checkpoint}")

    print("\n[ppo-eval] random 10-seed eval")
    random_results = eval_random(policy, device, t_max=args.t_max)
    Path(_ROOT / args.random_out).parent.mkdir(parents=True, exist_ok=True)
    with open(_ROOT / args.random_out, 'w') as f:
        json.dump(random_results, f, indent=2,
                  default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\n[ppo-eval] -> {args.random_out}")
    r_agg = random_results['aggregate']
    print(f"  random: goal={r_agg['goal_err_mm']['mean']:.0f}+/-"
          f"{r_agg['goal_err_mm']['std']:.0f}mm  "
          f"max_field mean/median/p95 = "
          f"{r_agg['max_field']['mean']:.3f} / "
          f"{r_agg['max_field']['median']:.3f} / "
          f"{r_agg['max_field']['p95']:.3f}")

    print("\n[ppo-eval] dp 10-seed eval")
    dp_results = eval_dp(policy, device, t_max=args.t_max)
    with open(_ROOT / args.dp_out, 'w') as f:
        json.dump(dp_results, f, indent=2,
                  default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\n[ppo-eval] -> {args.dp_out}")
    d_agg = dp_results['aggregate']
    print(f"  dp: goal={d_agg['goal_err_mm']['mean']:.0f}+/-"
          f"{d_agg['goal_err_mm']['std']:.0f}mm  "
          f"max_field mean/median/p95 = "
          f"{d_agg['max_field']['mean']:.3f} / "
          f"{d_agg['max_field']['median']:.3f} / "
          f"{d_agg['max_field']['p95']:.3f}")


if __name__ == '__main__':
    main()
