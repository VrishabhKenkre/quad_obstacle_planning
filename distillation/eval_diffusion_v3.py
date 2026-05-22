"""
distillation/eval_diffusion_v3.py -- complete v3 BC evaluation.

Runs:
  random 10-seed @ K=1
  random 10-seed @ K=3
  dp 10-seed @ K=1
  dp 10-seed @ K=3

For each rollout records per-step positions and computes the extra
metrics requested by the brief:
  trajectory_efficiency = straight_line_dist / actual_path_length
  overshoot_amount = max_t( (p_t - goal) . unit(goal - start) )  (>=0)
  max_alt_excursion  = max_t( p_t.z - goal.z )

Outputs:
  results/diffusion_distill_v3_10seed.json     (random, K=1 + K=3, with per-rollout metrics)
  results/diffusion_distill_v3_dp.json          (dp,     K=1 + K=3, with per-rollout metrics)

Usage:
  python distillation/eval_diffusion_v3.py \\
      --model data/diffusion_student_v3_ema.pt \\
      --data  data/planner_dataset_v3.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'external' / 'diffusion_policy'))

from voxelize import VoxelMap
from obstacle_course import make_obstacles
from collect_planner_data import OBS_DIM, ACT_DIM, START, GOAL
from diffusion_student import (build_diffusion_policy,
                                make_inference_scheduler,
                                build_normalizer_from_arrays,
                                DEFAULT_HORIZON)
from train_diffusion import SEEDS as RANDOM_EVAL_SEEDS
from eval_decision_points import (rollout_diffusion,
                                   rollout_diffusion_multisample,
                                   DEFAULT_DP_SEEDS)
from randomize_astar import decision_point_layout


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return dict(mean=float(arr.mean()),
                std=float(arr.std()),
                median=float(np.median(arr)),
                p95=float(np.percentile(arr, 95)),
                values=[float(v) for v in arr])


def _traj_metrics(xs: np.ndarray) -> dict:
    """xs: (N,3) post-step positions stored along the rollout."""
    goal = np.asarray(GOAL, dtype=np.float64)
    start = np.asarray(START, dtype=np.float64)
    seg = np.linalg.norm(np.diff(xs, axis=0), axis=1)
    path_len = float(seg.sum())
    straight = float(np.linalg.norm(goal - start))
    eff = float(straight / max(path_len, 1e-9))
    direction = goal - start
    dir_n = float(np.linalg.norm(direction))
    if dir_n < 1e-6:
        overshoot = 0.0
    else:
        unit = direction / dir_n
        projections = (xs - goal[None, :]) @ unit
        overshoot = float(np.maximum(0.0, projections).max())
    max_alt = float((xs[:, 2] - goal[2]).max())
    return dict(path_length_m=path_len,
                straight_line_m=straight,
                trajectory_efficiency=eff,
                overshoot_amount_m=overshoot,
                max_alt_excursion_m=max_alt)


def load_policy(model_path: str, data_path: str, device):
    data = np.load(_ROOT / data_path)
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
    policy.load_state_dict(torch.load(_ROOT / model_path, map_location='cpu'))
    policy = policy.to(device).eval()
    policy.noise_scheduler = make_inference_scheduler(num_inference_steps=8)
    policy.num_inference_steps = 8
    return policy


def eval_one_seed_random(policy, seed: int, device, K: int = 1,
                         T_max: float = 10.0) -> dict:
    obstacles = make_obstacles(seed=int(seed))
    vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
    if K and K > 1:
        r = rollout_diffusion_multisample(
            policy, vm, obstacles, device,
            T_max=T_max, diffusion_seed=1234 + int(seed), K=int(K))
    else:
        r = rollout_diffusion(
            policy, vm, obstacles, device,
            T_max=T_max, diffusion_seed=1234 + int(seed))
    xs = r.pop('xs')
    r.update(_traj_metrics(xs))
    return r


def eval_one_seed_dp(policy, seed: int, device, K: int = 1,
                     T_max: float = 10.0) -> dict:
    obstacles, _, _ = decision_point_layout(seed=int(seed))
    vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
    if K and K > 1:
        r = rollout_diffusion_multisample(
            policy, vm, obstacles, device,
            T_max=T_max, diffusion_seed=1234 + int(seed), K=int(K))
    else:
        r = rollout_diffusion(
            policy, vm, obstacles, device,
            T_max=T_max, diffusion_seed=1234 + int(seed))
    xs = r.pop('xs')
    r.update(_traj_metrics(xs))
    return r


def run_eval(policy, device, layout: str, K: int) -> dict:
    seeds = RANDOM_EVAL_SEEDS if layout == 'random' else DEFAULT_DP_SEEDS
    eval_fn = eval_one_seed_random if layout == 'random' else eval_one_seed_dp
    rows = {}
    for s in seeds:
        r = eval_fn(policy, s, device, K=K)
        rows[str(s)] = r
        print(f"  {layout} K={K} seed {s}: goal={r['goal_err_mm']:.0f}mm "
              f"maxf={r['max_field']:.3f} "
              f"eff={r['trajectory_efficiency']:.3f} "
              f"overshoot={r['overshoot_amount_m']*100:.1f}cm "
              f"maxz={r['max_alt_excursion_m']*100:+.1f}cm")
    keys = ('goal_err_mm', 'max_field', 'mean_field',
            'path_length_m', 'trajectory_efficiency',
            'overshoot_amount_m', 'max_alt_excursion_m')
    agg = {k: _stats([rows[s][k] for s in rows]) for k in keys}
    return dict(metadata=dict(seeds=list(seeds), K=int(K), layout=layout),
                aggregate=agg, per_seed=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str,
                    default='data/diffusion_student_v3_ema.pt')
    ap.add_argument('--data', type=str,
                    default='data/planner_dataset_v3.npz')
    ap.add_argument('--random-out', type=str,
                    default='results/diffusion_distill_v3_10seed.json')
    ap.add_argument('--dp-out', type=str,
                    default='results/diffusion_distill_v3_dp.json')
    ap.add_argument('--Ks', type=int, nargs='+', default=[1, 3])
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()
    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[v3-eval] device={device}")
    policy = load_policy(args.model, args.data, device)
    print(f"[v3-eval] loaded {args.model}")

    out_random = dict(metadata=dict(seeds=list(RANDOM_EVAL_SEEDS),
                                     controller='Diffusion_BC_v3'),
                       results_by_K={})
    for K in args.Ks:
        t0 = time.time()
        print(f"\n[v3-eval] === random K={K} ===")
        out_random['results_by_K'][str(K)] = run_eval(
            policy, device, 'random', K)
        print(f"  random K={K} done in {time.time()-t0:.1f}s")
    Path(_ROOT / args.random_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_random, open(_ROOT / args.random_out, 'w'), indent=2,
              default=lambda o: float(o)
              if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\n[v3-eval] -> {args.random_out}")

    out_dp = dict(metadata=dict(seeds=list(DEFAULT_DP_SEEDS),
                                 controller='Diffusion_BC_v3'),
                   results_by_K={})
    for K in args.Ks:
        t0 = time.time()
        print(f"\n[v3-eval] === dp K={K} ===")
        out_dp['results_by_K'][str(K)] = run_eval(policy, device, 'dp', K)
        print(f"  dp K={K} done in {time.time()-t0:.1f}s")
    json.dump(out_dp, open(_ROOT / args.dp_out, 'w'), indent=2,
              default=lambda o: float(o)
              if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"[v3-eval] -> {args.dp_out}")

    # Headline summary
    print(f"\n[v3-eval] === SUMMARY ===")
    for layout, store in (('random', out_random), ('dp', out_dp)):
        for K, r in store['results_by_K'].items():
            a = r['aggregate']
            print(f"  {layout:>6} K={K}: "
                  f"goal={a['goal_err_mm']['mean']:.0f}+/-"
                  f"{a['goal_err_mm']['std']:.0f}mm  "
                  f"max_field m/p95={a['max_field']['mean']:.3f}/"
                  f"{a['max_field']['p95']:.3f}  "
                  f"eff={a['trajectory_efficiency']['mean']:.3f}  "
                  f"overshoot={100*a['overshoot_amount_m']['mean']:.1f}cm  "
                  f"alt_excursion={100*a['max_alt_excursion_m']['mean']:+.1f}cm")


if __name__ == '__main__':
    main()
