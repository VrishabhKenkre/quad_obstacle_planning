"""
ppo_phase3_visual_audit.py -- rank all 20 eval rollouts of the PPO
phase-3 iter15 checkpoint by a "visual cleanness" score, so the
cleanest / ugliest can be rendered for a ship-vs-hybrid decision.

The existing phase-3 eval JSONs store only per-seed scalars (goal_err,
max_field) -- not per-step trajectories -- so this script re-executes
each rollout deterministically (torch.manual_seed(1234 + obstacle_seed),
the same convention as every other eval in the repo) and captures the
position trajectory xs.

Per-rollout metrics:
  final_goal_err_mm, max_field_along_rollout, max_alt_excursion,
  overshoot_amount, mean_path_height, efficiency, terminal_speed_m_s

Cleanness score (higher = cleaner):
  cleanness = -4.0 * max(0, max_field - 0.30)
              -1.0 * (final_goal_err_mm / 100)
              -2.0 * max(0, max_alt_excursion - 0.20)
              -3.0 * overshoot_amount
              +2.0 * efficiency
              -1.0 * terminal_speed_m_s

Output: results/ppo_phase3_visual_audit.json

Usage:
  python distillation/ppo_phase3_visual_audit.py
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
from obstacle_course import make_obstacles, obstacle_field_value
from collect_planner_data import OBS_DIM, ACT_DIM, START, GOAL
from diffusion_student import (build_diffusion_policy,
                                make_inference_scheduler,
                                build_normalizer_from_arrays,
                                DEFAULT_HORIZON)
from train_diffusion import SEEDS as RANDOM_EVAL_SEEDS
from eval_decision_points import rollout_diffusion, DEFAULT_DP_SEEDS
from randomize_astar import decision_point_layout

DT_CTRL = 0.02


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


def rollout_metrics(xs: np.ndarray, obstacles) -> dict:
    """xs: (N,3) post-step positions."""
    goal = np.asarray(GOAL, dtype=np.float64)
    start = np.asarray(START, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)

    final_goal_err_mm = float(np.linalg.norm(xs[-1] - goal) * 1000.0)

    fields = np.asarray([obstacle_field_value(p, obstacles) for p in xs])
    max_field = float(fields.max())

    max_alt_excursion = float(xs[:, 2].max() - goal[2])

    direction = goal - start
    dn = float(np.linalg.norm(direction))
    if dn < 1e-6:
        overshoot = 0.0
    else:
        unit = direction / dn
        proj = (xs - goal[None, :]) @ unit
        overshoot = float(np.maximum(0.0, proj).max())

    # mean path height relative to obstacle midheight (= goal_z = 1.0,
    # the start/goal plane the obstacles straddle)
    mean_path_height = float(xs[:, 2].mean() - goal[2])

    seg = np.linalg.norm(np.diff(xs, axis=0), axis=1)
    path_len = float(seg.sum())
    straight = float(np.linalg.norm(goal - start))
    efficiency = float(straight / max(path_len, 1e-9))

    # terminal speed from the last position delta
    if xs.shape[0] >= 2:
        terminal_speed = float(np.linalg.norm(xs[-1] - xs[-2]) / DT_CTRL)
    else:
        terminal_speed = 0.0

    return dict(
        final_goal_err_mm=final_goal_err_mm,
        max_field_along_rollout=max_field,
        max_alt_excursion=max_alt_excursion,
        overshoot_amount=overshoot,
        mean_path_height=mean_path_height,
        efficiency=efficiency,
        terminal_speed_m_s=terminal_speed,
        path_length_m=path_len,
    )


def cleanness_score(m: dict) -> float:
    return float(
        -4.0 * max(0.0, m['max_field_along_rollout'] - 0.30)
        - 1.0 * (m['final_goal_err_mm'] / 100.0)
        - 2.0 * max(0.0, m['max_alt_excursion'] - 0.20)
        - 3.0 * m['overshoot_amount']
        + 2.0 * m['efficiency']
        - 1.0 * m['terminal_speed_m_s']
    )


def run_one(policy, seed: int, seed_type: str, device,
            t_max: float = 10.0) -> dict:
    if seed_type == 'random':
        obstacles = make_obstacles(seed=int(seed))
    else:
        obstacles, _, _ = decision_point_layout(seed=int(seed))
    vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
    r = rollout_diffusion(policy, vm, obstacles, device,
                          T_max=t_max, diffusion_seed=1234 + int(seed))
    xs = np.asarray(r['xs'], dtype=np.float64)
    m = rollout_metrics(xs, obstacles)
    score = cleanness_score(m)
    return dict(seed=int(seed), seed_type=seed_type,
                metrics=m, cleanness_score=score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str,
                    default='data/diffusion_v2_ppo_phase3_iter15.pt')
    ap.add_argument('--data', type=str,
                    default='data/planner_dataset_v2.npz')
    ap.add_argument('--out', type=str,
                    default='results/ppo_phase3_visual_audit.json')
    ap.add_argument('--t-max', type=float, default=10.0)
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[audit] device={device}")
    policy = load_policy(args.model, args.data, device)
    print(f"[audit] loaded {args.model}")

    rollouts = []
    for seed in RANDOM_EVAL_SEEDS:
        t0 = time.time()
        r = run_one(policy, seed, 'random', device, t_max=args.t_max)
        rollouts.append(r)
        m = r['metrics']
        print(f"  random seed {seed:>5}: clean={r['cleanness_score']:+.3f}  "
              f"goal={m['final_goal_err_mm']:.0f}mm  "
              f"maxf={m['max_field_along_rollout']:.3f}  "
              f"alt_exc={m['max_alt_excursion']*100:+.0f}cm  "
              f"overshoot={m['overshoot_amount']*100:.1f}cm  "
              f"eff={m['efficiency']:.3f}  "
              f"({time.time()-t0:.1f}s)")
    for seed in DEFAULT_DP_SEEDS:
        t0 = time.time()
        r = run_one(policy, seed, 'dp', device, t_max=args.t_max)
        rollouts.append(r)
        m = r['metrics']
        print(f"  dp     seed {seed:>5}: clean={r['cleanness_score']:+.3f}  "
              f"goal={m['final_goal_err_mm']:.0f}mm  "
              f"maxf={m['max_field_along_rollout']:.3f}  "
              f"alt_exc={m['max_alt_excursion']*100:+.0f}cm  "
              f"overshoot={m['overshoot_amount']*100:.1f}cm  "
              f"eff={m['efficiency']:.3f}  "
              f"({time.time()-t0:.1f}s)")

    ranked = sorted(rollouts, key=lambda r: -r['cleanness_score'])
    out = dict(
        checkpoint=str(args.model),
        scoring_rule=('cleanness = -4*max(0,maxf-0.30) '
                      '-1*(goal_mm/100) -2*max(0,alt_exc-0.20) '
                      '-3*overshoot +2*efficiency -1*terminal_speed'),
        rollouts=rollouts,
        ranked=ranked,
    )
    Path(_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(_ROOT / args.out, 'w'), indent=2)
    print(f"\n[audit] -> {args.out}")

    print(f"\n[audit] === ranked by cleanness ===")
    for i, r in enumerate(ranked):
        print(f"  {i+1:>2}. {r['seed_type']:>6} seed {r['seed']:>5}  "
              f"clean={r['cleanness_score']:+.3f}")


if __name__ == '__main__':
    main()
