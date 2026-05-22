"""
dagger_tracker.py -- DAgger iterations for the hybrid controller's MLP
reference tracker.

A pure behavioural clone of the NMPC tracker lags a moving reference:
the drone falls progressively behind ref[:, i], and on the harder
decision-point references the lag compounds until the drone leaves the
training distribution and diverges. DAgger (Ross et al. 2011) fixes
this: roll out the *current* tracker, visit the lagging states it
actually produces, and relabel each with the NMPC expert's action
(which says "accelerate hard to catch up"). Retrain on the aggregated
dataset; repeat.

Each iteration:
  1. For each seed: plan once -> reference.
  2. Roll out the current MLP tracker (time-indexed reference). At each
     step record the 27-D observation, then solve the NMPC at that
     (tracker-visited) state for the relabel action.
  3. Aggregate (obs, NMPC_action) with the existing dataset.
  4. Retrain the tracker from scratch on the aggregated set.

Outputs (per iteration N):
  data/tracking_dataset_dagger{N}.npz   (aggregated dataset)
  data/mlp_tracker_v1.pt                (overwritten with the latest)

Usage:
  python distillation/dagger_tracker.py --iters 2 --n-random 120 --n-dp 30
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
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))

from quad_env import CrazyflieEnv
from obstacle_course import make_obstacles
from nonlinear_mpc import SE3_NMPC, rotors_to_mujoco
from hierarchical_ctrl import (plan_once, _take_ref_window,
                                _mujoco_state_to_nmpc_state)
from randomize_astar import decision_point_layout

from hybrid.mlp_tracker import MLPTracker, make_tracking_obs
from distillation.collect_tracking_data import (
    START, GOAL, TRACK_OBS_DIM, ACT_DIM, NMPC_N, _u_limits,
    RANDOM_COLLECT_SEEDS, DP_COLLECT_SEEDS)
from distillation.train_tracker import train


def dagger_rollout(seed: int, seed_type: str, tracker: MLPTracker,
                   u_mid, u_half, u_min, u_max,
                   nmpc_dt: float = 0.02, t_max: float = 10.0):
    """Roll out the current tracker; relabel every visited state with the
    NMPC expert. Returns (obs (M,27), actions (M,4))."""
    if seed_type == 'random':
        obstacles = make_obstacles(seed=int(seed))
        safety_margin = 0.15
    else:
        obstacles, _, _ = decision_point_layout(seed=int(seed))
        safety_margin = 0.30
    try:
        ref, _meta = plan_once(START, GOAL, obstacles,
                               safety_margin=safety_margin,
                               ref_dt=nmpc_dt, avg_speed=0.8)
    except Exception as e:
        print(f"    [skip] seed {seed} ({seed_type}): {e}")
        return None

    nmpc = SE3_NMPC(N=NMPC_N, dt=nmpc_dt, obstacles=obstacles,
                    q_pos=300, q_vel=10, q_quat=20, q_omega=0.1,
                    r_thrust=1e3, w_obs=800.0)
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=nmpc_dt)
    x_mj = env.reset(pos=START)
    n_steps = int(t_max / nmpc_dt)
    n_ref = ref.shape[1]

    obs_list = np.empty((n_steps, TRACK_OBS_DIM), dtype=np.float32)
    act_list = np.empty((n_steps, ACT_DIM), dtype=np.float32)
    for i in range(n_steps):
        ref_col = ref[:, min(i, n_ref - 1)]
        # --- expert relabel: NMPC action at the tracker-visited state ---
        rp_win, rv_win = _take_ref_window(ref, i, NMPC_N)
        x13 = _mujoco_state_to_nmpc_state(x_mj)
        u_rotors, _info = nmpc.solve(x13, rp_win, rv_win)
        u_mj = rotors_to_mujoco(u_rotors)
        action_expert = np.clip((u_mj - u_mid) / u_half,
                                -1.0, 1.0).astype(np.float32)
        obs_list[i] = make_tracking_obs(x_mj, ref_col)
        act_list[i] = action_expert
        # --- step the env with the TRACKER's action (on-policy) ---
        a_tracker = tracker.predict(x_mj, ref_col)
        u_apply = np.clip(u_mid + u_half * a_tracker, u_min, u_max)
        x_mj = env.step(u_apply)
    return dict(obs=obs_list, actions=act_list,
                goal_err_mm=float(np.linalg.norm(x_mj[0:3] - GOAL) * 1000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=2)
    ap.add_argument('--n-random', type=int, default=120)
    ap.add_argument('--n-dp', type=int, default=30)
    ap.add_argument('--base-data', type=str,
                    default='data/tracking_dataset_v1.npz')
    ap.add_argument('--model', type=str, default='data/mlp_tracker_v1.pt')
    ap.add_argument('--nmpc-dt', type=float, default=0.02)
    ap.add_argument('--t-max', type=float, default=10.0)
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    u_mid, u_half, u_min, u_max = _u_limits()
    random_seeds = RANDOM_COLLECT_SEEDS[:args.n_random]
    dp_seeds = DP_COLLECT_SEEDS[:args.n_dp]
    plan = ([(s, 'random') for s in random_seeds]
            + [(s, 'dp') for s in dp_seeds])

    base = np.load(_ROOT / args.base_data)
    agg_obs = [base['obs']]
    agg_act = [base['actions']]
    print(f"[dagger] base dataset: {base['obs'].shape[0]:,} samples")

    history = []
    for it in range(1, args.iters + 1):
        print(f"\n[dagger] === iteration {it}/{args.iters} === "
              f"({len(plan)} rollouts)")
        tracker = MLPTracker(args.model, device=args.device)
        t0 = time.time()
        it_obs, it_act = [], []
        goal_errs = []
        for idx, (seed, stype) in enumerate(plan):
            r = dagger_rollout(seed, stype, tracker, u_mid, u_half,
                               u_min, u_max, nmpc_dt=args.nmpc_dt,
                               t_max=args.t_max)
            if r is None:
                continue
            it_obs.append(r['obs'])
            it_act.append(r['actions'])
            goal_errs.append(r['goal_err_mm'])
            if (idx + 1) % 25 == 0 or idx == len(plan) - 1:
                print(f"    {idx+1}/{len(plan)} rollouts  "
                      f"({time.time()-t0:.0f}s, tracker goal "
                      f"mean so far {np.mean(goal_errs):.0f}mm)")
        new_obs = np.concatenate(it_obs, axis=0)
        new_act = np.concatenate(it_act, axis=0)
        agg_obs.append(new_obs)
        agg_act.append(new_act)
        obs_all = np.concatenate(agg_obs, axis=0)
        act_all = np.concatenate(agg_act, axis=0)
        out_npz = _ROOT / f'data/tracking_dataset_dagger{it}.npz'
        np.savez_compressed(out_npz, obs=obs_all, actions=act_all,
                            seed=np.zeros(obs_all.shape[0], dtype=np.int32),
                            step_idx=np.zeros(obs_all.shape[0], dtype=np.int32))
        print(f"[dagger] iter {it}: +{new_obs.shape[0]:,} on-policy "
              f"samples (tracker goal mean {np.mean(goal_errs):.0f}mm); "
              f"aggregated {obs_all.shape[0]:,}")

        # retrain from scratch on the aggregated dataset
        stats = train(out_npz, _ROOT / args.model,
                      _ROOT / f'results/tracker_training_dagger{it}.png',
                      epochs=100, batch_size=256, lr=1e-4,
                      device=(torch.device(args.device)
                              if args.device else None))
        print(f"[dagger] iter {it}: retrained, val MSE "
              f"{stats['best_val_mse']:.6f}")
        history.append(dict(iter=it,
                            on_policy_samples=int(new_obs.shape[0]),
                            aggregated_samples=int(obs_all.shape[0]),
                            tracker_goal_err_mm_mean=float(np.mean(goal_errs)),
                            val_mse=float(stats['best_val_mse'])))

    json.dump(dict(iters=args.iters, history=history),
              open(_ROOT / 'results/dagger_tracker_log.json', 'w'), indent=2)
    print(f"\n[dagger] done. log -> results/dagger_tracker_log.json")


if __name__ == '__main__':
    main()
