"""
collect_tracking_data.py -- build a (state, reference) -> action dataset
for the hybrid controller's learned tracker.

The hybrid system runs the hierarchical planner's A*+min-snap front-end
ONCE per episode to produce a reference trajectory, then a fast learned
tracker follows that reference. This script generates the tracker's
training data.

CRITICAL: the tracking loop injects DART-style noise (Laskey et al.
2017) into the *applied* action, exactly as collect_planner_data.py
does for the navigation dataset. A clean NMPC rollout keeps the drone
glued to the reference (~15 mm error), so a behavioural clone trained
on it never sees off-reference states and compounding error collapses
it in closed loop. With DART noise the drone drifts off-reference and
each recorded label is the NMPC's *clean* recovery action from the
drifted state -- the tracker learns to recover.

At every NMPC control step we record:

  observation (27-D):
    obs[0:12]  = current 12-D MuJoCo state (pos, vel, rpy, omega)
    obs[12:24] = reference column ref[:, i]  (min-snap pos/vel/acc/jerk)
    obs[24:27] = reference velocity ref[3:6, i]  (explicit feedforward)
  action (4-D):
    the NMPC's clean actuator command, normalised to [-1, 1] via
    CrazyflieEnv.{u_min,u_max}.

Output: data/tracking_dataset_v1.npz with arrays
  obs        (N, 27) float32
  actions    (N, 4)  float32  in [-1, 1]   (clean NMPC labels)
  seed       (N,)    int32
  step_idx   (N,)    int32

Usage:
  python distillation/collect_tracking_data.py
  python distillation/collect_tracking_data.py --n-random 200 --n-dp 30
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
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))

from quad_env import CrazyflieEnv
from obstacle_course import make_obstacles
from nonlinear_mpc import SE3_NMPC, rotors_to_mujoco
from hierarchical_ctrl import (plan_once, _take_ref_window,
                                _mujoco_state_to_nmpc_state)
from randomize_astar import decision_point_layout

START = np.array([-1.5, -1.5, 1.0])
GOAL = np.array([1.5, 1.5, 1.0])
TRACK_OBS_DIM = 27
ACT_DIM = 4
NMPC_N = 15

# DART noise std in normalised [-1,1] action space -- matches
# collect_planner_data.py so the tracker sees the same off-reference
# excursion magnitude the navigation student was trained against.
DART_SIGMA = np.array([0.030, 0.040, 0.040, 0.030], dtype=np.float32)

RANDOM_COLLECT_SEEDS = list(range(1000, 1000 + 200))
DP_COLLECT_SEEDS = list(range(2000, 2000 + 30))


def _u_limits():
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=0.02)
    return ((env.u_max + env.u_min) / 2.0,
            (env.u_max - env.u_min) / 2.0,
            env.u_min, env.u_max)


def collect_one(seed: int, seed_type: str, u_mid, u_half, u_min, u_max,
                rng: np.random.Generator,
                nmpc_dt: float = 0.02, t_max: float = 10.0) -> dict | None:
    """Run a DART-noisy NMPC tracking rollout; record (obs, clean action)."""
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

    # Run the FULL episode, not just ref.shape[1] steps: once the
    # min-snap reference is exhausted, _take_ref_window clamps to the
    # final column (goal, zero velocity), so the NMPC decelerates and
    # holds at the goal. Collecting this tail is essential -- otherwise
    # the cloned tracker never learns to stop, overshoots the goal at
    # deploy time, and flies away into states it never trained on.
    n_steps = int(t_max / nmpc_dt)
    n_ref = ref.shape[1]
    obs_list = np.empty((n_steps, TRACK_OBS_DIM), dtype=np.float32)
    act_list = np.empty((n_steps, ACT_DIM), dtype=np.float32)
    final_goal_err_mm = None
    for i in range(n_steps):
        rp_win, rv_win = _take_ref_window(ref, i, NMPC_N)
        x13 = _mujoco_state_to_nmpc_state(x_mj)
        u_rotors, _info = nmpc.solve(x13, rp_win, rv_win)
        u_mj = rotors_to_mujoco(u_rotors)

        # --- record (obs, CLEAN action) ---
        ref_col = ref[:, min(i, n_ref - 1)]
        obs_list[i, 0:12] = x_mj[0:12].astype(np.float32)
        obs_list[i, 12:24] = ref_col[0:12].astype(np.float32)
        obs_list[i, 24:27] = ref_col[3:6].astype(np.float32)
        action_clean = np.clip((u_mj - u_mid) / u_half,
                               -1.0, 1.0).astype(np.float32)
        act_list[i] = action_clean

        # --- DART noise: step env with a noisy action ---
        noise = (rng.standard_normal(ACT_DIM).astype(np.float32)
                 * DART_SIGMA)
        action_noisy = np.clip(action_clean + noise, -1.0, 1.0)
        u_apply = np.clip(u_mid + u_half * action_noisy, u_min, u_max)
        x_mj = env.step(u_apply)

    final_goal_err_mm = float(np.linalg.norm(x_mj[0:3] - GOAL) * 1000)
    return dict(obs=obs_list, actions=act_list, n_steps=int(n_steps),
                goal_err_mm=final_goal_err_mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-random', type=int, default=200)
    ap.add_argument('--n-dp', type=int, default=30)
    ap.add_argument('--nmpc-dt', type=float, default=0.02)
    ap.add_argument('--t-max', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=0, help='RNG seed for DART noise')
    ap.add_argument('--out', type=str, default='data/tracking_dataset_v1.npz')
    ap.add_argument('--stats-out', type=str,
                    default='data/tracking_dataset_v1_stats.json')
    args = ap.parse_args()

    u_mid, u_half, u_min, u_max = _u_limits()
    rng = np.random.default_rng(args.seed)
    random_seeds = RANDOM_COLLECT_SEEDS[:args.n_random]
    dp_seeds = DP_COLLECT_SEEDS[:args.n_dp]
    print(f"[track-collect] {len(random_seeds)} random + {len(dp_seeds)} dp "
          f"rollouts, nmpc_dt={args.nmpc_dt}, DART noise on")

    all_obs, all_act, all_seed, all_step = [], [], [], []
    n_ok = n_skip = 0
    teacher_goal_errs = []
    t0 = time.time()
    plan = ([(s, 'random') for s in random_seeds]
            + [(s, 'dp') for s in dp_seeds])
    for idx, (seed, stype) in enumerate(plan):
        r = collect_one(seed, stype, u_mid, u_half, u_min, u_max, rng,
                        nmpc_dt=args.nmpc_dt, t_max=args.t_max)
        if r is None:
            n_skip += 1
            continue
        n_ok += 1
        all_obs.append(r['obs'])
        all_act.append(r['actions'])
        all_seed.append(np.full(r['n_steps'], seed, dtype=np.int32))
        all_step.append(np.arange(r['n_steps'], dtype=np.int32))
        teacher_goal_errs.append(r['goal_err_mm'])
        if (idx + 1) % 25 == 0 or idx == len(plan) - 1:
            n_so_far = sum(o.shape[0] for o in all_obs)
            print(f"    {idx+1}/{len(plan)} rollouts  "
                  f"({n_so_far:,} samples, {time.time()-t0:.0f}s, "
                  f"last: {stype} seed {seed} "
                  f"goal={r['goal_err_mm']:.0f}mm)")

    obs = np.concatenate(all_obs, axis=0)
    act = np.concatenate(all_act, axis=0)
    seed_arr = np.concatenate(all_seed, axis=0)
    step_arr = np.concatenate(all_step, axis=0)

    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, obs=obs, actions=act,
                        seed=seed_arr, step_idx=step_arr)
    print(f"\n[track-collect] saved {obs.shape[0]:,} samples "
          f"({n_ok} rollouts, {n_skip} skipped) -> {out_path}")

    stats = dict(
        n_samples=int(obs.shape[0]), n_rollouts=int(n_ok),
        n_skipped=int(n_skip), obs_dim=TRACK_OBS_DIM, act_dim=ACT_DIM,
        nmpc_dt=float(args.nmpc_dt), dart_noise=True,
        dart_sigma=[float(x) for x in DART_SIGMA],
        n_random=len(random_seeds), n_dp=len(dp_seeds),
        teacher_goal_err_mm_mean=float(np.mean(teacher_goal_errs)),
        teacher_goal_err_mm_median=float(np.median(teacher_goal_errs)),
        collection_seconds=float(time.time() - t0),
    )
    json.dump(stats, open(_ROOT / args.stats_out, 'w'), indent=2)
    print(f"[track-collect] stats -> {args.stats_out}")
    print(f"  teacher (NMPC, DART-noisy rollout) final goal err: mean "
          f"{stats['teacher_goal_err_mm_mean']:.1f} mm, median "
          f"{stats['teacher_goal_err_mm_median']:.1f} mm")


if __name__ == '__main__':
    main()
