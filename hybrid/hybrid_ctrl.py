"""
hybrid/hybrid_ctrl.py -- orchestrator for the hybrid controller.

One episode:
  1. PlannerRunner plans once (A* + min-snap) -> reference trajectory
     sampled at the control dt.
  2. The MLP tracker closes the control loop: at every control step it
     reads the current MuJoCo state and the reference column for that
     step, predicts a normalised action, and steps the environment.

Per-step logging: reference position/velocity, actual position/velocity,
ESDF, action, tracker inference latency -- so the rollout can be
analysed and rendered afterwards.

The default control rate is 200 Hz (dt_ctrl = 0.005 s). The tracker's
~30 us inference leaves the 5 ms control budget almost entirely idle;
the one-time planning cost is amortised over the whole episode.
"""
from __future__ import annotations

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
from obstacle_course import obstacle_field_value

from hybrid.planner_runner import PlannerRunner

START = np.array([-1.5, -1.5, 1.0])
GOAL = np.array([1.5, 1.5, 1.0])


def run_hybrid(obstacles, tracker, start=START, goal=GOAL,
               safety_margin: float = 0.15,
               dt_ctrl: float = 0.005, dt_sim: float = 0.001,
               t_max: float = 10.0,
               avg_speed: float = 0.8,
               reference_mode: str = 'time',
               pursuit_lead: int = None,
               return_trajectory: bool = True) -> dict:
    """Run one hybrid-controller episode on a single obstacle layout.

    reference_mode:
      'time'    -- index the reference by control step i (time-synced).
                   A BC-cloned tracker lags a moving reference, so the
                   drone falls progressively behind ref[:, i] until it
                   leaves the training distribution and diverges.
      'pursuit' -- pure-pursuit carrot following: at each step find the
                   reference column geometrically closest to the drone
                   (searching forward only), then feed the tracker the
                   column `pursuit_lead` steps ahead of it. The tracker
                   therefore always sees a small, in-distribution
                   position error and the lag-induced divergence is
                   eliminated. This is the standard path-tracking fix
                   and needs no retraining.
    `pursuit_lead` defaults to ~80 ms of reference (the NMPC horizon
    scaled to the control dt) -- enough carrot to keep the drone moving
    forward along the path.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if pursuit_lead is None:
        pursuit_lead = max(1, int(round(0.08 / dt_ctrl)))

    # --- 1. plan once (reference sampled at the control dt) ---
    planner = PlannerRunner(start, goal, obstacles,
                            safety_margin=safety_margin,
                            ref_dt=dt_ctrl, avg_speed=avg_speed)
    ref_xyz = planner.ref[0:3, :].T   # (N_ref, 3) reference positions

    # --- 2. tracker loop ---
    env = CrazyflieEnv(dt_sim=dt_sim, dt_ctrl=dt_ctrl)
    state_mj = env.reset(pos=start)
    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0

    n_steps = int(t_max / dt_ctrl)
    xs = [state_mj.copy()]
    ref_pos_log = []
    ref_vel_log = []
    actual_pos_log = []
    actual_vel_log = []
    esdf_log = []
    action_log = []
    tracking_err_log = []
    inf_times = []
    field_vals = [obstacle_field_value(state_mj[0:3], obstacles)]

    pursuit_ptr = 0          # monotonic closest-reference-index pointer
    search_window = max(20, 2 * pursuit_lead)
    for i in range(n_steps):
        if reference_mode == 'pursuit':
            # closest reference column to the drone, searching forward
            lo = pursuit_ptr
            hi = min(planner.n_ref, pursuit_ptr + search_window)
            d2 = np.sum((ref_xyz[lo:hi] - state_mj[0:3]) ** 2, axis=1)
            pursuit_ptr = lo + int(np.argmin(d2))
            ref_col = planner.query(pursuit_ptr + pursuit_lead)
        else:
            ref_col = planner.query(i)
        a = tracker.predict(state_mj, ref_col)
        inf_times.append(tracker.last_inference_s)
        u = np.clip(u_mid + u_half * a, env.u_min, env.u_max)
        state_mj = env.step(u)

        ref_pos = ref_col[0:3]
        ref_vel = ref_col[3:6]
        actual_pos = state_mj[0:3]
        actual_vel = state_mj[3:6]
        # Tracking error = geometric distance to the CLOSEST point on the
        # reference path (not the look-ahead carrot) -- the honest
        # path-deviation metric.
        if reference_mode == 'pursuit':
            track_err = float(np.linalg.norm(
                ref_xyz[pursuit_ptr] - actual_pos))
        else:
            track_err = float(np.linalg.norm(ref_pos - actual_pos))

        ref_pos_log.append(ref_pos.copy())
        ref_vel_log.append(ref_vel.copy())
        actual_pos_log.append(actual_pos.copy())
        actual_vel_log.append(actual_vel.copy())
        esdf_log.append(float(env_safe_esdf(planner, actual_pos)))
        action_log.append(a.copy())
        tracking_err_log.append(track_err)
        xs.append(state_mj.copy())
        field_vals.append(obstacle_field_value(actual_pos, obstacles))

    xs = np.asarray(xs)
    field_vals = np.asarray(field_vals)
    ref_pos_arr = np.asarray(ref_pos_log)
    actual_pos_arr = np.asarray(actual_pos_log)
    track_err_arr = np.asarray(tracking_err_log)

    final_pos = xs[-1, 0:3]
    goal_err_mm = float(np.linalg.norm(final_pos - goal) * 1000.0)
    path_len_m = float(np.sum(np.linalg.norm(
        np.diff(xs[:, 0:3], axis=0), axis=1)))
    straight = float(np.linalg.norm(goal - start))
    efficiency = float(straight / max(path_len_m, 1e-9))
    duration_s = n_steps * dt_ctrl

    # Latency accounting:
    #   per-step tracker inference (measured), plus the one-time planning
    #   cost amortised over all control steps.
    median_tracker_us = float(np.median(inf_times) * 1e6)
    amortised_plan_us = float(planner.planning_time_ms * 1000.0
                              / max(n_steps, 1))
    effective_latency_us = median_tracker_us + amortised_plan_us

    out = dict(
        goal_err_mm=goal_err_mm,
        max_field=float(np.max(field_vals)),
        mean_field=float(np.mean(field_vals)),
        path_len_m=path_len_m,
        efficiency=efficiency,
        mean_speed_mps=path_len_m / max(duration_s, 1e-6),
        tracking_error_mean_mm=float(track_err_arr.mean() * 1000.0),
        tracking_error_max_mm=float(track_err_arr.max() * 1000.0),
        tracking_error_p95_mm=float(np.percentile(track_err_arr, 95)
                                    * 1000.0),
        median_tracker_inference_us=median_tracker_us,
        amortised_planning_us=amortised_plan_us,
        effective_latency_us=effective_latency_us,
        planning_time_ms=float(planner.planning_time_ms),
        astar_ms=float(planner.meta['astar_ms']),
        smooth_ms=float(planner.meta['smooth_ms']),
        n_steps=int(n_steps),
        dt_ctrl=float(dt_ctrl),
        ref_total_s=float(planner.total_time),
    )
    if return_trajectory:
        out.update(dict(
            xs=xs[:, 0:3].astype(np.float32),
            ref_pos=ref_pos_arr.astype(np.float32),
            actual_pos=actual_pos_arr.astype(np.float32),
            ref_vel=np.asarray(ref_vel_log, dtype=np.float32),
            actual_vel=np.asarray(actual_vel_log, dtype=np.float32),
            actions=np.asarray(action_log, dtype=np.float32),
            tracking_err=track_err_arr.astype(np.float32),
            esdf=np.asarray(esdf_log, dtype=np.float32),
            field_vals=field_vals.astype(np.float32),
            obstacles=obstacles,
            ref_full=planner.ref.astype(np.float32),
        ))
    return out


def env_safe_esdf(planner: PlannerRunner, pos: np.ndarray) -> float:
    """ESDF at `pos` using the voxel map the planner already built."""
    vm = planner.meta.get('voxel_map', None)
    if vm is None:
        return 0.0
    try:
        return float(vm.query_esdf(pos))
    except Exception:
        return 0.0
