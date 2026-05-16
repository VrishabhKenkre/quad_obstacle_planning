"""
hierarchical_ctrl.py -- glue the planner stack onto the existing SE(3) NMPC.

Pipeline:
    obstacles -> VoxelMap + ESDF
    A* (with safety_margin inflation)
    line-of-sight prune + endpoint snap
    minimum-snap smoothing -> (12, N_ref) reference
    NMPC tracking loop driving the existing CrazyflieEnv

This module does NOT modify src/. It imports SE3_NMPC, CrazyflieEnv, and the
obstacle helpers as-is, and exposes a single run_hierarchical(...) entry
point used by eval_planner.py and the demo recorder.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Local planning imports.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / 'src'))

from voxelize import VoxelMap
from astar import astar_3d, voxel_path_to_world, prune_path
from min_snap import smooth_waypoints

# NMPC + env (do NOT modify these).
from nonlinear_mpc import SE3_NMPC, rotors_to_mujoco
from quad_env import CrazyflieEnv
from obstacle_course import obstacle_field_value


def _mujoco_state_to_nmpc_state(x_mj: np.ndarray) -> np.ndarray:
    """12D MuJoCo state (pos, vel, rpy, omega_body) -> 13D NMPC state
    (pos, vel, quat [w,x,y,z], omega_body)."""
    p, v, rpy, w = x_mj[0:3], x_mj[3:6], x_mj[6:9], x_mj[9:12]
    phi, theta, psi = rpy
    cy, sy = np.cos(psi/2), np.sin(psi/2)
    cp, sp = np.cos(theta/2), np.sin(theta/2)
    cr, sr = np.cos(phi/2), np.sin(phi/2)
    q = np.array([cr*cp*cy + sr*sp*sy,
                  sr*cp*cy - cr*sp*sy,
                  cr*sp*cy + sr*cp*sy,
                  cr*cp*sy - sr*sp*cy])
    return np.concatenate([p, v, q, w])


def _take_ref_window(ref: np.ndarray, i: int, N: int) -> tuple:
    """Slice a (12, N_total) reference into an (N+1)-wide window starting
    at index i, padding with the final column once we run off the end."""
    N_total = ref.shape[1]
    rp = np.zeros((3, N + 1))
    rv = np.zeros((3, N + 1))
    for k in range(N + 1):
        j = min(i + k, N_total - 1)
        rp[:, k] = ref[0:3, j]
        rv[:, k] = ref[3:6, j]
    return rp, rv


def plan_once(start: np.ndarray, goal: np.ndarray, obstacles: list,
              safety_margin: float = 0.15,
              voxel_resolution: float = 0.05,
              ref_dt: float = 0.01,
              avg_speed: float = 0.8):
    """Build voxel map + ESDF, run A*, smooth. Returns (ref, meta)."""
    t0 = time.perf_counter()
    vm = VoxelMap(resolution=voxel_resolution)
    vm.from_obstacle_field(obstacles, threshold=0.3)
    vm.compute_esdf()
    t_vox = time.perf_counter() - t0

    t0 = time.perf_counter()
    path = astar_3d(start, goal, vm, safety_margin=safety_margin)
    t_astar = time.perf_counter() - t0
    if path is None:
        raise RuntimeError("A* failed to find a path")

    pw = voxel_path_to_world(path, vm)
    pw = prune_path(pw, vm, safety_margin=safety_margin, max_step_m=0.5)
    # Snap exact endpoints (replace the voxel-center quantisation).
    pw[0] = start
    pw[-1] = goal

    t0 = time.perf_counter()
    ref, smooth_meta = smooth_waypoints(pw, target_dt=ref_dt,
                                        target_avg_speed=avg_speed,
                                        return_meta=True)
    t_smooth = time.perf_counter() - t0

    meta = dict(
        n_voxel_path=len(path),
        n_waypoints=len(pw),
        voxel_build_ms=t_vox * 1000,
        astar_ms=t_astar * 1000,
        smooth_ms=t_smooth * 1000,
        planning_time_ms=(t_vox + t_astar + t_smooth) * 1000,
        ref_total_s=smooth_meta['total_time'],
        smoother=smooth_meta['solver'],
        voxel_map=vm,
        pruned_waypoints=pw,
    )
    return ref, meta


def run_hierarchical(start: np.ndarray, goal: np.ndarray, obstacles: list,
                     dt: float = 0.01, T_max: float = 10.0,
                     safety_margin: float = 0.15,
                     avg_speed: float = 0.8,
                     voxel_resolution: float = 0.05,
                     nmpc_N: int = 15,
                     nmpc_dt: float = 0.02,
                     return_trajectory: bool = True,
                     verbose: bool = False):
    """End-to-end hierarchical run on a single obstacle course.

    Returns a dict matching the keys expected by eval_planner.py.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    # ---- Plan once ----
    ref, plan_meta = plan_once(start, goal, obstacles,
                                safety_margin=safety_margin,
                                voxel_resolution=voxel_resolution,
                                ref_dt=nmpc_dt,
                                avg_speed=avg_speed)
    if verbose:
        print(f"  [plan] {plan_meta['n_voxel_path']} A* voxels -> "
              f"{plan_meta['n_waypoints']} waypoints -> "
              f"{ref.shape[1]} ref samples ({plan_meta['ref_total_s']:.2f} s)")
        print(f"  [plan] voxel {plan_meta['voxel_build_ms']:.0f} ms | "
              f"A* {plan_meta['astar_ms']:.0f} ms | "
              f"smooth {plan_meta['smooth_ms']:.0f} ms | "
              f"total {plan_meta['planning_time_ms']:.0f} ms")

    # NMPC built with obstacles passed in (the existing implementation
    # supports an empty list -- we feed a *short* list of nearby obstacles
    # so the optimiser doesn't blow up assembly time, but a global tracker
    # works fine with the empty list too).
    nmpc = SE3_NMPC(N=nmpc_N, dt=nmpc_dt, obstacles=obstacles,
                    q_pos=300, q_vel=10, q_quat=20, q_omega=0.1,
                    r_thrust=1e3, w_obs=800.0)
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=nmpc_dt)
    x_mj = env.reset(pos=start)

    # ---- Track ----
    n_steps = min(int(T_max / nmpc_dt), ref.shape[1])
    xs = [x_mj.copy()]
    us = []
    solve_times = []
    field_vals = [obstacle_field_value(x_mj[0:3], obstacles)]

    for i in range(n_steps):
        rp_win, rv_win = _take_ref_window(ref, i, nmpc_N)
        x13 = _mujoco_state_to_nmpc_state(x_mj)
        t0 = time.perf_counter()
        u_rotors, info = nmpc.solve(x13, rp_win, rv_win)
        solve_times.append(time.perf_counter() - t0)
        u_mj = rotors_to_mujoco(u_rotors)
        x_mj = env.step(u_mj)
        xs.append(x_mj.copy())
        us.append(u_mj.copy())
        field_vals.append(obstacle_field_value(x_mj[0:3], obstacles))

    xs = np.array(xs)
    us = np.array(us)
    solve_times = np.array(solve_times)
    field_vals = np.array(field_vals)

    # ---- Metrics ----
    final_pos = xs[-1, 0:3]
    goal_err_mm = float(np.linalg.norm(final_pos - goal) * 1000)
    path_len_m = float(np.sum(np.linalg.norm(np.diff(xs[:, 0:3], axis=0), axis=1)))
    max_field = float(np.max(field_vals))
    mean_field = float(np.mean(field_vals))
    # Speed = path / time, using the executed segment up to closest-approach to goal.
    duration_s = n_steps * nmpc_dt
    mean_speed = path_len_m / max(duration_s, 1e-6)
    median_solve_ms = float(np.median(solve_times) * 1000)
    mean_solve_ms = float(np.mean(solve_times) * 1000)

    out = dict(
        goal_err_mm=goal_err_mm,
        max_field=max_field,
        mean_field=mean_field,
        path_len_m=path_len_m,
        mean_speed_mps=float(mean_speed),
        median_solve_ms=median_solve_ms,
        mean_solve_ms=mean_solve_ms,
        planning_time_ms=plan_meta['planning_time_ms'],
        astar_ms=plan_meta['astar_ms'],
        smooth_ms=plan_meta['smooth_ms'],
        voxel_build_ms=plan_meta['voxel_build_ms'],
        smoother=plan_meta['smoother'],
        n_steps=int(n_steps),
        ref_total_s=plan_meta['ref_total_s'],
    )
    if return_trajectory:
        out.update(dict(xs=xs, us=us, solve_times=solve_times,
                        field_vals=field_vals, ref=ref,
                        pruned_waypoints=plan_meta['pruned_waypoints'],
                        start=start, goal=goal, obstacles=obstacles,
                        dt=nmpc_dt))
    if verbose:
        print(f"  [track] goal_err={goal_err_mm:.1f} mm | "
              f"max_field={max_field:.3f} | path_len={path_len_m:.2f} m | "
              f"speed={mean_speed:.2f} m/s | NMPC median {median_solve_ms:.0f} ms")
    return out


if __name__ == '__main__':
    from obstacle_course import make_obstacles
    obstacles = make_obstacles(seed=42)
    start = np.array([-1.5, -1.5, 1.0])
    goal = np.array([1.5, 1.5, 1.0])
    result = run_hierarchical(start, goal, obstacles, verbose=True)
    print(f"\nResult keys: {[k for k in result.keys() if not isinstance(result[k], np.ndarray) and not isinstance(result[k], list)]}")
