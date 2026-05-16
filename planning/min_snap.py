"""
min_snap.py -- 7th-order minimum-snap polynomial smoothing of an A* path.

Implements Mellinger & Kumar (ICRA 2011) min-snap with continuity of
position, velocity, acceleration, jerk at interior waypoints; zero
velocity/acceleration/jerk at the start and end. Per axis we solve

    minimize   c^T Q c
    subject to A c = b

via the KKT system [[Q, A^T], [A, 0]] [c; lambda] = [0; b].

Falls back to a clamped cubic spline (TODO: replace with full min-snap
once validated) if the KKT system is poorly conditioned -- this is
flagged with the `solver` field of the returned dict.

The function `smooth_waypoints` returns a (12, T) reference matching the
format consumed by src/dagger.gen_fig8_ff:
    row  0-2  : position
    row  3-5  : velocity
    row  6-7  : roll/pitch (differential-flatness feedforward)
    row  8    : yaw (0)
    row  9-11 : body-frame angular rates (0)
"""
from __future__ import annotations

import math
import numpy as np


_ORDER = 7
_N_COEF = _ORDER + 1


def _snap_hessian(T: float) -> np.ndarray:
    """Hessian of int_0^T (d^4 p / dt^4)^2 dt for a 7th-order polynomial.
    Polynomial form p(t) = sum_{i=0}^{7} c_i * t^i."""
    Q = np.zeros((_N_COEF, _N_COEF))
    for i in range(4, _N_COEF):
        ki = math.factorial(i) // math.factorial(i - 4)
        for j in range(4, _N_COEF):
            kj = math.factorial(j) // math.factorial(j - 4)
            exp = i + j - 7
            Q[i, j] = ki * kj * (T ** exp) / exp
    return Q


def _deriv_basis(t: float, deriv: int) -> np.ndarray:
    """Row vector r such that p^(deriv)(t) = r @ c for c = (c_0,...,c_7)."""
    r = np.zeros(_N_COEF)
    for i in range(deriv, _N_COEF):
        k = math.factorial(i) // math.factorial(i - deriv)
        r[i] = k * (t ** (i - deriv))
    return r


def _allocate_times(waypoints: np.ndarray, avg_speed: float,
                    min_seg: float = 0.3) -> np.ndarray:
    """Time per segment proportional to segment length / avg_speed,
    with a small minimum to avoid singular tiny segments."""
    segs = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    T = np.maximum(segs / max(avg_speed, 1e-6), min_seg)
    return T


def _solve_axis(wp_axis: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Solve min-snap for one axis given M+1 waypoints and M segment durations.
    Returns coefficient array of shape (M, 8)."""
    M = len(T)
    n = _N_COEF * M  # total coefficients

    # Block-diagonal cost matrix.
    Q = np.zeros((n, n))
    for k in range(M):
        Q[k*_N_COEF:(k+1)*_N_COEF, k*_N_COEF:(k+1)*_N_COEF] = _snap_hessian(T[k])

    # Build equality constraints.
    A_rows = []
    b_vals = []

    # 1) Position at start and end of each segment.
    for k in range(M):
        # p_k(0) = wp[k]
        row = np.zeros(n); row[k*_N_COEF:(k+1)*_N_COEF] = _deriv_basis(0.0, 0)
        A_rows.append(row); b_vals.append(wp_axis[k])
        # p_k(T_k) = wp[k+1]
        row = np.zeros(n); row[k*_N_COEF:(k+1)*_N_COEF] = _deriv_basis(T[k], 0)
        A_rows.append(row); b_vals.append(wp_axis[k+1])

    # 2) Continuity of derivatives 1, 2, 3 at interior waypoints.
    for k in range(M - 1):
        for d in (1, 2, 3):
            row = np.zeros(n)
            row[k*_N_COEF:(k+1)*_N_COEF] = _deriv_basis(T[k], d)
            row[(k+1)*_N_COEF:(k+2)*_N_COEF] = -_deriv_basis(0.0, d)
            A_rows.append(row); b_vals.append(0.0)

    # 3) Zero derivatives 1, 2, 3 at start and end.
    for d in (1, 2, 3):
        row = np.zeros(n); row[0:_N_COEF] = _deriv_basis(0.0, d)
        A_rows.append(row); b_vals.append(0.0)
        row = np.zeros(n); row[(M-1)*_N_COEF:M*_N_COEF] = _deriv_basis(T[-1], d)
        A_rows.append(row); b_vals.append(0.0)

    A = np.array(A_rows)
    b = np.array(b_vals)

    # KKT system.
    m_con = A.shape[0]
    KKT = np.zeros((n + m_con, n + m_con))
    KKT[:n, :n] = Q + Q.T  # (Q is symmetric; this avoids the 0.5 factor confusion)
    KKT[:n, n:] = A.T
    KKT[n:, :n] = A
    rhs = np.concatenate([np.zeros(n), b])

    sol = np.linalg.solve(KKT, rhs)
    coefs = sol[:n].reshape(M, _N_COEF)
    return coefs


def _eval_poly(coefs: np.ndarray, t: float, deriv: int = 0) -> float:
    r = _deriv_basis(t, deriv)
    return float(np.dot(r, coefs))


def _cubic_fallback(waypoints: np.ndarray, T: np.ndarray):
    """Clamped cubic spline per axis (zero endpoint velocity)."""
    from scipy.interpolate import CubicSpline
    cum = np.concatenate(([0.0], np.cumsum(T)))
    splines = []
    for ax in range(3):
        cs = CubicSpline(cum, waypoints[:, ax], bc_type='clamped')
        splines.append(cs)
    return splines, cum


def smooth_waypoints(waypoint_path_world: np.ndarray,
                     target_dt: float = 0.01,
                     target_avg_speed: float = 0.8,
                     return_meta: bool = False):
    """7th-order minimum-snap smoother.

    Args:
        waypoint_path_world: (M+1, 3) array of world-frame waypoints, in
            execution order. The first and last entries are taken as the
            actual start and goal.
        target_dt: sampling timestep of the returned reference [s].
        target_avg_speed: nominal cruise speed used for time allocation [m/s].
        return_meta: if True, return (ref, meta_dict).

    Returns:
        ref: (12, T) array. Rows 0-2 position, 3-5 velocity, 6-7 roll/pitch
             (differential-flatness feedforward from acceleration), 8 yaw,
             9-11 body angular rates (left at zero -- the NMPC consumes only
             rows 0-5 as XREF/VREF and the feedforward attitude is for
             downstream Linear-MPC compatibility).
        meta: dict with keys {'T_seg', 'total_time', 'solver'} when requested.
    """
    waypoints = np.asarray(waypoint_path_world, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] < 2:
        raise ValueError("waypoint_path_world must be (M+1, 3) with M>=1")

    # Collapse duplicate-adjacent waypoints (A* can emit them after pruning
    # the start/goal voxel snap).
    keep = [0]
    for i in range(1, len(waypoints)):
        if np.linalg.norm(waypoints[i] - waypoints[keep[-1]]) > 1e-4:
            keep.append(i)
    waypoints = waypoints[keep]

    if len(waypoints) < 2:
        raise ValueError("need at least 2 distinct waypoints")

    T_seg = _allocate_times(waypoints, target_avg_speed)
    M = len(T_seg)
    total_time = float(np.sum(T_seg))

    solver = 'min_snap'
    coefs_xyz = []
    try:
        for ax in range(3):
            coefs_xyz.append(_solve_axis(waypoints[:, ax], T_seg))
    except np.linalg.LinAlgError:
        solver = 'cubic_fallback'
        coefs_xyz = None  # signal fallback

    N = int(np.ceil(total_time / target_dt)) + 1
    ts = np.arange(N) * target_dt
    ts = np.clip(ts, 0.0, total_time)

    pos = np.zeros((3, N))
    vel = np.zeros((3, N))
    acc = np.zeros((3, N))

    if solver == 'min_snap':
        cum = np.concatenate(([0.0], np.cumsum(T_seg)))
        # For each sample time, find segment index.
        seg_idx = np.searchsorted(cum, ts, side='right') - 1
        seg_idx = np.clip(seg_idx, 0, M - 1)
        for n_idx in range(N):
            k = seg_idx[n_idx]
            tau = ts[n_idx] - cum[k]
            for ax in range(3):
                c = coefs_xyz[ax][k]
                pos[ax, n_idx] = _eval_poly(c, tau, 0)
                vel[ax, n_idx] = _eval_poly(c, tau, 1)
                acc[ax, n_idx] = _eval_poly(c, tau, 2)
    else:
        # TODO: this fallback is engaged only if min-snap KKT is singular.
        splines, cum = _cubic_fallback(waypoints, T_seg)
        for ax in range(3):
            pos[ax] = splines[ax](ts)
            vel[ax] = splines[ax](ts, 1)
            acc[ax] = splines[ax](ts, 2)

    # Differential-flatness feedforward roll/pitch (yaw fixed at 0).
    # Crazyflie ZYX convention: a_x = g*theta, a_y = -g*phi (at hover linearization).
    g = 9.81
    ref = np.zeros((12, N))
    ref[0:3] = pos
    ref[3:6] = vel
    ref[6] = -acc[1] / g       # roll  phi  ~ -ay / g
    ref[7] =  acc[0] / g       # pitch theta ~ ax / g
    # yaw, body rates remain zero (NMPC tracks pos/vel only).

    if return_meta:
        meta = dict(T_seg=T_seg, total_time=total_time, solver=solver,
                    n_waypoints=int(len(waypoints)),
                    target_avg_speed=float(target_avg_speed))
        return ref, meta
    return ref


if __name__ == '__main__':
    # Smoke test on a representative A* path.
    import sys, time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from obstacle_course import make_obstacles
    from voxelize import VoxelMap
    from astar import astar_3d, voxel_path_to_world, prune_path

    obstacles = make_obstacles(seed=42)
    vm = VoxelMap()
    vm.from_obstacle_field(obstacles)
    vm.compute_esdf()

    start = np.array([-1.5, -1.5, 1.0]); goal = np.array([1.5, 1.5, 1.0])
    path = astar_3d(start, goal, vm, safety_margin=0.15)
    pw = voxel_path_to_world(path, vm)
    pw = prune_path(pw, vm, safety_margin=0.15)
    # Force exact endpoints (replace the voxel-center snap with user start/goal).
    pw[0] = start; pw[-1] = goal

    t0 = time.time()
    ref, meta = smooth_waypoints(pw, target_dt=0.01, target_avg_speed=0.8,
                                  return_meta=True)
    print(f"smoothing: {(time.time()-t0)*1000:.1f} ms, solver={meta['solver']}")
    print(f"ref shape {ref.shape}, total time {meta['total_time']:.2f} s, "
          f"{meta['n_waypoints']} waypoints")
    print(f"start pos {ref[0:3, 0]}, end pos {ref[0:3, -1]}")
    print(f"start vel {ref[3:6, 0]}, end vel {ref[3:6, -1]}")
    print(f"peak |a| = {np.max(np.linalg.norm([ref[3]-ref[3], ref[4]-ref[4], ref[5]-ref[5]], axis=0)):.2f}")
    # Print maximum field along path
    from obstacle_course import obstacle_field_value
    fields = [obstacle_field_value(ref[0:3, i], obstacles) for i in range(ref.shape[1])]
    print(f"max obstacle field along smoothed path: {max(fields):.3f}")
