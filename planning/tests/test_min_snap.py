import numpy as np
import pytest

from min_snap import smooth_waypoints


def test_endpoints_match():
    wp = np.array([[-1.0, -1.0, 1.0],
                   [ 0.0,  0.5, 1.2],
                   [ 1.0,  1.0, 1.0]])
    ref, meta = smooth_waypoints(wp, target_dt=0.01,
                                  target_avg_speed=0.8, return_meta=True)
    # Position endpoints match supplied start/goal to better than 1 mm.
    assert np.linalg.norm(ref[0:3, 0] - wp[0]) < 1e-3
    assert np.linalg.norm(ref[0:3, -1] - wp[-1]) < 1e-3


def test_zero_endpoint_velocity():
    wp = np.array([[0.0, 0.0, 1.0],
                   [1.0, 0.0, 1.0],
                   [2.0, 0.0, 1.0]])
    ref = smooth_waypoints(wp, target_dt=0.01, target_avg_speed=0.5)
    assert np.linalg.norm(ref[3:6, 0]) < 1e-6
    assert np.linalg.norm(ref[3:6, -1]) < 1e-6


def test_continuity_at_interior_waypoints():
    """Numerical velocity should be consistent across the entire trajectory
    (no jumps), confirming derivative continuity at segment boundaries."""
    wp = np.array([[0.0, 0.0, 1.0],
                   [1.0, 0.5, 1.0],
                   [2.0, 0.0, 1.0],
                   [3.0, 0.5, 1.0]])
    dt = 0.005
    ref = smooth_waypoints(wp, target_dt=dt, target_avg_speed=0.6)
    # Numerical velocity from finite diff of position.
    pos = ref[0:3]
    vel_num = np.diff(pos, axis=1) / dt
    # Analytic velocity rows.
    vel_an = ref[3:6, :-1]
    # Difference should be small (smooth poly).
    err = np.max(np.linalg.norm(vel_num - vel_an, axis=0))
    assert err < 0.2  # m/s, allowing for finite-difference truncation


def test_returned_shape_12xT():
    wp = np.array([[0.0, 0.0, 1.0],
                   [1.0, 1.0, 1.0]])
    ref = smooth_waypoints(wp, target_dt=0.01, target_avg_speed=0.8)
    assert ref.shape[0] == 12
    assert ref.shape[1] > 10


def test_min_snap_solver_used():
    """For a well-conditioned 3-waypoint path the min-snap KKT system
    should be non-singular and selected over the cubic fallback."""
    wp = np.array([[-1.0, -1.0, 1.0],
                   [ 0.0,  0.0, 1.2],
                   [ 1.0,  1.0, 1.0]])
    ref, meta = smooth_waypoints(wp, target_dt=0.01,
                                  target_avg_speed=0.8, return_meta=True)
    assert meta['solver'] == 'min_snap'
