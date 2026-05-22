"""
hybrid/planner_runner.py -- the planning front-end of the hybrid
controller.

The hierarchical planner's A*+min-snap stage is run ONCE per episode
(the user-chosen "plan once, retain reference" mode -- a true 5 Hz
receding-horizon loop is infeasible because one A*+min-snap replan
costs ~370 ms, see the sprint report). The resulting min-snap
reference is a (12, N) array sampled at `ref_dt`; rows 0-2 are
position, 3-5 velocity, 6-8 acceleration, 9-11 jerk.

`PlannerRunner` wraps `planning.hierarchical_ctrl.plan_once` and exposes
the reference plus a `query(i)` accessor that clamps past the end (so
the tracker holds the final reference pose once the trajectory is
exhausted, i.e. hover at goal).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))

from hierarchical_ctrl import plan_once


class PlannerRunner:
    """Plan-once front-end: builds the min-snap reference for one episode."""

    def __init__(self, start, goal, obstacles,
                 safety_margin: float = 0.15,
                 ref_dt: float = 0.005,
                 avg_speed: float = 0.8,
                 voxel_resolution: float = 0.05):
        self.start = np.asarray(start, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        self.obstacles = obstacles
        self.ref_dt = float(ref_dt)

        # The single planning call. plan_once builds VoxelMap+ESDF, runs
        # A*, prunes, and min-snap-smooths to a (12, N) reference.
        self.ref, self.meta = plan_once(
            self.start, self.goal, obstacles,
            safety_margin=safety_margin,
            voxel_resolution=voxel_resolution,
            ref_dt=ref_dt, avg_speed=avg_speed)
        self.n_ref = self.ref.shape[1]
        # The planning cost (A* + min-snap + voxel build), in ms -- this
        # is the one-time cost amortised over the whole episode.
        self.planning_time_ms = float(self.meta['planning_time_ms'])

    def query(self, i: int) -> np.ndarray:
        """Reference column at index i, clamped to [0, n_ref-1]."""
        j = int(min(max(i, 0), self.n_ref - 1))
        return self.ref[:, j]

    def query_time(self, t: float) -> np.ndarray:
        """Reference at wall time t (s), nearest-column lookup."""
        return self.query(int(round(t / self.ref_dt)))

    @property
    def total_time(self) -> float:
        return self.n_ref * self.ref_dt
