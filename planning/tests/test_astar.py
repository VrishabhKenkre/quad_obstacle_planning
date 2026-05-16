import numpy as np
import pytest

from voxelize import VoxelMap
from astar import astar_3d, voxel_path_to_world, prune_path


def _empty_map():
    vm = VoxelMap(bounds=((-1, 1), (-1, 1), (0, 2)), resolution=0.1)
    return vm


def test_trivial_no_obstacles():
    vm = _empty_map()
    start = np.array([-0.5, -0.5, 1.0])
    goal = np.array([0.5, 0.5, 1.0])
    path = astar_3d(start, goal, vm, safety_margin=0.0)
    assert path is not None
    # Convert to world and check straight-line distance is close.
    pw = voxel_path_to_world(path, vm)
    plen = float(np.sum(np.linalg.norm(np.diff(pw, axis=0), axis=1)))
    straight = float(np.linalg.norm(goal - start))
    assert plen < straight + 0.2  # within voxel quantisation


def test_single_obstacle_detour():
    vm = VoxelMap(bounds=((-1, 1), (-1, 1), (0, 2)), resolution=0.05)
    obstacles = [dict(center=[0, 0, 1.0], sigma=[0.2, 0.2, 0.2], weight=1.0)]
    vm.from_obstacle_field(obstacles, threshold=0.3)
    start = np.array([-0.6, 0.0, 1.0])
    goal = np.array([0.6, 0.0, 1.0])
    path = astar_3d(start, goal, vm, safety_margin=0.05)
    assert path is not None
    pw = voxel_path_to_world(path, vm)
    # The path should NOT pass through the obstacle center.
    nearest = np.min(np.linalg.norm(pw - np.array([0, 0, 1.0]), axis=1))
    assert nearest > 0.1  # gives obstacle a wide berth


def test_unreachable_returns_none():
    """Seal the goal off with a wall of obstacles and confirm A* returns None."""
    vm = VoxelMap(bounds=((-0.5, 0.5), (-0.5, 0.5), (0.5, 1.5)), resolution=0.05)
    obstacles = []
    # Wall at x = 0.0, covering the full yz cross section.
    for y in np.linspace(-0.5, 0.5, 25):
        for z in np.linspace(0.6, 1.4, 17):
            obstacles.append(dict(center=[0.0, y, z],
                                   sigma=[0.05, 0.05, 0.05], weight=1.0))
    vm.from_obstacle_field(obstacles, threshold=0.3)
    start = np.array([-0.3, 0.0, 1.0])
    goal = np.array([0.3, 0.0, 1.0])
    path = astar_3d(start, goal, vm, safety_margin=0.05, max_iter=20000)
    assert path is None


def test_start_equals_goal():
    vm = _empty_map()
    p = np.array([0.0, 0.0, 1.0])
    path = astar_3d(p, p, vm, safety_margin=0.0)
    assert path is not None
    assert len(path) == 1


def test_prune_reduces_waypoints():
    vm = _empty_map()
    start = np.array([-0.5, -0.5, 1.0])
    goal = np.array([0.5, 0.5, 1.0])
    path = astar_3d(start, goal, vm, safety_margin=0.0)
    pw = voxel_path_to_world(path, vm)
    pp = prune_path(pw, vm, safety_margin=0.0, max_step_m=2.0)
    # Empty map -> the pruned path should reduce to ~2-3 waypoints.
    assert len(pp) <= max(3, len(pw) // 3)
