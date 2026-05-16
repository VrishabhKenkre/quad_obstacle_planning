import numpy as np
import pytest

from voxelize import VoxelMap


def test_shape_and_resolution():
    vm = VoxelMap(bounds=((-1, 1), (-1, 1), (0, 2)), resolution=0.1)
    assert vm.shape == (20, 20, 20)
    assert vm.resolution == 0.1


def test_world_voxel_roundtrip():
    vm = VoxelMap(bounds=((-2, 2), (-2, 2), (0, 2.5)), resolution=0.05)
    # Each voxel center is at origin + (i + 0.5) * res. Round-trip the
    # voxel-to-world position back through world-to-voxel.
    for ix, iy, iz in [(0, 0, 0), (10, 20, 5), (79, 79, 49)]:
        p = vm.voxel_to_world(ix, iy, iz)
        assert vm.world_to_voxel(p) == (ix, iy, iz)


def test_occupancy_threshold():
    """A single Gaussian centred at origin should produce a roughly
    spherical occupied region whose radius matches the analytic threshold."""
    vm = VoxelMap(bounds=((-1, 1), (-1, 1), (0, 2)), resolution=0.05)
    obstacles = [dict(center=[0, 0, 1.0], sigma=[0.2, 0.2, 0.2], weight=1.0)]
    vm.from_obstacle_field(obstacles, threshold=0.5)
    # Voxel at exact center is occupied.
    assert vm.is_occupied(np.array([0.0, 0.0, 1.0]))
    # Voxel at 1m away is free.
    assert not vm.is_occupied(np.array([0.9, 0.9, 1.0]))


def test_esdf_far_from_obstacles():
    """In free space far from every obstacle the ESDF should equal the
    closest-voxel distance (within the resolution)."""
    vm = VoxelMap(bounds=((-1, 1), (-1, 1), (0, 2)), resolution=0.05)
    obstacles = [dict(center=[0, 0, 1.0], sigma=[0.1, 0.1, 0.1], weight=1.0)]
    vm.from_obstacle_field(obstacles, threshold=0.5)
    vm.compute_esdf()
    # Point well outside the obstacle: distance should be ~ |p - center|
    # minus the obstacle radius (~sigma*sqrt(-2 ln 0.5) ≈ 0.117 for sigma=0.1).
    p = np.array([0.5, 0.0, 1.0])
    d = vm.query_esdf(p)
    assert d > 0.3
    assert d < 0.55


def test_esdf_inside_obstacle_negative():
    vm = VoxelMap(bounds=((-1, 1), (-1, 1), (0, 2)), resolution=0.05)
    obstacles = [dict(center=[0, 0, 1.0], sigma=[0.25, 0.25, 0.25], weight=1.0)]
    vm.from_obstacle_field(obstacles, threshold=0.3)
    vm.compute_esdf()
    d_center = vm.query_esdf(np.array([0.0, 0.0, 1.0]))
    assert d_center < 0.0  # inside obstacle


def test_esdf_gradient_points_outward():
    """ESDF gradient at a point near an obstacle should point away from it."""
    vm = VoxelMap(bounds=((-1, 1), (-1, 1), (0, 2)), resolution=0.05)
    obstacles = [dict(center=[0, 0, 1.0], sigma=[0.2, 0.2, 0.2], weight=1.0)]
    vm.from_obstacle_field(obstacles, threshold=0.3)
    vm.compute_esdf()
    p = np.array([0.4, 0.0, 1.0])
    g = vm.query_esdf_gradient(p)
    # Outward direction from obstacle to p is +x.
    assert g[0] > 0.0
    assert np.linalg.norm(g[1:]) < abs(g[0])


def test_inflate_grows_occupancy():
    vm = VoxelMap(bounds=((-1, 1), (-1, 1), (0, 2)), resolution=0.05)
    obstacles = [dict(center=[0, 0, 1.0], sigma=[0.15, 0.15, 0.15], weight=1.0)]
    vm.from_obstacle_field(obstacles, threshold=0.3)
    occ_no = vm.occupancy.sum()
    occ_15 = vm.inflate(0.15).sum()
    assert occ_15 > occ_no
