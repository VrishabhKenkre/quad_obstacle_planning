"""
voxelize.py -- VoxelMap + ESDF over the Gaussian obstacle field.

Builds a 3D occupancy grid by sampling the analytic obstacle field at each
voxel center, then computes a Euclidean signed-distance field (ESDF) via
scipy.ndimage.distance_transform_edt.

API:
    VoxelMap(bounds, resolution)
        .from_obstacle_field(obstacles, threshold=0.3)
        .compute_esdf()
        .query_esdf(p)            -> trilinear interpolated distance [m]
        .query_esdf_gradient(p)   -> central-difference 3-vector
        .world_to_voxel(p)        -> (ix, iy, iz)
        .voxel_to_world(ix, iy, iz) -> 3-vector world point
        .is_occupied(p)           -> bool

Coordinate convention: voxel centers at world coordinates
    p = origin + (i + 0.5) * resolution
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


class VoxelMap:
    """3D occupancy grid + ESDF over an axis-aligned bounding box."""

    def __init__(self, bounds=((-2.0, 2.0), (-2.0, 2.0), (0.0, 2.5)),
                 resolution: float = 0.05):
        self.bounds = np.array(bounds, dtype=np.float64)
        self.resolution = float(resolution)
        self.origin = self.bounds[:, 0].copy()
        size = self.bounds[:, 1] - self.bounds[:, 0]
        self.shape = tuple(int(np.round(s / self.resolution)) for s in size)
        self.nx, self.ny, self.nz = self.shape
        self.occupancy = np.zeros(self.shape, dtype=bool)
        self.esdf = None  # populated by compute_esdf()

    # ---- Coordinate transforms -----------------------------------------

    def world_to_voxel(self, p):
        """World coordinate -> integer voxel index (floored). Out-of-range
        indices are clipped to grid extent so callers can probe safely."""
        p = np.asarray(p, dtype=np.float64)
        idx = np.floor((p - self.origin) / self.resolution).astype(np.int64)
        for k in range(3):
            idx[..., k] = np.clip(idx[..., k], 0, self.shape[k] - 1)
        return tuple(int(x) for x in idx) if idx.ndim == 1 else idx

    def voxel_to_world(self, ix, iy, iz):
        return self.origin + (np.array([ix, iy, iz]) + 0.5) * self.resolution

    def in_bounds(self, p):
        p = np.asarray(p, dtype=np.float64)
        return bool(np.all(p >= self.bounds[:, 0]) and
                    np.all(p <= self.bounds[:, 1]))

    # ---- Occupancy from obstacle field ---------------------------------

    def from_obstacle_field(self, obstacles, threshold: float = 0.3):
        """Build occupancy by sampling sum-of-Gaussians at every voxel center.

        Voxel is occupied iff field_value > threshold. The Gaussian field is
        the same one used by src/obstacle_course.obstacle_field_value.
        """
        # Voxel-center coordinates as a (nx, ny, nz, 3) tensor.
        ix = np.arange(self.nx) + 0.5
        iy = np.arange(self.ny) + 0.5
        iz = np.arange(self.nz) + 0.5
        X = self.origin[0] + ix * self.resolution
        Y = self.origin[1] + iy * self.resolution
        Z = self.origin[2] + iz * self.resolution
        # Accumulate field directly without forming the full (nx,ny,nz,3) tensor.
        field = np.zeros(self.shape, dtype=np.float32)
        for obs in obstacles:
            c = np.asarray(obs['center'], dtype=np.float32)
            s = np.asarray(obs['sigma'], dtype=np.float32)
            w = float(obs.get('weight', 1.0))
            dx = ((X - c[0]) / s[0])[:, None, None]
            dy = ((Y - c[1]) / s[1])[None, :, None]
            dz = ((Z - c[2]) / s[2])[None, None, :]
            field += w * np.exp(-0.5 * (dx * dx + dy * dy + dz * dz)).astype(np.float32)
        self.field_cache = field
        self.occupancy = field > threshold
        self.threshold = float(threshold)
        self.esdf = None
        return self.occupancy

    def inflate(self, margin_m: float):
        """Return a new occupancy array with obstacles inflated by margin_m.
        Uses an EDT of the obstacle-free region: any voxel within margin_m
        of an obstacle becomes occupied."""
        if margin_m <= 0:
            return self.occupancy.copy()
        free = ~self.occupancy
        # distance_transform_edt returns distance (in voxel units) from
        # each non-zero pixel to the nearest zero pixel.
        d_free_to_obs = distance_transform_edt(free) * self.resolution
        return d_free_to_obs <= margin_m

    # ---- ESDF -----------------------------------------------------------

    def compute_esdf(self):
        """Euclidean signed-distance field [m]. Positive in free space
        (distance to nearest obstacle), zero or negative inside obstacles
        (so we keep a single sign convention: positive = free distance,
        zero = boundary, negative = penetration depth)."""
        occ = self.occupancy
        # Distance from each free voxel to nearest obstacle voxel.
        d_free = distance_transform_edt(~occ) * self.resolution
        # Distance from each obstacle voxel to nearest free voxel (penetration).
        d_obs = distance_transform_edt(occ) * self.resolution
        self.esdf = (d_free - d_obs).astype(np.float32)
        return self.esdf

    def _trilinear(self, field, p):
        """Trilinear interpolation of a scalar field at world point p.
        Falls back to nearest-clamped value if p is outside the grid."""
        p = np.asarray(p, dtype=np.float64)
        # Continuous index (voxel-center frame).
        c = (p - self.origin) / self.resolution - 0.5
        # Lower integer corner, clipped so we always have a valid 8-cube.
        i0 = np.floor(c).astype(np.int64)
        for k in range(3):
            i0[k] = np.clip(i0[k], 0, self.shape[k] - 2)
        f = c - i0
        for k in range(3):
            f[k] = np.clip(f[k], 0.0, 1.0)
        ix, iy, iz = i0
        fx, fy, fz = f
        c000 = field[ix,   iy,   iz]
        c100 = field[ix+1, iy,   iz]
        c010 = field[ix,   iy+1, iz]
        c110 = field[ix+1, iy+1, iz]
        c001 = field[ix,   iy,   iz+1]
        c101 = field[ix+1, iy,   iz+1]
        c011 = field[ix,   iy+1, iz+1]
        c111 = field[ix+1, iy+1, iz+1]
        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return float(c0 * (1 - fz) + c1 * fz)

    def query_esdf(self, p):
        if self.esdf is None:
            raise RuntimeError("Call compute_esdf() first")
        return self._trilinear(self.esdf, p)

    def query_esdf_gradient(self, p):
        """Central-difference gradient of the ESDF, in m/m (dimensionless)."""
        if self.esdf is None:
            raise RuntimeError("Call compute_esdf() first")
        h = self.resolution
        p = np.asarray(p, dtype=np.float64)
        grad = np.zeros(3)
        for k in range(3):
            pp = p.copy(); pm = p.copy()
            pp[k] += h; pm[k] -= h
            grad[k] = (self._trilinear(self.esdf, pp)
                       - self._trilinear(self.esdf, pm)) / (2.0 * h)
        return grad

    def is_occupied(self, p):
        p = np.asarray(p, dtype=np.float64)
        if not self.in_bounds(p):
            return False
        ix, iy, iz = self.world_to_voxel(p)
        return bool(self.occupancy[ix, iy, iz])

    # ---- Reporting ------------------------------------------------------

    def stats(self):
        n_occ = int(self.occupancy.sum())
        total = int(np.prod(self.shape))
        return dict(shape=self.shape, resolution=self.resolution,
                    occupied_voxels=n_occ, free_voxels=total - n_occ,
                    occupied_frac=n_occ / total)


if __name__ == '__main__':
    # Smoke test against obstacle_course's field generator.
    import sys, time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    from obstacle_course import make_obstacles, obstacle_field_value

    obstacles = make_obstacles(seed=42)
    vm = VoxelMap()
    t0 = time.time()
    vm.from_obstacle_field(obstacles, threshold=0.3)
    t_occ = time.time() - t0

    t0 = time.time()
    vm.compute_esdf()
    t_esdf = time.time() - t0

    print(f"shape={vm.shape}, res={vm.resolution} m")
    print(f"occupancy build: {t_occ*1000:.1f} ms")
    print(f"ESDF build:      {t_esdf*1000:.1f} ms")
    print(f"stats: {vm.stats()}")

    # Probe at goal and at an obstacle center.
    p_goal = np.array([1.5, 1.5, 1.0])
    p_obs = np.array(obstacles[0]['center'])
    print(f"ESDF at goal {p_goal}: {vm.query_esdf(p_goal):.3f} m")
    print(f"ESDF at obstacle center {p_obs}: {vm.query_esdf(p_obs):.3f} m")
    print(f"Field at goal: {obstacle_field_value(p_goal, obstacles):.3f}")
    print(f"Field at obstacle: {obstacle_field_value(p_obs, obstacles):.3f}")
