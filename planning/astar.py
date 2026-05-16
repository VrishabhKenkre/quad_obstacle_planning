"""
astar.py -- 3D A* on the voxel grid produced by voxelize.VoxelMap.

Heuristic: Euclidean distance to goal (admissible for 26-connectivity).
Neighbours: 26-connectivity with edge weights = euclidean distance.

API:
    astar_3d(start_xyz, goal_xyz, voxel_map, safety_margin=0.15)
        -> list[(ix,iy,iz)] voxel path, or None if no path
    voxel_path_to_world(path, voxel_map) -> (N, 3) array
"""
from __future__ import annotations

import heapq
import numpy as np


# 26-connected neighbour offsets and their L2 distances.
def _build_neighbours():
    offs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offs.append((dx, dy, dz, float(np.sqrt(dx*dx + dy*dy + dz*dz))))
    return tuple(offs)


_NEIGHBOURS_26 = _build_neighbours()


def _euclid(a, b):
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2))


def astar_3d(start_xyz, goal_xyz, voxel_map, safety_margin: float = 0.15,
             max_iter: int = 2_000_000):
    """Return a list of voxel indices from start to goal on `voxel_map`, or None.

    Args:
        start_xyz: world-frame start (3,).
        goal_xyz:  world-frame goal  (3,).
        voxel_map: VoxelMap with .occupancy filled.
        safety_margin: extra clearance [m] applied by morphological inflation.

    The inflated occupancy is used only for planning; voxel_map.occupancy is
    untouched. Start/goal voxels are forced free even after inflation (the
    drone starts/ends at user-specified poses that we assume are valid).
    """
    occ = voxel_map.inflate(safety_margin)
    nx, ny, nz = occ.shape

    s = voxel_map.world_to_voxel(start_xyz)
    g = voxel_map.world_to_voxel(goal_xyz)

    if s == g:
        return [s]

    # Force endpoints free.
    occ[s] = False
    occ[g] = False

    # Open set: heap entries (f, counter, node).
    open_heap = []
    counter = 0
    g_cost = {s: 0.0}
    came_from = {}
    h0 = _euclid(s, g)
    heapq.heappush(open_heap, (h0, counter, s))
    counter += 1

    iters = 0
    while open_heap:
        iters += 1
        if iters > max_iter:
            return None
        _, _, current = heapq.heappop(open_heap)
        if current == g:
            # Reconstruct.
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy, cz = current
        gc_curr = g_cost[current]
        for dx, dy, dz, w in _NEIGHBOURS_26:
            nxn, nyn, nzn = cx + dx, cy + dy, cz + dz
            if not (0 <= nxn < nx and 0 <= nyn < ny and 0 <= nzn < nz):
                continue
            if occ[nxn, nyn, nzn]:
                continue
            neigh = (nxn, nyn, nzn)
            tentative = gc_curr + w
            prev = g_cost.get(neigh)
            if prev is not None and tentative >= prev:
                continue
            g_cost[neigh] = tentative
            came_from[neigh] = current
            f = tentative + _euclid(neigh, g)
            heapq.heappush(open_heap, (f, counter, neigh))
            counter += 1

    return None


def voxel_path_to_world(path, voxel_map) -> np.ndarray:
    """Convert a list of voxel indices to (N, 3) world coordinates."""
    pts = np.empty((len(path), 3), dtype=np.float64)
    for i, (ix, iy, iz) in enumerate(path):
        pts[i] = voxel_map.voxel_to_world(ix, iy, iz)
    return pts


def prune_path(path_world: np.ndarray, voxel_map, safety_margin: float = 0.15,
               max_step_m: float = 0.4) -> np.ndarray:
    """Line-of-sight prune the world-frame path: drop intermediate vertices
    that the next vertex can be reached from without entering an obstacle
    (inflated by `safety_margin`). Limits the resulting segment length to
    `max_step_m` so the smoother gets enough waypoints to follow curves."""
    if len(path_world) <= 2:
        return path_world.copy()
    occ_inflated = voxel_map.inflate(safety_margin)
    keep = [0]
    i = 0
    n = len(path_world)
    while i < n - 1:
        # Greedy: extend j as far as collision-free and within max_step_m.
        j_best = i + 1
        for j in range(i + 2, n):
            seg_len = np.linalg.norm(path_world[j] - path_world[i])
            if seg_len > max_step_m:
                break
            if _segment_clear(path_world[i], path_world[j],
                              occ_inflated, voxel_map):
                j_best = j
            else:
                break
        keep.append(j_best)
        i = j_best
    return path_world[keep]


def _segment_clear(p0, p1, occ, voxel_map) -> bool:
    """Sample the segment p0->p1 at 0.5*voxel_res spacing; reject if any
    sample lands in an occupied voxel of `occ`."""
    seg = p1 - p0
    n_samples = max(2, int(np.ceil(np.linalg.norm(seg) / (0.5 * voxel_map.resolution))))
    for k in range(n_samples + 1):
        t = k / n_samples
        p = p0 + t * seg
        ix, iy, iz = voxel_map.world_to_voxel(p)
        if occ[ix, iy, iz]:
            return False
    return True


if __name__ == '__main__':
    import sys, time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    from obstacle_course import make_obstacles
    from voxelize import VoxelMap

    obstacles = make_obstacles(seed=42)
    vm = VoxelMap()
    vm.from_obstacle_field(obstacles, threshold=0.3)
    vm.compute_esdf()

    start = np.array([-1.5, -1.5, 1.0])
    goal = np.array([1.5, 1.5, 1.0])

    t0 = time.time()
    path = astar_3d(start, goal, vm, safety_margin=0.15)
    t_astar = time.time() - t0
    print(f"A* solved in {t_astar*1000:.1f} ms ({len(path)} voxels)")

    pw = voxel_path_to_world(path, vm)
    plen = float(np.sum(np.linalg.norm(np.diff(pw, axis=0), axis=1)))
    print(f"raw path length: {plen:.3f} m")

    pp = prune_path(pw, vm, safety_margin=0.15)
    plen_p = float(np.sum(np.linalg.norm(np.diff(pp, axis=0), axis=1)))
    print(f"pruned: {len(pp)} waypoints, length {plen_p:.3f} m")
