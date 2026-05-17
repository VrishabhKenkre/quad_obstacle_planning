"""
randomize_astar.py -- generate K different A* paths for the same obstacle
layout, producing multi-modal training data.

The stock planning/astar.astar_3d breaks ties by node-insertion counter, so
two runs with identical inputs always yield the same path. We add two
sources of stochasticity that respect optimality bounds:

  (1) Noise on the heuristic:  h'(n) = h(n) * (1 + eps_n)
      with eps_n ~ U(-eps_max, +eps_max).
      Bounded heuristic perturbation -> bounded suboptimality.

  (2) Optional directional bias (used for decision-point seeds):
      h'(n) += bias_lambda * (n - start) . direction
      A negative dot product pulls the search toward `direction`.

Filtering: only paths whose pruned world-length is within
`length_ratio_max` * shortest_length are kept, so we drop genuinely bad
alternatives but keep "go-left" vs "go-right" variants of comparable cost.

API:
    randomized_astar_paths(start, goal, voxel_map, k, ...)
        -> list[np.ndarray]   (each (Mi,3) world-frame pruned waypoints)
"""
from __future__ import annotations

import heapq
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'planning'))

from astar import _NEIGHBOURS_26, _euclid, voxel_path_to_world, prune_path  # noqa: E402


def _astar_3d_perturbed(start_xyz, goal_xyz, voxel_map,
                        safety_margin: float = 0.15,
                        h_noise: float = 0.05,
                        edge_noise: float = 0.15,
                        bias_dir: Optional[np.ndarray] = None,
                        bias_lambda: float = 0.0,
                        z_penalty_per_m: float = 0.0,
                        rng: Optional[np.random.Generator] = None,
                        max_iter: int = 2_000_000):
    """A* with noisy/biased costs. Returns voxel-index path or None.

    Two sources of perturbation:
      * `h_noise`   -- multiplicative noise on the (admissible) heuristic.
                        Changes search order, can rarely change output.
      * `edge_noise` -- multiplicative noise on each *traversal* edge weight.
                        Changes the actual cost function, so it really does
                        produce different paths. Bounded suboptimality:
                        worst-case cost increases by ~edge_noise.

    `bias_dir`/`bias_lambda` add a g-cost penalty for moves that go against
    the desired direction (used to force left/right detours at
    decision-point seeds).  `z_penalty_per_m` penalises altitude gain so the
    search prefers in-plane detours over climbing.
    """
    if rng is None:
        rng = np.random.default_rng()

    occ = voxel_map.inflate(safety_margin)
    nx, ny, nz = occ.shape

    s = voxel_map.world_to_voxel(start_xyz)
    g = voxel_map.world_to_voxel(goal_xyz)
    if s == g:
        return [s]
    occ[s] = False
    occ[g] = False

    s_world = np.asarray(voxel_map.voxel_to_world(*s))
    g_world = np.asarray(voxel_map.voxel_to_world(*g))
    bias_dir_n = None
    if bias_dir is not None and bias_lambda != 0.0:
        d = np.asarray(bias_dir, dtype=np.float64)
        nrm = np.linalg.norm(d)
        if nrm > 1e-9:
            bias_dir_n = d / nrm

    res = voxel_map.resolution

    def heuristic_world(node):
        # admissible Euclidean heuristic, scaled by an optional noise factor
        # in [1 - h_noise, 1 + h_noise]. Stays nearly admissible.
        dvox = (node[0] - g[0], node[1] - g[1], node[2] - g[2])
        h = float(np.sqrt(dvox[0]*dvox[0] + dvox[1]*dvox[1] + dvox[2]*dvox[2])) * res
        if h_noise > 0:
            h *= (1.0 + float(rng.uniform(-h_noise, h_noise)))
        return h

    open_heap = []
    counter = 0
    g_cost = {s: 0.0}
    came_from = {}
    heapq.heappush(open_heap, (heuristic_world(s), counter, s))
    counter += 1

    iters = 0
    while open_heap:
        iters += 1
        if iters > max_iter:
            return None
        _, _, current = heapq.heappop(open_heap)
        if current == g:
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
            # Multiplicative edge noise: changes the cost surface so A*
            # actually returns a different shortest path.
            ew = w * res
            if edge_noise > 0:
                ew *= (1.0 + float(rng.uniform(-edge_noise, edge_noise)))
            # Direction bias as a g-cost penalty for moves that point away
            # from bias_dir.
            if bias_dir_n is not None:
                step_world = np.array([dx, dy, dz], dtype=np.float64) * res
                dot = float(np.dot(step_world, bias_dir_n))
                # discount moves toward bias, penalise moves away
                ew += bias_lambda * (-dot)
            # Penalise altitude gain (used so decision-point seeds detour
            # in-plane rather than fly over the obstacle).
            if z_penalty_per_m > 0 and dz > 0:
                ew += z_penalty_per_m * (dz * res)
            ew = max(ew, 1e-6)

            tentative = gc_curr + ew
            prev = g_cost.get(neigh)
            if prev is not None and tentative >= prev:
                continue
            g_cost[neigh] = tentative
            came_from[neigh] = current
            f = tentative + heuristic_world(neigh)
            heapq.heappush(open_heap, (f, counter, neigh))
            counter += 1
    return None


def _path_signature(pw: np.ndarray, n_bins: int = 5, q_per_m: int = 3) -> tuple:
    """Cheap fingerprint of a world-path so we can dedup near-duplicate
    variants. Bins (x, y) at evenly-spaced fractions of arc length, at a
    coarse quantisation (default 3/m = 33 cm cells) so two genuinely
    similar paths are matched as duplicates but left/right variants are
    not."""
    if len(pw) < 2:
        return tuple()
    seg = np.linalg.norm(np.diff(pw, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    L = cum[-1]
    if L < 1e-6:
        return tuple()
    samples = []
    for f in np.linspace(0.2, 0.8, n_bins):
        target = f * L
        i = int(np.searchsorted(cum, target))
        i = max(1, min(i, len(cum) - 1))
        t = (target - cum[i-1]) / max(cum[i] - cum[i-1], 1e-9)
        p = pw[i-1] + t * (pw[i] - pw[i-1])
        samples.append((int(round(p[0] * q_per_m)),
                        int(round(p[1] * q_per_m))))
    return tuple(samples)


def randomized_astar_paths(
        start: np.ndarray,
        goal: np.ndarray,
        voxel_map,
        k: int = 3,
        safety_margin: float = 0.15,
        max_step_m: float = 0.5,
        length_ratio_max: float = 1.2,
        h_noise: float = 0.15,
        seed: int = 0,
        max_attempts_multiplier: int = 12,
        forced_bias_pairs: Optional[List[Tuple[np.ndarray, float]]] = None,
        edge_noise: float = 0.15,
        z_penalty_per_m: float = 0.10,
        random_perp_bias_prob: float = 0.5,
        random_perp_bias_lambda: float = 0.30,
        verbose: bool = False,
        ) -> List[np.ndarray]:
    """Return up to k distinct pruned world-frame paths from start to goal.

    Args:
        k: target number of paths to keep.
        length_ratio_max: drop variants longer than this * best length.
        h_noise: half-width of the multiplicative heuristic noise.
        seed: base seed for the internal RNG.
        forced_bias_pairs: optional list of (direction, lambda) tuples to
            force specific path families (used for decision-point seeds).
            If supplied, these are tried first; random variants fill the rest.
        max_attempts_multiplier: cap on number of A* re-runs (k * mult).
    """
    rng_master = np.random.default_rng(seed)
    kept: List[np.ndarray] = []
    sigs = set()
    lengths: List[float] = []
    best_len = None

    attempts = []
    if forced_bias_pairs:
        for d, lam in forced_bias_pairs:
            attempts.append(dict(h_noise=0.0, edge_noise=0.0,
                                 bias_dir=d, bias_lambda=lam,
                                 z_penalty_per_m=z_penalty_per_m))
    # Fill the remainder with edge-noise attempts. Half of them get an
    # additional random perpendicular bias to encourage diverse families.
    n_random = max(0, k * max_attempts_multiplier - len(attempts))
    sg = np.asarray(goal) - np.asarray(start)
    perp = np.array([-sg[1], sg[0], 0.0])
    pnrm = np.linalg.norm(perp)
    perp_unit = perp / pnrm if pnrm > 1e-9 else None
    pre_rng = np.random.default_rng(seed + 17)
    for _ in range(n_random):
        bias_dir = None
        bias_lambda = 0.0
        if perp_unit is not None and pre_rng.random() < random_perp_bias_prob:
            bias_dir = perp_unit * (1.0 if pre_rng.random() < 0.5 else -1.0)
            bias_lambda = random_perp_bias_lambda * pre_rng.uniform(0.4, 1.2)
        attempts.append(dict(h_noise=h_noise, edge_noise=edge_noise,
                             bias_dir=bias_dir, bias_lambda=bias_lambda,
                             z_penalty_per_m=z_penalty_per_m))

    for cfg in attempts:
        if len(kept) >= k:
            break
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        vox_path = _astar_3d_perturbed(
            start, goal, voxel_map,
            safety_margin=safety_margin,
            h_noise=cfg['h_noise'],
            edge_noise=cfg.get('edge_noise', edge_noise),
            bias_dir=cfg['bias_dir'],
            bias_lambda=cfg['bias_lambda'],
            z_penalty_per_m=cfg.get('z_penalty_per_m', z_penalty_per_m),
            rng=rng,
        )
        if vox_path is None:
            continue
        pw = voxel_path_to_world(vox_path, voxel_map)
        pw = prune_path(pw, voxel_map, safety_margin=safety_margin,
                        max_step_m=max_step_m)
        # snap exact endpoints
        pw[0] = start
        pw[-1] = goal
        length = float(np.sum(np.linalg.norm(np.diff(pw, axis=0), axis=1)))
        if best_len is None or length < best_len:
            best_len = length
        if length > length_ratio_max * best_len:
            if verbose:
                print(f"  drop variant len={length:.2f} (best={best_len:.2f})")
            continue
        sig = _path_signature(pw)
        if sig in sigs:
            continue
        sigs.add(sig)
        kept.append(pw)
        lengths.append(length)
        if verbose:
            print(f"  keep variant {len(kept)}: len={length:.2f}, "
                  f"{len(pw)} waypoints")

    # Final filter: any path now exceeding the ratio (best_len may have improved later)
    if best_len is not None:
        kept_filt = []
        for pw, L in zip(kept, lengths):
            if L <= length_ratio_max * best_len:
                kept_filt.append(pw)
        kept = kept_filt

    return kept


def decision_point_layout(seed: int,
                          start: np.ndarray = np.array([-1.5, -1.5, 1.0]),
                          goal: np.ndarray = np.array([1.5, 1.5, 1.0])):
    """Build a synthetic obstacle layout with one wide obstacle straddling
    the start-goal line, forcing a clearly bimodal left/right choice.

    Returns (obstacles, left_bias, right_bias) where the bias pairs are
    (direction, lambda) tuples suitable for forced_bias_pairs.

    The obstacle is placed at the midpoint, with sigma large enough that
    the safety-inflated occupancy genuinely blocks the straight path.
    A small amount of layout jitter (seeded) gives the 30 decision-point
    seeds some variety without losing the bimodality.
    """
    rng = np.random.default_rng(int(seed) + 10_000)
    mid = 0.5 * (start + goal)
    # small jitter in xy keeps the 30 seeds non-identical without losing
    # the bimodality; z is held near 1.0 (start/goal height).
    jitter = np.array([rng.uniform(-0.10, 0.10),
                       rng.uniform(-0.10, 0.10),
                       0.0])
    center = mid + jitter
    sigma_xy = float(rng.uniform(0.42, 0.55))
    # Tall column (sigma_z large) so A* cannot trivially fly over the
    # obstacle: the only sub-cost detours are left or right in xy.
    sigma_z = 1.20
    obstacles = [dict(center=center.tolist(),
                      sigma=[sigma_xy, sigma_xy, sigma_z],
                      weight=1.6)]
    # 1-2 small filler obstacles, kept well clear of the central gap
    for _ in range(int(rng.integers(1, 3))):
        cx = rng.uniform(-1.0, 1.0)
        cy = rng.uniform(-1.0, 1.0)
        if np.linalg.norm([cx - center[0], cy - center[1]]) < 1.0:
            continue
        cz = rng.uniform(0.7, 1.3)
        s = float(rng.uniform(0.18, 0.25))
        obstacles.append(dict(center=[cx, cy, cz], sigma=[s, s, s],
                              weight=1.0))

    # bias directions: perpendicular to (goal-start) in xy plane
    sg = goal - start
    perp = np.array([-sg[1], sg[0], 0.0])
    perp = perp / max(np.linalg.norm(perp), 1e-9)
    # lam was 1.5 in v1 of the dataset, which paired with safety_margin=0.15
    # produced left/right alternatives ~0.17m from the obstacle surface --
    # too tight, so the diffusion student sometimes grazed obstacles even
    # when sampling the "correct" mode. Bumping to 2.5 spreads the
    # alternatives further apart; the actual clearance bound is set by
    # safety_margin (now 0.30m for dp collection, see collect_planner_data).
    lam = 2.5
    left_bias = (+perp, lam)
    right_bias = (-perp, lam)
    return obstacles, left_bias, right_bias


if __name__ == '__main__':
    # Smoke test.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'planning'))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    from voxelize import VoxelMap
    from obstacle_course import make_obstacles

    start = np.array([-1.5, -1.5, 1.0])
    goal = np.array([1.5, 1.5, 1.0])

    print("=== Random A* on seed 42 (K=3) ===")
    obstacles = make_obstacles(seed=42)
    vm = VoxelMap()
    vm.from_obstacle_field(obstacles)
    vm.compute_esdf()
    paths = randomized_astar_paths(start, goal, vm, k=3, h_noise=0.08,
                                   seed=42, verbose=True)
    for i, pw in enumerate(paths):
        L = float(np.sum(np.linalg.norm(np.diff(pw, axis=0), axis=1)))
        print(f"  variant {i}: len={L:.3f}, n_wp={len(pw)}")

    print("\n=== Decision-point layout seed 0 ===")
    obs, lb, rb = decision_point_layout(seed=0)
    vm2 = VoxelMap()
    vm2.from_obstacle_field(obs)
    vm2.compute_esdf()
    paths = randomized_astar_paths(start, goal, vm2, k=2,
                                   forced_bias_pairs=[lb, rb],
                                   length_ratio_max=1.3,
                                   verbose=True)
    print(f"  produced {len(paths)} forced-bias variants")
    for i, pw in enumerate(paths):
        # avg y-coordinate at the midpoint tells us left/right
        mid_y = float(np.interp(0.5, np.linspace(0, 1, len(pw)), pw[:, 1]))
        print(f"  variant {i}: mid_y={mid_y:+.2f}")
