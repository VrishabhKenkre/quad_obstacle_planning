"""
collect_planner_data.py -- generate a multi-modal (obs, action) dataset
from the hierarchical planner teacher.

Pipeline per (seed, A*-variant):
    1. build VoxelMap+ESDF for the seed's obstacles
    2. run randomized A* (planning/astar variant -> several paths)
    3. min-snap smooth -> 12xN reference
    4. SE(3) NMPC tracking loop. At each step:
         - compute obs = [state_error_to_goal(12), next 4 ref waypoints(12)]
         - obs has shape (24,)
         - query NMPC -> u_mj (4-D actuator command)
         - normalise u_mj to action ∈ [-1,1] using CrazyflieEnv.{u_min,u_max}
         - SAVE (obs, action_clean)   <-- this is the teacher label
         - inject DART noise: a_apply = clip(action_clean + N(0, sigma_dart))
         - step env with a_apply
    5. append to global arrays.

Decision-point seeds: 30 hand-crafted obstacle layouts (single tall obstacle
straddling the start-goal line) where we deliberately keep both the
"go left" and "go right" planner trajectories.

Output:
  data/planner_dataset_v1.npz with arrays
    observations   (N, 24)  float32
    actions        (N, 4)   float32  in [-1,1]
    seeds          (N,)     int32
    variant_ids    (N,)     int32   distinguishes K paths per seed
    is_decision_pt (N,)     bool
    step_indices   (N,)     int32

Usage:
    python distillation/collect_planner_data.py            # full run
    python distillation/collect_planner_data.py --quick    # 5 seeds + 2 dp
    python distillation/collect_planner_data.py --seeds N --decision-seeds M
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / 'planning'))
sys.path.insert(0, str(_HERE.parent / 'src'))

from voxelize import VoxelMap
from min_snap import smooth_waypoints
from nonlinear_mpc import SE3_NMPC, rotors_to_mujoco, M as M_QUAD, G as G_QUAD
from quad_env import CrazyflieEnv
from obstacle_course import make_obstacles, obstacle_field_value
from hierarchical_ctrl import _mujoco_state_to_nmpc_state

from randomize_astar import randomized_astar_paths, decision_point_layout


# ---- Constants -----------------------------------------------------------

START = np.array([-1.5, -1.5, 1.0])
GOAL = np.array([1.5, 1.5, 1.0])

# Observation: 12 state + 1 sdf + 3 sdf-gradient + 8 sdf-lookahead-probes
# = 24-D. Carries obstacle-awareness but NOT planner intent, so the same
# obs is reachable from multiple planner trajectories (left vs right) ->
# multi-modal targets the MLP cannot represent.
OBS_DIM = 24
ACT_DIM = 4

# DART noise schedule (in normalized [-1,1] action space).
# Modest amplitude so trajectories stay roughly stable but visit
# off-distribution states.
DART_SIGMA = np.array([0.030, 0.040, 0.040, 0.030], dtype=np.float32)

# 8 SDF lookahead-probe offsets in METRES (world-frame).
# 6 cardinal points at +/-0.30 m and two further-out probes along x,y.
PROBE_OFFSETS = np.array([
    [+0.30,  0.00,  0.00],
    [-0.30,  0.00,  0.00],
    [ 0.00, +0.30,  0.00],
    [ 0.00, -0.30,  0.00],
    [ 0.00,  0.00, +0.30],
    [ 0.00,  0.00, -0.30],
    [+0.50, +0.50,  0.00],
    [-0.50, -0.50,  0.00],
], dtype=np.float32)


def make_observation(state_mj: np.ndarray, goal: np.ndarray,
                     voxel_map) -> np.ndarray:
    """Compute 24-D observation: state + SDF context (no planner ref leak).

    Layout:
      obs[0:3]   = pos - goal         (position error to goal)
      obs[3:6]   = vel
      obs[6:9]   = euler              (roll, pitch, yaw)
      obs[9:12]  = omega              (body rates)
      obs[12]    = sdf(p)             (distance to nearest obstacle, m)
      obs[13:16] = grad sdf(p)        (3-vector, dimensionless)
      obs[16:24] = sdf at 8 lookahead probes around p

    Multi-modality emerges because two planner variants (e.g. left and
    right around the same central obstacle) pass through nearly the same
    state with nearly the same SDF context -- but take opposite actions.
    """
    pos = state_mj[0:3]
    vel = state_mj[3:6]
    euler = state_mj[6:9]
    omega = state_mj[9:12]
    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[0:3] = (pos - goal).astype(np.float32)
    obs[3:6] = vel.astype(np.float32)
    obs[6:9] = euler.astype(np.float32)
    obs[9:12] = omega.astype(np.float32)
    obs[12] = float(voxel_map.query_esdf(pos))
    grad = voxel_map.query_esdf_gradient(pos)
    obs[13:16] = grad.astype(np.float32)
    for k in range(PROBE_OFFSETS.shape[0]):
        obs[16 + k] = float(voxel_map.query_esdf(pos + PROBE_OFFSETS[k]))
    return obs


def take_ref_window(ref: np.ndarray, i: int, N: int):
    """(N+1)-column slice of the reference for NMPC, padded at the tail."""
    n_total = ref.shape[1]
    rp = np.zeros((3, N + 1))
    rv = np.zeros((3, N + 1))
    for k in range(N + 1):
        j = min(i + k, n_total - 1)
        rp[:, k] = ref[0:3, j]
        rv[:, k] = ref[3:6, j]
    return rp, rv


# ---- Single rollout ------------------------------------------------------

def rollout_planner(nmpc: SE3_NMPC,
                    env: CrazyflieEnv,
                    voxel_map,
                    ref: np.ndarray,
                    goal: np.ndarray,
                    dart_sigma: np.ndarray,
                    rng: np.random.Generator,
                    nmpc_N: int = 15,
                    nmpc_dt: float = 0.02,
                    T_max: float = 10.0,
                    max_pos_err: float = 2.0,
                    ) -> dict:
    """Track `ref` with NMPC, recording (obs, clean_action) and stepping
    the env with DART-noisy actions. Returns dict of arrays.

    The teacher label saved is the *clean* NMPC action at the noisy state.
    """
    nmpc.prev_X = None
    nmpc.prev_U = None
    state_mj = env.reset(pos=ref[0:3, 0])

    n_steps = min(int(T_max / nmpc_dt), ref.shape[1])
    obss = np.empty((n_steps, OBS_DIM), dtype=np.float32)
    acts = np.empty((n_steps, ACT_DIM), dtype=np.float32)

    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0

    saved = 0
    aborted = False
    for i in range(n_steps):
        obs = make_observation(state_mj, goal, voxel_map)
        rp_win, rv_win = take_ref_window(ref, i, nmpc_N)
        x13 = _mujoco_state_to_nmpc_state(state_mj)
        u_rotors, info = nmpc.solve(x13, rp_win, rv_win)
        u_mj = rotors_to_mujoco(u_rotors)
        # normalise to [-1,1]
        action_clean = np.clip((u_mj - u_mid) / u_half, -1.0, 1.0).astype(np.float32)
        obss[saved] = obs
        acts[saved] = action_clean
        saved += 1

        # DART noise -> step env with noisy action (clip then un-normalise)
        noise = rng.normal(0.0, dart_sigma).astype(np.float32)
        action_noisy = np.clip(action_clean + noise, -1.0, 1.0)
        u_apply = u_mid + u_half * action_noisy
        u_apply = np.clip(u_apply, env.u_min, env.u_max)
        state_mj = env.step(u_apply)

        # abort guard: if the drone is wildly off-trajectory, dump rollout.
        if (np.linalg.norm(state_mj[0:3] - ref[0:3, min(i, ref.shape[1]-1)])
                > max_pos_err):
            aborted = True
            break

    return dict(
        observations=obss[:saved],
        actions=acts[:saved],
        n_steps=saved,
        aborted=aborted,
        final_pos=state_mj[0:3].copy(),
    )


# ---- Per-seed collection -------------------------------------------------

def collect_for_seed(obstacles: list,
                     voxel_map,
                     paths_world: List[np.ndarray],
                     seed: int,
                     start_variant_id: int,
                     is_decision_pt: bool,
                     dart_sigma: np.ndarray,
                     rng_master: np.random.Generator,
                     nmpc_N: int = 15,
                     nmpc_dt: float = 0.02,
                     ref_avg_speed: float = 0.8,
                     T_max: float = 10.0,
                     verbose: bool = False,
                     ) -> Tuple[List[dict], int]:
    """Run K rollouts (one per path variant) with a single NMPC build.

    Returns (list-of-rollouts, next_variant_id).
    """
    # Build NMPC once per obstacle layout (the obstacle term is baked in).
    t0 = time.perf_counter()
    nmpc = SE3_NMPC(N=nmpc_N, dt=nmpc_dt, obstacles=obstacles,
                    q_pos=300, q_vel=10, q_quat=20, q_omega=0.1,
                    r_thrust=1e3, w_obs=800.0)
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=nmpc_dt)
    t_build = time.perf_counter() - t0

    rollouts = []
    variant_id = start_variant_id
    for pw in paths_world:
        try:
            ref, _ = smooth_waypoints(pw, target_dt=nmpc_dt,
                                      target_avg_speed=ref_avg_speed,
                                      return_meta=True)
        except Exception as e:
            if verbose:
                print(f"  [seed {seed} variant {variant_id}] smoothing failed: {e}")
            variant_id += 1
            continue

        rng = np.random.default_rng(
            int(rng_master.integers(0, 2**31 - 1)))
        t0 = time.perf_counter()
        ro = rollout_planner(nmpc, env, voxel_map, ref, GOAL,
                             dart_sigma, rng,
                             nmpc_N=nmpc_N, nmpc_dt=nmpc_dt, T_max=T_max)
        t_track = time.perf_counter() - t0
        ro.update(dict(seed=seed, variant_id=variant_id,
                       is_decision_pt=is_decision_pt,
                       track_time_s=t_track))
        rollouts.append(ro)
        variant_id += 1
        if verbose:
            final_err = float(np.linalg.norm(ro['final_pos'] - GOAL) * 1000)
            print(f"  [seed {seed} variant {variant_id-1}] "
                  f"{ro['n_steps']} steps, final_err={final_err:.0f} mm, "
                  f"aborted={ro['aborted']}, track={t_track:.1f}s")

    if verbose:
        print(f"  [seed {seed}] nmpc build {t_build:.1f}s, "
              f"{len(rollouts)} rollouts kept")
    return rollouts, variant_id


# ---- Main collect --------------------------------------------------------

def _enumerate_random_seeds(n_random_seeds: int, start_seed: int,
                            skip_eval_seeds: bool) -> List[int]:
    eval_seeds = {42, 7, 13, 99, 256, 128, 314, 2024, 777, 1337}
    out = []
    s = start_seed
    while len(out) < n_random_seeds:
        if not (skip_eval_seeds and s in eval_seeds):
            out.append(s)
        s += 1
    return out


DP_SAFETY_MARGIN = 0.30  # v2: was 0.15; the larger margin pushes the
                          # planner's left/right alternatives wider apart
                          # so the diffusion student never grazes obstacles
                          # even when sampling on the boundary between modes.


def _worker_collect(args):
    """Worker entry: collect for a list of (kind, seed) tasks."""
    (tasks, k_per_seed, T_max, seed_offset) = args
    rng_master = np.random.default_rng(0xC0FFEE + seed_offset)
    results = []
    for kind, s in tasks:
        if kind == 'random':
            obstacles = make_obstacles(seed=s)
            vm = VoxelMap()
            vm.from_obstacle_field(obstacles)
            vm.compute_esdf()
            paths = randomized_astar_paths(
                START, GOAL, vm, k=k_per_seed,
                safety_margin=0.15, h_noise=0.15, edge_noise=0.15,
                seed=s)
            is_dp = False
            tag_seed = s
        else:  # 'decision'
            obstacles, lb, rb = decision_point_layout(seed=s)
            vm = VoxelMap()
            vm.from_obstacle_field(obstacles)
            vm.compute_esdf()
            paths = randomized_astar_paths(
                START, GOAL, vm, k=2,
                safety_margin=DP_SAFETY_MARGIN, length_ratio_max=1.30,
                forced_bias_pairs=[lb, rb],
                z_penalty_per_m=0.4, seed=20_000 + s)
            if len(paths) < 2:
                extra = randomized_astar_paths(
                    START, GOAL, vm, k=2,
                    safety_margin=DP_SAFETY_MARGIN, length_ratio_max=1.30,
                    z_penalty_per_m=0.4, seed=20_000 + s)
                paths = (paths + extra)[:2]
            is_dp = True
            tag_seed = 10**6 + s
        if not paths:
            continue
        rollouts, _ = collect_for_seed(
            obstacles, vm, paths, seed=tag_seed,
            start_variant_id=0,  # caller will reassign globally
            is_decision_pt=is_dp,
            dart_sigma=DART_SIGMA, rng_master=rng_master,
            T_max=T_max, verbose=False)
        for ro in rollouts:
            results.append(dict(
                observations=ro['observations'],
                actions=ro['actions'],
                seed=int(ro['seed']),
                is_decision_pt=is_dp,
                n_steps=int(ro['n_steps']),
                aborted=bool(ro['aborted']),
                final_pos=ro['final_pos'].copy(),
            ))
    return results


def run_collection(n_random_seeds: int = 200,
                   k_per_seed: int = 3,
                   n_decision_seeds: int = 30,
                   out_path: str = 'data/planner_dataset_v1.npz',
                   start_seed: int = 1000,
                   T_max: float = 10.0,
                   verbose: bool = True,
                   skip_eval_seeds: bool = True,
                   n_workers: int = 1,
                   dp_only: bool = False,
                   v1_path: str = 'data/planner_dataset_v1.npz',
                   ) -> dict:
    """Run the full data collection. `skip_eval_seeds` excludes the 10
    canonical eval seeds so the policy never trains on its test set.

    If n_workers > 1, the seed list is sharded across worker processes
    via multiprocessing.Pool. CasADi/IPOPT is single-threaded per process,
    so this gives near-linear speedup until you saturate physical cores.

    When `dp_only=True`, only the decision-point seeds are collected
    here. The non-dp rows are loaded from `v1_path` and concatenated into
    the output, so `out_path` ends up being a drop-in v2 dataset with
    fresh dp samples and unchanged random samples.
    """
    rng_master = np.random.default_rng(0xC0FFEE)
    eval_seeds = {42, 7, 13, 99, 256, 128, 314, 2024, 777, 1337}

    all_obs = []
    all_act = []
    all_seed = []
    all_variant = []
    all_dp = []
    all_step = []
    rollout_stats = []

    next_variant_id = 0

    # Build the (kind, seed) task list and fan out across workers.
    if dp_only:
        random_seeds = []
        tasks = [('decision', j) for j in range(n_decision_seeds)]
        if verbose:
            print(f"[collect] dp-only mode: {n_decision_seeds} dp seeds. "
                  f"Random samples will be copied from {v1_path}.")
    else:
        random_seeds = _enumerate_random_seeds(
            n_random_seeds, start_seed, skip_eval_seeds)
        tasks = ([('random', s) for s in random_seeds]
                 + [('decision', j) for j in range(n_decision_seeds)])
    import multiprocessing as mp
    n_w = max(1, int(n_workers))
    shards = [tasks[i::n_w] for i in range(n_w)]
    worker_args = [(shards[i], k_per_seed, T_max, i) for i in range(n_w)]
    if verbose:
        print(f"[collect] {n_w} workers, {len(tasks)} total seeds")
    if n_w == 1:
        shard_results = [_worker_collect(worker_args[0])]
    else:
        with mp.get_context('spawn').Pool(processes=n_w) as pool:
            shard_results = pool.map(_worker_collect, worker_args)

    global_variant = 0
    for shard in shard_results:
        for ro in shard:
            n = ro['n_steps']
            all_obs.append(ro['observations'])
            all_act.append(ro['actions'])
            all_seed.append(np.full(n, ro['seed'], dtype=np.int32))
            all_variant.append(np.full(n, global_variant, dtype=np.int32))
            all_dp.append(np.full(n, ro['is_decision_pt'], dtype=bool))
            all_step.append(np.arange(n, dtype=np.int32))
            rollout_stats.append(dict(
                seed=int(ro['seed']),
                variant_id=int(global_variant),
                n_steps=int(n),
                aborted=bool(ro['aborted']),
                final_err_mm=float(np.linalg.norm(
                    ro['final_pos'] - GOAL) * 1000),
                is_decision_pt=bool(ro['is_decision_pt']),
            ))
            global_variant += 1
    if verbose:
        n_dp = sum(1 for r in rollout_stats if r['is_decision_pt'])
        print(f"[collect] {len(rollout_stats)} rollouts ({n_dp} dp + "
              f"{len(rollout_stats) - n_dp} random)")

    if not all_obs and not dp_only:
        raise RuntimeError("no rollouts collected")

    if all_obs:
        observations = np.concatenate(all_obs, axis=0)
        actions = np.concatenate(all_act, axis=0)
        seeds = np.concatenate(all_seed, axis=0)
        variant_ids = np.concatenate(all_variant, axis=0)
        is_decision_pt = np.concatenate(all_dp, axis=0)
        step_indices = np.concatenate(all_step, axis=0)
    else:
        observations = np.zeros((0, OBS_DIM), dtype=np.float32)
        actions = np.zeros((0, ACT_DIM), dtype=np.float32)
        seeds = np.zeros(0, dtype=np.int32)
        variant_ids = np.zeros(0, dtype=np.int32)
        is_decision_pt = np.zeros(0, dtype=bool)
        step_indices = np.zeros(0, dtype=np.int32)

    # --- dp-only merge: glue the freshly-collected dp rows onto v1's
    # random rows so the resulting file is a complete v2 dataset.
    if dp_only:
        v1_full = _HERE.parent / v1_path
        if not v1_full.exists():
            raise FileNotFoundError(f"v1 dataset missing: {v1_full}")
        v1 = np.load(v1_full)
        v1_obs = v1['observations']
        v1_act = v1['actions']
        v1_seed = v1['seeds']
        v1_variant = v1['variant_ids']
        v1_dp = v1['is_decision_pt']
        v1_step = v1['step_indices']
        # Drop v1's old dp rows; keep only the random rows.
        keep = ~v1_dp
        v1_obs = v1_obs[keep]; v1_act = v1_act[keep]
        v1_seed = v1_seed[keep]; v1_variant = v1_variant[keep]
        v1_dp = v1_dp[keep]; v1_step = v1_step[keep]
        if verbose:
            print(f"[collect] loaded {v1_obs.shape[0]:,} random samples "
                  f"from v1; merging with {observations.shape[0]:,} new "
                  f"dp samples")
        # Shift the new dp variant ids so they don't clash with v1's.
        max_v1_variant = int(v1_variant.max()) + 1 if v1_variant.size else 0
        variant_ids = variant_ids + max_v1_variant
        observations = np.concatenate([v1_obs, observations], axis=0)
        actions = np.concatenate([v1_act, actions], axis=0)
        seeds = np.concatenate([v1_seed, seeds], axis=0)
        variant_ids = np.concatenate([v1_variant, variant_ids], axis=0)
        is_decision_pt = np.concatenate([v1_dp, is_decision_pt], axis=0)
        step_indices = np.concatenate([v1_step, step_indices], axis=0)

    out_path = str(_HERE.parent / out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        observations=observations,
        actions=actions,
        seeds=seeds,
        variant_ids=variant_ids,
        is_decision_pt=is_decision_pt,
        step_indices=step_indices,
    )
    # Sidecar stats.
    stats_path = out_path.replace('.npz', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(dict(
            n_random_seeds=int(n_random_seeds),
            k_per_seed=int(k_per_seed),
            n_decision_seeds=int(n_decision_seeds),
            n_samples=int(observations.shape[0]),
            n_rollouts=int(len(rollout_stats)),
            obs_dim=OBS_DIM, act_dim=ACT_DIM,
            dart_sigma=DART_SIGMA.tolist(),
            probe_offsets=PROBE_OFFSETS.tolist(),
            obs_layout="pos_err(3) + vel(3) + euler(3) + omega(3) + sdf(1) + sdf_grad(3) + 8*sdf_probe",
            rollouts=rollout_stats,
        ), f, indent=2)

    if verbose:
        n_dp = int(is_decision_pt.sum())
        print(f"\n[done] {observations.shape[0]:,} samples written to {out_path}")
        print(f"  decision-point samples: {n_dp:,}")
        print(f"  random samples:         {observations.shape[0] - n_dp:,}")
        print(f"  rollouts: {len(rollout_stats)}")
    return dict(observations=observations, actions=actions,
                seeds=seeds, variant_ids=variant_ids,
                is_decision_pt=is_decision_pt,
                step_indices=step_indices,
                stats_path=stats_path, out_path=out_path)


# ---- CLI -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=200,
                        help='number of random obstacle seeds')
    parser.add_argument('--k', type=int, default=3,
                        help='A* variants per random seed')
    parser.add_argument('--decision-seeds', type=int, default=30)
    parser.add_argument('--out', type=str, default='data/planner_dataset_v1.npz')
    parser.add_argument('--quick', action='store_true',
                        help='smoke-test: 5 random + 2 decision-point')
    parser.add_argument('--t-max', type=float, default=10.0)
    parser.add_argument('--workers', type=int, default=4,
                        help='parallel worker processes')
    parser.add_argument('--dp-only', action='store_true',
                        help='re-collect only the decision-point seeds; '
                             'random samples are copied from --v1')
    parser.add_argument('--v1', type=str,
                        default='data/planner_dataset_v1.npz',
                        help='source for the random samples when --dp-only')
    args = parser.parse_args()
    if args.quick:
        args.seeds = 5
        args.k = 3
        args.decision_seeds = 2
        args.out = 'data/planner_dataset_smoke.npz'
    print(f"[collect] {args.seeds} random seeds * k={args.k} + "
          f"{args.decision_seeds} decision-point seeds")
    run_collection(
        n_random_seeds=args.seeds,
        k_per_seed=args.k,
        n_decision_seeds=args.decision_seeds,
        out_path=args.out,
        T_max=args.t_max,
        n_workers=args.workers,
        verbose=True,
        dp_only=args.dp_only,
        v1_path=args.v1,
    )


if __name__ == '__main__':
    main()
