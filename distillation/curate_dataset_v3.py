"""
curate_dataset_v3.py -- filter planner_dataset_v2.npz down to a curated
v3 dataset by removing low-quality planner rollouts.

Pipeline:
  1. Reconstruct positions from observations[:, 0:3] + GOAL.
  2. Group samples by (seed, variant_id) -> one rollout per group.
  3. For each rollout compute the metrics listed in the brief:
       n_steps, initial_pos, final_pos, goal_pos, straight_line_distance,
       path_length, efficiency, final_goal_err_mm,
       max_field_along_rollout, mean_field_along_rollout,
       terminal_speed_m_s, max_z_along_rollout, overshoot_amount,
       ugliness_score.
  4. Apply primary filter; adaptively relax if <250 rollouts survive.
  5. Save v3 npz (sample-level filtered) + stats JSON + audit JSON.

Outputs:
  data/planner_dataset_v3.npz
  data/planner_dataset_v3_stats.json
  data/planner_dataset_v2_rollout_audit.json  (per-rollout metrics for v2)

Usage:
  python distillation/curate_dataset_v3.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / 'distillation'))
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))

from voxelize import VoxelMap
from obstacle_course import make_obstacles, obstacle_field_value
from collect_planner_data import START, GOAL
from randomize_astar import decision_point_layout


DT_CTRL = 0.02


# ---- Filter criteria -----------------------------------------------------

# Primary filter (per the brief).
PRIMARY = dict(
    final_goal_err_mm_max=30.0,
    max_field_max=0.10,
    terminal_speed_max=0.5,
    efficiency_min=0.65,
    overshoot_max=0.10,
)

# Ordered list of (criterion_key, new_value) relaxations.
RELAX_STEPS = [
    ('efficiency_min', 0.60),
    ('max_field_max', 0.12),
    ('terminal_speed_max', 0.7),
    ('overshoot_max', 0.15),
    ('efficiency_min', 0.55),
]


def ugliness_score(metrics: dict) -> float:
    return float(
        0.40 * (metrics['max_field_along_rollout'] / 0.10) +
        0.25 * (metrics['final_goal_err_mm'] / 30.0) +
        0.15 * (1.0 - metrics['efficiency']) / max(0.01, (1.0 - 0.65)) +
        0.10 * (metrics['terminal_speed_m_s'] / 0.5) +
        0.10 * (metrics['overshoot_amount'] / 0.10)
    )


def _passes(metrics: dict, crit: dict) -> bool:
    return (metrics['final_goal_err_mm'] < crit['final_goal_err_mm_max']
            and metrics['max_field_along_rollout'] < crit['max_field_max']
            and metrics['terminal_speed_m_s'] < crit['terminal_speed_max']
            and metrics['efficiency'] > crit['efficiency_min']
            and metrics['overshoot_amount'] < crit['overshoot_max'])


def per_criterion_pass_rate(rollouts: list, crit: dict) -> dict:
    n = max(len(rollouts), 1)
    return dict(
        final_goal_err_lt_30=sum(
            1 for r in rollouts
            if r['final_goal_err_mm'] < crit['final_goal_err_mm_max']) / n,
        max_field_lt=sum(
            1 for r in rollouts
            if r['max_field_along_rollout'] < crit['max_field_max']) / n,
        terminal_speed_lt=sum(
            1 for r in rollouts
            if r['terminal_speed_m_s'] < crit['terminal_speed_max']) / n,
        efficiency_gt=sum(
            1 for r in rollouts
            if r['efficiency'] > crit['efficiency_min']) / n,
        overshoot_lt=sum(
            1 for r in rollouts
            if r['overshoot_amount'] < crit['overshoot_max']) / n,
    )


def _build_obstacles_for(seed: int, is_dp: bool):
    """Reconstruct obstacles for a rollout. Caches by (seed, is_dp)."""
    key = (int(seed), bool(is_dp))
    if key not in _build_obstacles_for._cache:
        if is_dp:
            obstacles, _, _ = decision_point_layout(seed=int(seed))
        else:
            obstacles = make_obstacles(seed=int(seed))
        _build_obstacles_for._cache[key] = obstacles
    return _build_obstacles_for._cache[key]
_build_obstacles_for._cache = {}


def compute_rollout_metrics(seed: int, variant_id: int, is_dp: bool,
                            obs: np.ndarray, step_indices: np.ndarray
                            ) -> dict:
    """Reconstruct positions from obs[:, 0:3] + GOAL and compute the brief's
    per-rollout metrics. obs is the slice for this rollout, sorted by
    step_indices ascending.
    """
    goal = np.asarray(GOAL, dtype=np.float64)
    positions = obs[:, 0:3].astype(np.float64) + goal[None, :]
    velocities = obs[:, 3:6].astype(np.float64)
    obstacles = _build_obstacles_for(seed, is_dp)

    n = positions.shape[0]
    p0 = positions[0]
    p_final = positions[-1]
    straight = float(np.linalg.norm(p0 - goal))
    seg = np.diff(positions, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    path_length = float(seg_len.sum())
    eff = float(straight / max(path_length, 1e-9))
    final_goal_err_m = float(np.linalg.norm(p_final - goal))

    fields = np.asarray([obstacle_field_value(p, obstacles)
                          for p in positions])
    max_field = float(fields.max())
    mean_field = float(fields.mean())

    # terminal speed: from the stored velocity vector at the last step
    terminal_speed = float(np.linalg.norm(velocities[-1]))

    max_z = float(positions[:, 2].max())

    # Overshoot: project (p_t - goal) onto (goal - initial) direction.
    # Positive projection = drone is past the goal along that axis.
    direction = goal - p0
    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-6:
        overshoot = 0.0
    else:
        unit_dir = direction / dir_norm
        # signed_distance_past_goal_t = (p_t - goal) . unit_dir
        projections = (positions - goal[None, :]) @ unit_dir
        overshoot = float(np.maximum(0.0, projections).max())

    metrics = dict(
        seed=int(seed),
        variant_id=int(variant_id),
        is_decision_pt=bool(is_dp),
        n_steps=int(n),
        initial_pos=[float(x) for x in p0],
        final_pos=[float(x) for x in p_final],
        goal_pos=[float(x) for x in goal],
        straight_line_distance=float(straight),
        path_length=float(path_length),
        efficiency=float(eff),
        final_goal_err_mm=float(final_goal_err_m * 1000.0),
        max_field_along_rollout=float(max_field),
        mean_field_along_rollout=float(mean_field),
        terminal_speed_m_s=float(terminal_speed),
        max_z_along_rollout=float(max_z),
        overshoot_amount=float(overshoot),
    )
    metrics['ugliness_score'] = float(ugliness_score(metrics))
    return metrics


def variant_label(seed: int, variant_id: int, is_dp: bool) -> str:
    """Descriptive label used in audit dumps."""
    if is_dp:
        # dp variants are 0 (left) / 1 (right) by collect convention
        return f"dp_{'left' if int(variant_id) == 0 else 'right'}"
    return f"random_K{int(variant_id)}"


def curate(data_path: Path, out_npz: Path, out_stats: Path,
           out_audit: Path, target_min: int = 250) -> dict:
    print(f"[curate] loading {data_path}")
    data = np.load(data_path)
    obs = data['observations']
    actions = data['actions']
    seeds = data['seeds']
    variant_ids = data['variant_ids']
    is_dp = data['is_decision_pt']
    step_indices = data['step_indices']
    n_samples = obs.shape[0]
    print(f"[curate] {n_samples:,} samples")

    # Group by (seed, variant_id)
    keys = np.stack([seeds, variant_ids], axis=1).astype(np.int64)
    keys_combined = (keys[:, 0].astype(np.int64) * 100_000
                     + keys[:, 1].astype(np.int64))
    unique_keys, inverse, counts = np.unique(
        keys_combined, return_inverse=True, return_counts=True)
    n_rollouts = len(unique_keys)
    print(f"[curate] {n_rollouts} rollouts")

    rollouts = []
    for ri in range(n_rollouts):
        mask = (inverse == ri)
        idx = np.flatnonzero(mask)
        # Sort by step_indices to be safe
        order = np.argsort(step_indices[idx])
        idx_sorted = idx[order]
        obs_r = obs[idx_sorted]
        seed = int(seeds[idx_sorted[0]])
        variant = int(variant_ids[idx_sorted[0]])
        dp = bool(is_dp[idx_sorted[0]])
        m = compute_rollout_metrics(seed, variant, dp,
                                    obs_r, step_indices[idx_sorted])
        m['variant'] = variant_label(seed, variant, dp)
        # Stash the global indices so we can rebuild the npz later
        m['_sample_indices'] = idx_sorted.tolist()
        rollouts.append(m)
        if (ri + 1) % 100 == 0 or ri == n_rollouts - 1:
            print(f"  ... {ri+1}/{n_rollouts} rollouts metric'd")

    # Sort by ugliness (descending) for the audit file
    rollouts_sorted = sorted(rollouts, key=lambda r: -r['ugliness_score'])

    out_audit.parent.mkdir(parents=True, exist_ok=True)
    audit_dump = []
    for r in rollouts_sorted:
        d = {k: v for k, v in r.items() if k != '_sample_indices'}
        audit_dump.append(d)
    json.dump(dict(
        dataset=str(data_path),
        n_rollouts=int(n_rollouts),
        rollouts=audit_dump,
    ), open(out_audit, 'w'), indent=2)
    print(f"[curate] wrote audit -> {out_audit}")

    # Primary filter
    crit = dict(PRIMARY)
    surviving = [r for r in rollouts if _passes(r, crit)]
    print(f"\n[curate] primary criteria: {crit}")
    print(f"[curate] surviving with primary: {len(surviving)} "
          f"/ {n_rollouts} ({100*len(surviving)/n_rollouts:.1f}%)")

    relaxations = []
    if len(surviving) < target_min:
        for (k, v) in RELAX_STEPS:
            old_v = crit[k]
            crit[k] = v
            relaxations.append({'criterion': k, 'old': old_v, 'new': v})
            surviving = [r for r in rollouts if _passes(r, crit)]
            print(f"[curate] relaxed {k}: {old_v} -> {v}; "
                  f"survivors {len(surviving)} / {n_rollouts}")
            if len(surviving) >= target_min:
                break
    print(f"\n[curate] FINAL criteria: {crit}")
    print(f"[curate] final survivors: {len(surviving)} / {n_rollouts}")

    pcr = per_criterion_pass_rate(rollouts, crit)
    print(f"[curate] per-criterion pass rate:")
    for k, v in pcr.items():
        print(f"    {k}: {v:.3f}")

    # Build v3 npz by concatenating sample indices of surviving rollouts.
    survivor_idx = np.concatenate([np.asarray(r['_sample_indices'])
                                    for r in surviving]) \
                    if surviving else np.array([], dtype=np.int64)
    survivor_idx.sort()
    print(f"[curate] v3 sample count = {survivor_idx.size:,} "
          f"({100*survivor_idx.size/n_samples:.1f}% of v2)")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        observations=obs[survivor_idx],
        actions=actions[survivor_idx],
        seeds=seeds[survivor_idx],
        variant_ids=variant_ids[survivor_idx],
        is_decision_pt=is_dp[survivor_idx],
        step_indices=step_indices[survivor_idx],
    )
    print(f"[curate] wrote {out_npz} ({out_npz.stat().st_size/1024**2:.1f} MB)")

    stats = dict(
        source_dataset=str(data_path),
        n_rollouts_source=int(n_rollouts),
        n_rollouts_surviving=int(len(surviving)),
        n_samples_source=int(n_samples),
        n_samples_surviving=int(survivor_idx.size),
        criteria_used=crit,
        relaxations_applied=relaxations,
        per_criterion_pass_rate_under_final_criteria=pcr,
    )
    json.dump(stats, open(out_stats, 'w'), indent=2)
    print(f"[curate] wrote {out_stats}")

    return dict(
        criteria_used=crit,
        relaxations_applied=relaxations,
        rollouts=rollouts,
        surviving=surviving,
        rollouts_sorted_by_ugliness=rollouts_sorted,
        per_criterion_pass_rate=pcr,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str,
                    default='data/planner_dataset_v2.npz')
    ap.add_argument('--out-npz', type=str,
                    default='data/planner_dataset_v3.npz')
    ap.add_argument('--out-stats', type=str,
                    default='data/planner_dataset_v3_stats.json')
    ap.add_argument('--out-audit', type=str,
                    default='data/planner_dataset_v2_rollout_audit.json')
    ap.add_argument('--target-min', type=int, default=250)
    args = ap.parse_args()
    curate(_ROOT / args.data, _ROOT / args.out_npz,
           _ROOT / args.out_stats, _ROOT / args.out_audit,
           target_min=int(args.target_min))


if __name__ == '__main__':
    main()
