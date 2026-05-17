"""
audit_modes.py -- measure the multi-modality of the (obs, action) dataset.

Two questions matter for the diffusion-vs-MLP story:

  (1) Do many observations get multiple distinct actions in the dataset?
      A pure-MLP regressor averages multimodal targets and ends up
      committing to neither mode. Diffusion can represent both.

  (2) When the modes are present, are they actually separable -- i.e.,
      action *variance* within a small obs-neighbourhood is large?

We measure both: for each query obs, find its k=5 nearest neighbours in
obs-space (excluding self). Cluster size = neighbour count; cluster is
declared MULTIMODAL when the per-action max-min span exceeds a threshold,
unimodal otherwise.

Inputs are read from data/planner_dataset_v1.npz; outputs land in
data/multimodality_audit.json.

Usage:
    python distillation/audit_modes.py [--data data/...] [--k 5]
                                       [--radius 0.10] [--sample 5000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent


def standardize_obs(obs: np.ndarray) -> np.ndarray:
    """z-score across the dataset so the kNN distances are scale-fair."""
    mu = obs.mean(axis=0, keepdims=True)
    sd = obs.std(axis=0, keepdims=True) + 1e-6
    return (obs - mu) / sd


def audit(data_path: str,
          out_path: str | None = None,
          k: int = 5,
          radius: float | None = None,
          n_sample: int = 5000,
          action_span_thresh: float = 0.10,
          rng_seed: int = 0,
          ) -> dict:
    """Returns a dict of summary stats and writes it to JSON.

    Reports two flavours of multi-modality:
      * within-cluster:   kNN spans actions across whatever neighbours are
                          closest in obs-space (typically same-rollout).
      * cross-variant:    same idea, but neighbours are restricted to come
                          from a DIFFERENT variant_id (different planner
                          rollout). This is the metric that actually
                          tests the diffusion-vs-MLP story.
    """
    data = np.load(data_path)
    obs = data['observations']           # (N, 24)
    act = data['actions']                # (N, 4)
    is_dp = data['is_decision_pt']       # (N,)
    seeds = data['seeds']
    variant_ids = data['variant_ids']    # (N,)
    print(f"[audit] {obs.shape[0]:,} samples loaded from {data_path}")

    # Standardise obs for distance fairness.
    obs_z = standardize_obs(obs).astype(np.float32)

    # Sample query points (uniformly + extra from decision-point seeds).
    rng = np.random.default_rng(rng_seed)
    n_total = obs.shape[0]
    n_query = min(n_sample, n_total)
    # 50/50 split between random and decision-point, when available
    idx_dp = np.flatnonzero(is_dp)
    idx_rand = np.flatnonzero(~is_dp)
    n_q_dp = min(n_query // 2, len(idx_dp))
    n_q_rand = n_query - n_q_dp
    q_dp = rng.choice(idx_dp, size=n_q_dp, replace=False) if n_q_dp > 0 else np.array([], dtype=int)
    q_rand = rng.choice(idx_rand, size=n_q_rand, replace=False) if n_q_rand > 0 else np.array([], dtype=int)
    q_idx = np.concatenate([q_dp, q_rand])

    # Build NN over the FULL dataset (so query points see all neighbours).
    tree = cKDTree(obs_z)

    if radius is None:
        sample_idx = rng.choice(n_total, size=min(2000, n_total),
                                replace=False)
        d1, _ = tree.query(obs_z[sample_idx], k=2)
        radius = float(np.percentile(d1[:, 1], 60))
        print(f"[audit] auto radius={radius:.3f} (60th pct of 1-NN dist)")

    print(f"[audit] querying {len(q_idx):,} clusters (k={k}, "
          f"radius_diag<={radius:.3f})")

    cluster_stats = []
    n_multimodal = 0
    n_unimodal = 0
    n_within_radius = 0
    action_spans = []
    cluster_radii = []

    # Within-cluster (vanilla) kNN
    dists_all, nbrs_all = tree.query(obs_z[q_idx], k=k+1)

    # For cross-variant analysis we'll need a much larger k so we can
    # filter to neighbours from a different variant_id.
    k_big = min(200, n_total - 1)
    dists_big, nbrs_big = tree.query(obs_z[q_idx], k=k_big)
    cross_spans = []
    cross_dists = []
    cross_spans_dp = []

    for q_pos, qi in enumerate(q_idx):
        dist = dists_all[q_pos]; nbr = nbrs_all[q_pos]
        keep = nbr != qi
        dist = dist[keep]; nbr = nbr[keep]
        cluster = np.concatenate([[qi], nbr])
        cluster_actions = act[cluster]
        span = float((cluster_actions.max(axis=0) -
                      cluster_actions.min(axis=0)).max())
        action_spans.append(span)
        cluster_radii.append(float(dist.max()))
        multimodal = span > action_span_thresh
        if multimodal:
            n_multimodal += 1
        else:
            n_unimodal += 1
        if dist.max() <= radius:
            n_within_radius += 1
        cluster_stats.append(dict(
            size=int(len(cluster)),
            action_span=span,
            max_dist=float(dist.max()),
            multimodal=bool(multimodal),
            is_dp=bool(is_dp[qi]),
        ))

        # Cross-variant: find first neighbour with a different variant_id
        q_var = variant_ids[qi]
        big_nbrs = nbrs_big[q_pos]
        big_d = dists_big[q_pos]
        diff_mask = variant_ids[big_nbrs] != q_var
        if diff_mask.sum() < 1:
            continue
        cross_nbrs = big_nbrs[diff_mask][:k]
        cross_d = big_d[diff_mask][:k]
        cross_cluster_acts = act[np.concatenate([[qi], cross_nbrs])]
        cspan = float((cross_cluster_acts.max(axis=0) -
                       cross_cluster_acts.min(axis=0)).max())
        cross_spans.append(cspan)
        cross_dists.append(float(cross_d.max()))
        if is_dp[qi]:
            cross_spans_dp.append(cspan)
    n_singleton = 0

    total_classed = n_multimodal + n_unimodal
    frac_mm = (n_multimodal / total_classed) if total_classed > 0 else 0.0
    frac_singleton = 0.0
    frac_within_radius = n_within_radius / max(len(q_idx), 1)

    # Per-subset breakdown
    dp_classified = [s for s in cluster_stats if s.get('multimodal') is not None and s['is_dp']]
    rd_classified = [s for s in cluster_stats if s.get('multimodal') is not None and not s['is_dp']]
    frac_mm_dp = (sum(1 for s in dp_classified if s['multimodal']) /
                  max(len(dp_classified), 1))
    frac_mm_rd = (sum(1 for s in rd_classified if s['multimodal']) /
                  max(len(rd_classified), 1))

    # Multi-threshold sweep for cross-variant clusters.
    thresh_grid = [0.05, 0.08, 0.10, 0.15, 0.20]
    cross_arr = np.asarray(cross_spans)
    cross_arr_dp = np.asarray(cross_spans_dp)
    cross_summary = {}
    for t in thresh_grid:
        cross_summary[f'frac_mm@{t:.2f}'] = float((cross_arr > t).mean()) if len(cross_arr) else 0.0
        cross_summary[f'frac_mm_dp@{t:.2f}'] = float((cross_arr_dp > t).mean()) if len(cross_arr_dp) else 0.0

    report = dict(
        data_path=str(data_path),
        n_samples=int(obs.shape[0]),
        n_query=int(len(q_idx)),
        k=k,
        radius_diagnostic=float(radius),
        action_span_thresh=float(action_span_thresh),
        n_multimodal_clusters=int(n_multimodal),
        n_unimodal_clusters=int(n_unimodal),
        n_singleton_clusters=int(n_singleton),
        frac_multimodal=float(frac_mm),
        frac_multimodal_decision_pt=float(frac_mm_dp),
        frac_multimodal_random=float(frac_mm_rd),
        frac_within_radius=float(frac_within_radius),
        frac_singleton=float(frac_singleton),
        action_span_stats=dict(
            mean=float(np.mean(action_spans)) if action_spans else 0.0,
            median=float(np.median(action_spans)) if action_spans else 0.0,
            p95=float(np.percentile(action_spans, 95)) if action_spans else 0.0,
            max=float(np.max(action_spans)) if action_spans else 0.0,
        ),
        cluster_radius_stats=dict(
            mean=float(np.mean(cluster_radii)) if cluster_radii else 0.0,
            median=float(np.median(cluster_radii)) if cluster_radii else 0.0,
        ),
        cross_variant=dict(
            n_classified=int(len(cross_arr)),
            n_classified_dp=int(len(cross_arr_dp)),
            span_stats=dict(
                median=float(np.median(cross_arr)) if len(cross_arr) else 0.0,
                p75=float(np.percentile(cross_arr, 75)) if len(cross_arr) else 0.0,
                p95=float(np.percentile(cross_arr, 95)) if len(cross_arr) else 0.0,
                max=float(cross_arr.max()) if len(cross_arr) else 0.0,
            ),
            span_stats_dp=dict(
                median=float(np.median(cross_arr_dp)) if len(cross_arr_dp) else 0.0,
                p75=float(np.percentile(cross_arr_dp, 75)) if len(cross_arr_dp) else 0.0,
                p95=float(np.percentile(cross_arr_dp, 95)) if len(cross_arr_dp) else 0.0,
                max=float(cross_arr_dp.max()) if len(cross_arr_dp) else 0.0,
            ),
            distance_stats=dict(
                median=float(np.median(cross_dists)) if cross_dists else 0.0,
                p75=float(np.percentile(cross_dists, 75)) if cross_dists else 0.0,
            ),
            sweep=cross_summary,
        ),
    )

    print("\n=== Multi-modality audit ===")
    print(f"  within-cluster k={k} (same- and cross-variant neighbours)")
    print(f"    total clusters             : {total_classed:,}")
    print(f"    span median / p95          : "
          f"{report['action_span_stats']['median']:.3f} / "
          f"{report['action_span_stats']['p95']:.3f}")
    print(f"    fraction multimodal (>{action_span_thresh:.2f}): {frac_mm:.3f}")
    print(f"      (decision-point subset)  : {frac_mm_dp:.3f}")
    print(f"      (random-seed subset)     : {frac_mm_rd:.3f}")
    print()
    print(f"  cross-variant k={k} (neighbours forced to different rollouts)")
    cv = report['cross_variant']
    print(f"    classified : {cv['n_classified']:,} "
          f"(dp subset: {cv['n_classified_dp']:,})")
    print(f"    span median / p75 / p95    : "
          f"{cv['span_stats']['median']:.3f} / {cv['span_stats']['p75']:.3f}"
          f" / {cv['span_stats']['p95']:.3f}")
    print(f"    span median / p75 (dp)     : "
          f"{cv['span_stats_dp']['median']:.3f} / {cv['span_stats_dp']['p75']:.3f}")
    print(f"    nearest cross-variant dist : "
          f"{cv['distance_stats']['median']:.3f} (median)")
    for t in [0.05, 0.08, 0.10, 0.15, 0.20]:
        print(f"    multimodal at thresh {t:.2f} : "
              f"all={cv['sweep'][f'frac_mm@{t:.2f}']:.3f}, "
              f"dp={cv['sweep'][f'frac_mm_dp@{t:.2f}']:.3f}")

    if out_path is None:
        out_path = str(_HERE.parent / 'data' / 'multimodality_audit.json')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  -> {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='data/planner_dataset_v1.npz')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--radius', type=float, default=None,
                        help='diagnostic radius in z-scored obs units; '
                             'omit to auto-pick 60th-pct 1-NN distance')
    parser.add_argument('--sample', type=int, default=5000)
    parser.add_argument('--span-thresh', type=float, default=0.20,
                        help='max action-span (per dim) that counts as multimodal')
    args = parser.parse_args()

    data_path = args.data
    if not Path(data_path).is_absolute():
        data_path = str(_HERE.parent / data_path)
    audit(data_path, out_path=args.out, k=args.k, radius=args.radius,
          n_sample=args.sample, action_span_thresh=args.span_thresh)


if __name__ == '__main__':
    main()
