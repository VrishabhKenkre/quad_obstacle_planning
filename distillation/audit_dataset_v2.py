"""
audit_dataset_v2.py -- deeper cross-seed multi-modality audit for dataset v2.

The original audit_modes.py reports a single global k-NN multi-modality
fraction (45 %). This script breaks the audit down two ways that are
informative for the diffusion-vs-MLP story:

  (1) Per-obstacle-layout (per-seed) multi-modality fraction. For each
      unique seed, build a per-seed k-NN over the standardized 24-D
      observation, and report the fraction of k=10 neighbourhoods whose
      action span exceeds 0.20 (the nominal action range of the
      diffusion student).

  (2) Action-mode-count distribution across the whole dataset. For each
      of N sampled clusters (k=30 NN), fit GMMs with n in {1, 2, 3, 4}
      components on the cluster's 4-D actions and pick the BIC-optimal
      n. Report how many clusters land at each n.

Outputs:
  results/dataset_audit_v2.json
  results/multimodality_per_obstacle.png
  results/action_mode_distribution.png

Usage:
  python distillation/audit_dataset_v2.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


# ---- Hand-rolled GMM (sklearn isn't in requirements_distillation.txt) ----

def _gauss_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """log N(X; mean, cov), full covariance, shape (N,)."""
    d = X.shape[1]
    diff = X - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(logdet):
        return np.full(X.shape[0], -np.inf)
    try:
        sol = np.linalg.solve(cov, diff.T).T
    except np.linalg.LinAlgError:
        return np.full(X.shape[0], -np.inf)
    quad = np.einsum('ij,ij->i', diff, sol)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def _gmm_fit_info_criteria(X: np.ndarray, n_components: int,
                           reg_covar: float = 1e-3, max_iter: int = 60,
                           n_init: int = 3, seed: int = 0,
                           covariance_type: str = 'diag') -> tuple:
    """Fit a GMM via EM (`n_init` random restarts) and return (BIC, AIC).
    `covariance_type` in {'full', 'diag'}. Returns (+inf, +inf) on failure.
    Diagonal covariance has many fewer parameters per component and so is
    much more willing to add modes given a small cluster -- the right
    setting for this audit.
    """
    n, d = X.shape
    if n_components > n:
        return float('inf'), float('inf')

    def _diag_logpdf(X, mean, diag_cov):
        diff = X - mean
        return -0.5 * (
            d * np.log(2 * np.pi)
            + np.log(diag_cov + 1e-30).sum()
            + (diff * diff / (diag_cov + 1e-30)).sum(axis=1))

    if n_components == 1:
        mean = X.mean(axis=0)
        if covariance_type == 'diag':
            cov = X.var(axis=0) + reg_covar
            ll = _diag_logpdf(X, mean, cov).sum()
            n_params = d + d
        else:
            cov = np.cov(X.T) + reg_covar * np.eye(d)
            ll = _gauss_logpdf(X, mean, cov).sum()
            n_params = d + d * (d + 1) // 2
        bic = -2 * ll + n_params * np.log(n)
        aic = -2 * ll + 2 * n_params
        return float(bic), float(aic)

    rng = np.random.default_rng(seed)
    best_ll = -float('inf')
    for init in range(n_init):
        idx = rng.choice(n, size=n_components, replace=False)
        means = X[idx].copy()
        if covariance_type == 'diag':
            cov_init = X.var(axis=0) + reg_covar
            covs = np.tile(cov_init, (n_components, 1))
        else:
            cov_init = np.cov(X.T) + reg_covar * np.eye(d)
            covs = np.tile(cov_init, (n_components, 1, 1))
        weights = np.full(n_components, 1.0 / n_components)
        prev_ll = -np.inf
        ll = -np.inf

        for _ in range(max_iter):
            log_resp = np.full((n, n_components), -np.inf)
            for kk in range(n_components):
                if covariance_type == 'diag':
                    log_resp[:, kk] = np.log(weights[kk] + 1e-30) \
                        + _diag_logpdf(X, means[kk], covs[kk])
                else:
                    log_resp[:, kk] = np.log(weights[kk] + 1e-30) \
                        + _gauss_logpdf(X, means[kk], covs[kk])
            row_max = log_resp.max(axis=1, keepdims=True)
            log_norm = row_max + np.log(
                np.exp(log_resp - row_max).sum(axis=1, keepdims=True) + 1e-30)
            ll = float(log_norm.sum())
            if not np.isfinite(ll):
                break
            log_resp = log_resp - log_norm
            resp = np.exp(log_resp)
            Nk = resp.sum(axis=0) + 1e-12
            weights = Nk / n
            means = (resp.T @ X) / Nk[:, None]
            if covariance_type == 'diag':
                for kk in range(n_components):
                    diff = X - means[kk]
                    covs[kk] = (resp[:, kk:kk+1] * diff * diff).sum(axis=0) / Nk[kk]
                    covs[kk] = covs[kk] + reg_covar
            else:
                for kk in range(n_components):
                    diff = X - means[kk]
                    covs[kk] = (resp[:, kk:kk+1] * diff).T @ diff / Nk[kk]
                    covs[kk] = covs[kk] + reg_covar * np.eye(d)
            if abs(ll - prev_ll) < 1e-5 * max(abs(prev_ll), 1.0):
                break
            prev_ll = ll
        if np.isfinite(ll) and ll > best_ll:
            best_ll = ll
    if not np.isfinite(best_ll):
        return float('inf'), float('inf')

    if covariance_type == 'diag':
        n_params = (n_components - 1) + n_components * d + n_components * d
    else:
        n_params = (n_components - 1) + n_components * d + \
                   n_components * (d * (d + 1) // 2)
    bic = -2 * best_ll + n_params * np.log(n)
    aic = -2 * best_ll + 2 * n_params
    return float(bic), float(aic)


def standardize_obs(obs: np.ndarray) -> np.ndarray:
    mu = obs.mean(axis=0, keepdims=True)
    sd = obs.std(axis=0, keepdims=True) + 1e-6
    return ((obs - mu) / sd).astype(np.float32)


def per_seed_audit(obs_z: np.ndarray, act: np.ndarray, seeds: np.ndarray,
                   variant_ids: np.ndarray,
                   is_dp: np.ndarray, k: int = 10,
                   span_thresh: float = 0.20,
                   max_query_per_seed: int = 300,
                   radius: float = 0.10,
                   rng_seed: int = 0) -> dict:
    """For each unique seed, build a per-seed kNN cluster as
        query + k/V nearest neighbours drawn from EACH variant in the seed
    so cross-variant action diversity is exposed (a vanilla kNN cluster
    inside a single rollout sees almost-identical actions and undercounts
    multi-modality). Returns the fraction of clusters whose max
    pairwise L2 action distance exceeds `span_thresh`.
    """
    rng = np.random.default_rng(rng_seed)
    unique_seeds = np.unique(seeds)
    per_seed = {}
    for s in unique_seeds:
        s_mask = seeds == s
        s_idx = np.flatnonzero(s_mask)
        if len(s_idx) < k + 1:
            continue
        obs_s = obs_z[s_idx]
        act_s = act[s_idx]
        var_s = variant_ids[s_idx]
        is_dp_s = bool(is_dp[s_idx[0]])
        var_unique = np.unique(var_s)
        V = max(1, len(var_unique))

        # Per-variant cKDTree so we can request k/V neighbours from each
        # variant independently.
        per_var_trees = {}
        per_var_idx = {}
        for v in var_unique:
            v_local = np.flatnonzero(var_s == v)
            per_var_idx[int(v)] = v_local
            per_var_trees[int(v)] = cKDTree(obs_s[v_local])

        n_query = min(max_query_per_seed, len(s_idx))
        q_idx_local = rng.choice(len(s_idx), size=n_query, replace=False)
        k_per_var = max(2, int(np.ceil(k / V)))

        spans = []
        within_radius = 0
        for i in range(n_query):
            qi = q_idx_local[i]
            cluster_local = [qi]
            max_d = 0.0
            for v in var_unique:
                tree_v = per_var_trees[int(v)]
                v_local = per_var_idx[int(v)]
                kk = min(k_per_var, len(v_local))
                if kk <= 0:
                    continue
                d, nb = tree_v.query(obs_s[qi:qi+1], k=kk)
                d = np.atleast_1d(d).ravel()
                nb = np.atleast_1d(nb).ravel()
                cluster_local.extend(v_local[nb].tolist())
                if d.size:
                    max_d = max(max_d, float(d.max()))
            cluster_local = list(dict.fromkeys(cluster_local))  # dedupe
            ca = act_s[cluster_local]
            diffs = ca[:, None, :] - ca[None, :, :]
            pair_d = np.linalg.norm(diffs, axis=-1)
            spans.append(float(pair_d.max()))
            if max_d <= radius:
                within_radius += 1
        spans = np.asarray(spans)
        n_mm = int((spans > span_thresh).sum())
        sweep = {f'frac_mm@{t:.2f}': float((spans > t).mean())
                 for t in (0.05, 0.10, 0.15, 0.20, 0.30)}
        per_seed[int(s)] = dict(
            n_samples=int(len(s_idx)),
            n_clusters=int(n_query),
            n_variants=int(V),
            n_multimodal=int(n_mm),
            fraction_multimodal=float(n_mm / max(n_query, 1)),
            median_span=float(np.median(spans)),
            p95_span=float(np.percentile(spans, 95)),
            within_radius_frac=float(within_radius / max(n_query, 1)),
            is_decision_point_seed=is_dp_s,
            sweep=sweep,
        )
    return per_seed


def dataset_gmm_audit(obs_z: np.ndarray, act: np.ndarray,
                      seeds: np.ndarray, variant_ids: np.ndarray,
                      is_dp: np.ndarray,
                      n_sample: int = 2000, k: int = 30,
                      bic_n_components: tuple = (1, 2, 3, 4),
                      rng_seed: int = 1) -> dict:
    """Sample N clusters across the whole dataset and pick the BIC-optimal
    GMM component count on each cluster's 4-D action distribution.

    Clusters are built per-query by drawing k/V neighbours from EACH
    variant within the query's seed, exactly as in `per_seed_audit`. This
    is essential -- a vanilla global kNN cluster lives inside a single
    rollout and contains only one mode by construction, so the BIC
    almost always selects n=1.

    Returns histograms of n_components and the raw per-cluster intra
    -cluster action variances.
    """
    rng = np.random.default_rng(rng_seed)
    idx_dp = np.flatnonzero(is_dp)
    idx_rand = np.flatnonzero(~is_dp)
    n_q_dp = min(n_sample // 2, len(idx_dp))
    n_q_rand = n_sample - n_q_dp
    q_dp = rng.choice(idx_dp, size=n_q_dp, replace=False) if n_q_dp > 0 else np.array([], dtype=int)
    q_rand = rng.choice(idx_rand, size=n_q_rand, replace=False) if n_q_rand > 0 else np.array([], dtype=int)
    q_idx = np.concatenate([q_dp, q_rand])

    # Pre-index seed -> indices, and within-seed variant -> indices
    unique_seeds = np.unique(seeds)
    seed_idx = {int(s): np.flatnonzero(seeds == s) for s in unique_seeds}
    seed_var_idx = {}
    seed_var_trees = {}
    for s in unique_seeds:
        s_i = seed_idx[int(s)]
        var_here = variant_ids[s_i]
        u_v = np.unique(var_here)
        per_var = {}
        per_tree = {}
        for v in u_v:
            v_local = s_i[var_here == v]
            per_var[int(v)] = v_local
            per_tree[int(v)] = cKDTree(obs_z[v_local])
        seed_var_idx[int(s)] = per_var
        seed_var_trees[int(s)] = per_tree

    n_comp_counts_bic = {int(n): 0 for n in bic_n_components}
    n_comp_counts_aic = {int(n): 0 for n in bic_n_components}
    intra_variances = []
    cluster_max_radii = []
    per_n_is_dp_bic = {int(n): [] for n in bic_n_components}

    warnings.filterwarnings('ignore')
    for i in range(len(q_idx)):
        qi = int(q_idx[i])
        s = int(seeds[qi])
        per_var = seed_var_idx[s]
        per_tree = seed_var_trees[s]
        V = len(per_var)
        k_per_var = max(2, int(np.ceil(k / V)))
        cluster_global = [qi]
        max_d = 0.0
        for v, v_idx_global in per_var.items():
            kk = min(k_per_var, len(v_idx_global))
            if kk <= 0:
                continue
            d, nb = per_tree[v].query(obs_z[qi:qi+1], k=kk)
            d = np.atleast_1d(d).ravel()
            nb = np.atleast_1d(nb).ravel()
            cluster_global.extend(v_idx_global[nb].tolist())
            if d.size:
                max_d = max(max_d, float(d.max()))
        cluster_global = list(dict.fromkeys(cluster_global))
        ca = act[cluster_global]
        intra_variances.append(float(ca.var(axis=0).sum()))
        cluster_max_radii.append(max_d)
        bics, aics = [], []
        for n in bic_n_components:
            b, a = _gmm_fit_info_criteria(
                ca, n_components=n, reg_covar=1e-3, max_iter=60,
                n_init=3, seed=rng_seed + i, covariance_type='diag')
            bics.append(b)
            aics.append(a)
        best_n_bic = int(bic_n_components[int(np.argmin(bics))])
        best_n_aic = int(bic_n_components[int(np.argmin(aics))])
        n_comp_counts_bic[best_n_bic] += 1
        n_comp_counts_aic[best_n_aic] += 1
        per_n_is_dp_bic[best_n_bic].append(bool(is_dp[qi]))

    return dict(
        n_sample=int(len(q_idx)),
        k=int(k),
        covariance_type='diag',
        n_components_counts={str(k_): int(v) for k_, v in n_comp_counts_bic.items()},
        n_components_counts_aic={str(k_): int(v) for k_, v in n_comp_counts_aic.items()},
        n_components_dp_fraction={
            str(n): (float(np.mean(per_n_is_dp_bic[n]))
                     if per_n_is_dp_bic[n] else 0.0)
            for n in bic_n_components
        },
        intra_variance_stats=dict(
            mean=float(np.mean(intra_variances)),
            median=float(np.median(intra_variances)),
            p95=float(np.percentile(intra_variances, 95)),
        ),
        cluster_radius_stats=dict(
            median=float(np.median(cluster_max_radii)),
            p95=float(np.percentile(cluster_max_radii, 95)),
        ),
        _intra_variances=intra_variances,  # raw list for the histogram
    )


def plot_per_seed(per_seed: dict, out_path: Path):
    """Bar chart: per-seed % multimodal clusters, sorted, with separate
    bar groups for random vs decision-point seeds."""
    items = list(per_seed.items())
    rand = [(s, v) for s, v in items if not v['is_decision_point_seed']]
    dp = [(s, v) for s, v in items if v['is_decision_point_seed']]
    rand.sort(key=lambda x: x[1]['fraction_multimodal'])
    dp.sort(key=lambda x: x[1]['fraction_multimodal'])
    rand_fracs = [v['fraction_multimodal'] for _, v in rand]
    dp_fracs = [v['fraction_multimodal'] for _, v in dp]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    x_rand = np.arange(len(rand_fracs))
    x_dp = np.arange(len(rand_fracs), len(rand_fracs) + len(dp_fracs))
    ax.bar(x_rand, rand_fracs, color='#4c78a8', alpha=0.85,
           label=f'random seeds (n={len(rand_fracs)})')
    ax.bar(x_dp, dp_fracs, color='#e45756', alpha=0.9,
           label=f'decision-point seeds (n={len(dp_fracs)})')
    if rand_fracs:
        med_rand = float(np.median(rand_fracs))
        ax.axhline(med_rand, color='#4c78a8', linestyle='--', lw=1,
                   alpha=0.6, label=f'random median = {med_rand:.2f}')
    if dp_fracs:
        med_dp = float(np.median(dp_fracs))
        ax.axhline(med_dp, color='#e45756', linestyle='--', lw=1,
                   alpha=0.6, label=f'decision-point median = {med_dp:.2f}')
    ax.set_xlabel('seed (sorted within each group)')
    ax.set_ylabel('fraction of k=10 clusters with action span > 0.20')
    ax.set_title('Per-seed multi-modality fraction in planner_dataset_v2.npz')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_gmm_distribution(gmm_audit: dict, out_path: Path):
    """Two-panel figure: (left) GMM n_components distribution bar chart
    side-by-side for BIC and AIC, (right) intra-cluster action variance
    histogram on a log scale."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)
    counts_bic = gmm_audit['n_components_counts']
    counts_aic = gmm_audit['n_components_counts_aic']
    keys = sorted(counts_bic.keys(), key=int)
    vals_bic = [counts_bic[k] for k in keys]
    vals_aic = [counts_aic.get(k, 0) for k in keys]
    total = sum(vals_bic)
    mm_bic = sum(counts_bic[k] for k in keys if int(k) >= 2)
    mm_aic = sum(counts_aic.get(k, 0) for k in keys if int(k) >= 2)
    k_used = gmm_audit.get('k', 0)

    x = np.arange(len(keys))
    width = 0.4
    b1 = axes[0].bar(x - width/2, vals_bic, width=width,
                     color='#4c78a8', label='BIC')
    b2 = axes[0].bar(x + width/2, vals_aic, width=width,
                     color='#f58518', label='AIC')
    for bars, vals in [(b1, vals_bic), (b2, vals_aic)]:
        for b, v in zip(bars, vals):
            if v > 0:
                axes[0].text(b.get_x() + b.get_width() / 2,
                             b.get_height() + total*0.005,
                             f'{v}', ha='center', va='bottom', fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(keys)
    axes[0].set_xlabel(f'selected n_components per cluster (k={k_used})')
    axes[0].set_ylabel('number of clusters')
    axes[0].set_title(
        f'GMM mode-count distribution (diag-cov)\n'
        f'multimodal (n$\\geq$2): BIC {100*mm_bic/max(total,1):.0f}%, '
        f'AIC {100*mm_aic/max(total,1):.0f}%')
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].grid(True, alpha=0.25, axis='y')
    axes[0].set_ylim(0, max(vals_bic) * 1.18)

    var = np.asarray(gmm_audit['_intra_variances'])
    var = var[var > 0]
    axes[1].hist(np.log10(var + 1e-9), bins=50, color='#888888',
                 edgecolor='black', alpha=0.85)
    axes[1].set_xlabel('log10(intra-cluster total action variance)')
    axes[1].set_ylabel('number of clusters')
    axes[1].set_title('Intra-cluster action variance (log scale)')
    axes[1].axvline(np.log10(0.04), color='red', linestyle='--', lw=1,
                    label='span $\\approx$ 0.20 region')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25, axis='y')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str,
                    default='data/planner_dataset_v2.npz')
    ap.add_argument('--out-json', type=str,
                    default='results/dataset_audit_v2.json')
    ap.add_argument('--plot-per-seed', type=str,
                    default='results/multimodality_per_obstacle.png')
    ap.add_argument('--plot-gmm', type=str,
                    default='results/action_mode_distribution.png')
    ap.add_argument('--k-per-seed', type=int, default=10)
    ap.add_argument('--k-gmm', type=int, default=50)
    ap.add_argument('--span-thresh', type=float, default=0.20)
    ap.add_argument('--max-query-per-seed', type=int, default=300)
    ap.add_argument('--n-sample-gmm', type=int, default=2000)
    args = ap.parse_args()

    data_path = _ROOT / args.data
    data = np.load(data_path)
    obs = data['observations']
    act = data['actions']
    seeds = data['seeds']
    variant_ids = data['variant_ids']
    is_dp = data['is_decision_pt']
    print(f"[audit-v2] loaded {obs.shape[0]:,} samples from {data_path.name}, "
          f"{len(np.unique(seeds))} unique seeds")

    obs_z = standardize_obs(obs)

    print(f"[audit-v2] per-seed cross-variant kNN audit (k={args.k_per_seed}, "
          f"span_thresh={args.span_thresh})")
    per_seed = per_seed_audit(
        obs_z, act, seeds, variant_ids, is_dp,
        k=args.k_per_seed, span_thresh=args.span_thresh,
        max_query_per_seed=args.max_query_per_seed)
    dp_fracs = [v['fraction_multimodal']
                for v in per_seed.values() if v['is_decision_point_seed']]
    rand_fracs = [v['fraction_multimodal']
                  for v in per_seed.values() if not v['is_decision_point_seed']]
    n_clusters_total = sum(v['n_clusters'] for v in per_seed.values())
    n_mm_total = sum(v['n_multimodal'] for v in per_seed.values())
    frac_mm_knn = n_mm_total / max(n_clusters_total, 1)
    print(f"  per-seed clusters: {n_clusters_total:,}; "
          f"multimodal (kNN, span>{args.span_thresh}): "
          f"{n_mm_total:,} ({100*frac_mm_knn:.1f}%)")
    print(f"  median dp seed fraction:      {np.median(dp_fracs):.3f}")
    print(f"  median random seed fraction:  {np.median(rand_fracs):.3f}")

    # Threshold sweep at the dataset level
    sweep_keys = ['frac_mm@0.05', 'frac_mm@0.10', 'frac_mm@0.15',
                  'frac_mm@0.20', 'frac_mm@0.30']
    sweep_all, sweep_dp, sweep_rand = {}, {}, {}
    for sk in sweep_keys:
        vals_all = [v['sweep'][sk] for v in per_seed.values()]
        vals_dp = [v['sweep'][sk] for v in per_seed.values()
                   if v['is_decision_point_seed']]
        vals_rd = [v['sweep'][sk] for v in per_seed.values()
                   if not v['is_decision_point_seed']]
        sweep_all[sk] = float(np.mean(vals_all)) if vals_all else 0.0
        sweep_dp[sk] = float(np.mean(vals_dp)) if vals_dp else 0.0
        sweep_rand[sk] = float(np.mean(vals_rd)) if vals_rd else 0.0
    print("  threshold sweep (per-seed mean fraction multimodal):")
    for sk in sweep_keys:
        print(f"    {sk}: all={sweep_all[sk]:.3f}  "
              f"dp={sweep_dp[sk]:.3f}  rand={sweep_rand[sk]:.3f}")

    print(f"[audit-v2] dataset-wide GMM audit (n_sample={args.n_sample_gmm}, "
          f"k={args.k_gmm}, BIC over n=1..4)")
    gmm = dataset_gmm_audit(
        obs_z, act, seeds, variant_ids, is_dp,
        n_sample=args.n_sample_gmm, k=args.k_gmm)
    cnts = gmm['n_components_counts']
    total = sum(cnts.values())
    n_bimodal = int(cnts.get('2', 0))
    n_3plus = sum(int(cnts.get(str(n), 0)) for n in (3, 4))
    n_mm_gmm = n_bimodal + n_3plus
    frac_mm_gmm = n_mm_gmm / max(total, 1)
    cnts_aic = gmm['n_components_counts_aic']
    n_mm_aic = sum(int(cnts_aic.get(str(n), 0)) for n in (2, 3, 4))
    frac_mm_aic = n_mm_aic / max(total, 1)
    print(f"  GMM mode distribution (BIC, diag-cov): "
          f"n=1 {cnts.get('1',0)}, n=2 {cnts.get('2',0)}, "
          f"n=3 {cnts.get('3',0)}, n=4 {cnts.get('4',0)}")
    print(f"  GMM mode distribution (AIC, diag-cov): "
          f"n=1 {cnts_aic.get('1',0)}, n=2 {cnts_aic.get('2',0)}, "
          f"n=3 {cnts_aic.get('3',0)}, n=4 {cnts_aic.get('4',0)}")
    print(f"  fraction multimodal (BIC GMM, n>=2): "
          f"{n_mm_gmm}/{total} = {100*frac_mm_gmm:.1f}%")
    print(f"  fraction multimodal (AIC GMM, n>=2): "
          f"{n_mm_aic}/{total} = {100*frac_mm_aic:.1f}%")

    headline = (f"{100*frac_mm_gmm:.0f}% of action clusters in dataset v2 are "
                f"multimodal under BIC-selected GMM (n>=2); "
                f"{100*sweep_all['frac_mm@0.10']:.0f}% of cross-variant k=10 "
                f"neighbourhoods have max pairwise action distance > 0.10.")
    print(f"\n  HEADLINE: {headline}")

    report = dict(
        dataset=str(args.data),
        n_samples=int(obs.shape[0]),
        n_clusters_total=int(n_clusters_total),
        n_clusters_multimodal_kNN=int(n_mm_total),
        fraction_multimodal_kNN=float(frac_mm_knn),
        n_clusters_bimodal_gmm=int(n_bimodal),
        **{'n_clusters_3+_modal_gmm': int(n_3plus)},
        fraction_multimodal_gmm=float(frac_mm_gmm),
        fraction_multimodal_gmm_aic=float(frac_mm_aic),
        threshold_sweep_per_seed_mean=dict(
            all=sweep_all, dp=sweep_dp, random=sweep_rand,
        ),
        per_seed={str(s): v for s, v in per_seed.items()},
        gmm_audit={k: v for k, v in gmm.items() if k != '_intra_variances'},
        summary=dict(
            median_dp_seed_multimodality_fraction=float(np.median(dp_fracs))
                if dp_fracs else 0.0,
            median_random_seed_multimodality_fraction=float(np.median(rand_fracs))
                if rand_fracs else 0.0,
            headline_claim_text=headline,
        ),
    )
    out_json = _ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  -> {out_json}")

    plot_per_seed(per_seed, _ROOT / args.plot_per_seed)
    plot_gmm_distribution(gmm, _ROOT / args.plot_gmm)


if __name__ == '__main__':
    main()
