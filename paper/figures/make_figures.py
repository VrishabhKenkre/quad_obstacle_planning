#!/usr/bin/env python
"""
paper/figures/make_figures.py -- the six figures for the IEEE paper
"Diffusion Policies Preserve Multi-modal Plans". Re-rendered from the eval
artifacts already in results/; no GPU and no experiments needed.

Figures:
  fig1_teaser.png           -- MLP into obstacle vs diffusion around it
  fig2_pipeline.png         -- planner block diagram (A* / min-snap / MPC)
  fig3_headline_scatter.png -- goal error vs decision-point P95 field
  fig4_multimodality.png    -- k-NN + GMM dataset multi-modality audit
  fig6_rl_safety.png        -- BC vs RL-fine-tune peak-field distribution

fig5_diffusion_clean.png is a frame pulled from videos/diffusion_clean_dp.mp4
with ffmpeg, so it is not produced here.

Usage:
    python paper/figures/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_RES = _ROOT / 'results'
_FIG = _ROOT / 'paper' / 'figures'

# IEEEtran column / text widths, inches
COL_W = 3.5
TEXT_W = 7.16

# Okabe-Ito colourblind-safe palette
C_PLANNER = '#000000'
C_MLP = '#D55E00'      # vermillion
C_BC = '#0072B2'       # blue
C_RL = '#009E73'       # bluish green
C_START = '#117733'    # green
C_GOAL = '#E69F00'     # amber


# ---- Style ---------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 6.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.4,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


def _save(fig, name):
    fig.savefig(_FIG / name, dpi=300)
    plt.close(fig)
    print(f'  wrote {name}')


# ---- Fig 1: teaser -- MLP into obstacle vs diffusion around it ------------
# Re-rendered from the decision-point seed-6 trajectories that also drive
# videos/mlp_vs_diffusion_multimodal.mp4 (the raw video frame is only 720p).
def figure1():
    z = np.load(_RES / 'decision_point_eval_v2_multi3.npz', allow_pickle=True)
    obs = z['seed_6_obstacles'][0]
    ox, oy, _ = [float(v) for v in obs[0]]
    R = float(obs[1][0])
    start = np.array([-1.5, -1.5])
    goal = np.array([1.5, 1.5])

    mlp = z['seed_6_MLP_BC_xs'][:, :2]
    dif = z['seed_6_Diffusion_BC_xs'][:, :2]
    plan_L = z['seed_6_planner_path_0'][:, :2]
    plan_R = z['seed_6_planner_path_1'][:, :2]

    # soft penalty-field halo around the obstacle, just for readability
    n = 500
    gx, gy = np.meshgrid(np.linspace(-2, 2, n), np.linspace(-2, 2, n))
    dist = np.sqrt((gx - ox) ** 2 + (gy - oy) ** 2)
    field = np.clip((R + 0.80 - dist) / 0.80, 0, 1) ** 1.4
    field = np.where(field < 0.02, np.nan, field)

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W + 0.55, 3.55))
    panels = [
        (axes[0], mlp, C_MLP, 'Deterministic MLP — averages the two plans',
         'halts inside\nobstacle'),
        (axes[1], dif, C_BC, 'Diffusion policy — commits to one plan',
         'reaches\ngoal'),
    ]
    for ax, traj, col, title, endlabel in panels:
        ax.imshow(field, extent=[-2, 2, -2, 2], origin='lower',
                  cmap='Reds', alpha=0.55, vmin=0, vmax=1.3,
                  zorder=0, interpolation='bilinear')
        ax.add_patch(Circle((ox, oy), R, facecolor='#b2182b', alpha=0.55,
                            edgecolor='#7f0e1c', lw=1.0, zorder=1))
        ax.add_patch(Circle((ox, oy), R, fill=False, ls=(0, (4, 2)),
                            edgecolor='#7f0e1c', lw=0.9, zorder=2))
        # the two valid planner homotopy classes
        ax.plot(plan_L[:, 0], plan_L[:, 1], color='0.45', lw=1.0,
                ls='--', zorder=3, label='planner plans (left / right)')
        ax.plot(plan_R[:, 0], plan_R[:, 1], color='0.45', lw=1.0,
                ls='--', zorder=3)
        # student trajectory
        ax.plot(traj[:, 0], traj[:, 1], color=col, lw=2.4, zorder=5,
                solid_capstyle='round', label='student trajectory')
        ax.plot(traj[-1, 0], traj[-1, 1], 'o', color=col, ms=7,
                mec='white', mew=1.0, zorder=6)
        # start / goal
        ax.plot(*start, marker='^', color=C_START, ms=10, mec='white',
                mew=0.8, ls='none', zorder=6, label='start')
        ax.plot(*goal, marker='*', color=C_GOAL, ms=16, mec='black',
                mew=0.7, ls='none', zorder=6, label='goal')
        ax.annotate(endlabel, xy=(traj[-1, 0], traj[-1, 1]),
                    xytext=(traj[-1, 0] + 0.15, traj[-1, 1] - 0.78),
                    fontsize=6.8, ha='left', color=col,
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.8))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_title(title, color=col, fontsize=9, pad=4)
        ax.grid(True, lw=0.3, color='0.85')
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', framealpha=0.92, handlelength=1.6,
                  borderpad=0.4)
    axes[0].set_ylabel('y [m]')
    fig.tight_layout(pad=0.4, w_pad=1.2)
    _save(fig, 'fig1_teaser.png')


# ---- Fig 2: hierarchical planner pipeline schematic ----------------------
def figure2():
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    ax.set_xlim(0, 11.0)
    ax.set_ylim(0.1, 2.6)
    ax.axis('off')

    bw, bh, cy = 2.30, 1.32, 1.32
    blocks = [
        (2.65, 'A* voxel\nsearch', '5 cm grid', '#dbe9f6'),
        (5.45, 'Min-snap\nsmoothing', '7th-order poly', '#dff0e4'),
        (8.25, 'Nonlinear\nMPC', '300 ms horizon', '#fde8d6'),
    ]
    for cx, label, sub, fc in blocks:
        ax.add_patch(FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2), bw, bh,
            boxstyle='round,pad=0.02,rounding_size=0.10',
            facecolor=fc, edgecolor='#33415c', lw=1.1))
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=8.2, weight='bold')
        # sub-label sits just below the box so it can never be clipped
        ax.text(cx, cy - bh / 2 - 0.22, sub, ha='center', va='center',
                fontsize=6.4, style='italic', color='0.30')

    ax.text(0.55, cy, 'Goal', ha='center', va='center', fontsize=8.4,
            weight='bold', color='#33415c')
    ax.text(10.40, cy, 'Drone', ha='center', va='center', fontsize=8.4,
            weight='bold', color='#33415c')

    for x0, x1 in [(1.10, 1.45), (3.85, 4.25), (6.65, 7.05), (9.45, 9.85)]:
        ax.add_patch(FancyArrowPatch(
            (x0, cy), (x1, cy), arrowstyle='-|>', mutation_scale=11,
            lw=1.3, color='#33415c'))

    ax.text(5.45, 2.40, 'end-to-end: 23 ms per control cycle',
            ha='center', va='center', fontsize=6.8, style='italic',
            color='0.30')
    fig.tight_layout(pad=0.15)
    _save(fig, 'fig2_pipeline.png')


# ---- Fig 3: headline scatter -- goal error vs decision-point P95 field ----
def figure3():
    # one point per controller; values are the paper's final Tables I/II
    pts = [
        ('Hierarchical planner',  14,   0.06, C_PLANNER, 'o', 60),
        ('MLP DAgger+DART',       2406, 1.23, C_MLP,     's', 70),
        ('Diffusion (BC)',        100,  0.98, C_BC,      'D', 60),
        ('Diffusion (+RL)',       78,   0.32, C_RL,      '*', 150),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    ax.axhline(1.0, ls=':', color='0.4', lw=1.0)
    ax.text(11, 1.03, 'inside obstacle', fontsize=6.5, color='0.4')

    for _, gx, gy, col, mk, sz in pts:
        ax.scatter([gx], [gy], s=sz, marker=mk, color=col,
                   edgecolor='black', lw=0.7, zorder=5)

    ax.annotate('Hierarchical planner', (14, 0.06), (20, 0.16),
                fontsize=6.8, color=C_PLANNER)
    ax.annotate('MLP DAgger+DART', (2406, 1.23), (430, 1.30),
                fontsize=6.8, color=C_MLP, ha='left')
    ax.annotate('Diffusion (BC)', (100, 0.98), (118, 0.86),
                fontsize=6.8, color=C_BC)
    ax.annotate('Diffusion (+RL fine-tune)', (78, 0.32), (95, 0.40),
                fontsize=6.8, color=C_RL)

    # call out the diffusion-vs-MLP goal-error gap
    ax.annotate('', xy=(2406, 0.62), xytext=(100, 0.62),
                arrowprops=dict(arrowstyle='<|-|>', color='0.25', lw=1.1))
    ax.text(490, 0.66, r'$24\times$ goal error', fontsize=7.2,
            color='0.15', ha='center', weight='bold')

    ax.set_xscale('log')
    ax.set_xlim(9, 6000)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel('decision-point goal error [mm]  (log scale)')
    ax.set_ylabel('decision-point P95 obstacle field')
    ax.grid(True, which='both', lw=0.3, color='0.88')
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    _save(fig, 'fig3_headline_scatter.png')


# ---- Fig 4: dataset multi-modality audit (k-NN sweep + GMM cross-check) ---
def figure4():
    d = json.loads((_RES / 'dataset_audit_v2.json').read_text())
    sweep = d['threshold_sweep_per_seed_mean']['all']
    thr = np.array([0.05, 0.10, 0.15, 0.20, 0.30])
    frac = np.array([sweep[f'frac_mm@{t:.2f}'] for t in thr])
    knn = d['fraction_multimodal_kNN']            # 0.263
    gmm = d['gmm_audit']['n_components_counts']    # BIC-selected

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.7))

    # (a) k-NN action-span threshold sweep
    ax1.plot(thr, frac * 100, '-o', color=C_BC, ms=5, lw=1.6)
    ax1.axvline(0.20, ls='--', color=C_MLP, lw=1.1)
    ax1.scatter([0.20], [knn * 100], s=90, color=C_MLP, zorder=6,
                edgecolor='black', lw=0.7)
    ax1.annotate(f'action range → {knn * 100:.0f}%',
                 xy=(0.20, knn * 100), xytext=(0.214, 55),
                 fontsize=7.2, color=C_MLP, weight='bold',
                 arrowprops=dict(arrowstyle='-|>', color=C_MLP, lw=1.0))
    ax1.set_xlabel('k-NN cluster action-span threshold')
    ax1.set_ylabel('clusters multi-modal [%]')
    ax1.set_title('(a) Nearest-neighbour audit '
                  '(68,976 observation clusters)', fontsize=7.8)
    ax1.set_ylim(0, 108)
    ax1.set_xlim(0.03, 0.32)
    ax1.grid(True, lw=0.3, color='0.88')
    ax1.set_axisbelow(True)

    # (b) GMM mode-count cross-check
    comps = [1, 2, 3, 4]
    counts = [gmm[str(c)] for c in comps]
    bars = ax2.bar(comps, counts, width=0.62, color=C_BC,
                   edgecolor='black', lw=0.7)
    for c in comps[1:]:                       # highlight the multi-modal bars
        bars[c - 1].set_color(C_MLP)
    ax2.set_yscale('log')
    ax2.set_ylim(0.6, 4000)
    for rect, cnt in zip(bars, counts):
        ax2.text(rect.get_x() + rect.get_width() / 2,
                 cnt * 1.35, str(cnt), ha='center', fontsize=7)
    multimodal = sum(counts[1:]) / sum(counts) * 100
    ax2.set_xticks(comps)
    ax2.set_xlabel('BIC-selected Gaussian components per cluster')
    ax2.set_ylabel('number of clusters')
    ax2.set_title(f'(b) GMM cross-check: '
                  rf'$\geq$2 components in {multimodal:.0f}% of clusters',
                  fontsize=7.8)
    ax2.grid(True, axis='y', lw=0.3, color='0.88')
    ax2.set_axisbelow(True)

    fig.tight_layout(pad=0.3, h_pad=1.1)
    _save(fig, 'fig4_multimodality.png')


# ---- Fig 6: RL safety distribution, BC vs fine-tuned ---------------------
def figure6():
    bc = json.loads((_RES / 'decision_point_eval_v2_single.json').read_text())
    rl = json.loads((_RES / 'diffusion_v2_ppo_phase3_dp.json').read_text())
    bc_field = np.array([bc['per_seed_summary'][s]['Diffusion_BC']['max_field']
                         for s in bc['per_seed_summary']])
    rl_field = np.array([rl['per_seed'][s]['max_field']
                         for s in rl['per_seed']])
    # round to 2 dp so the plot matches the paper caption (1.36 -> 0.32)
    bc_p95 = round(np.percentile(bc_field, 95), 2)
    rl_p95 = round(np.percentile(rl_field, 95), 2)
    reduction = (bc_p95 - rl_p95) / bc_p95 * 100

    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    data = [bc_field, rl_field]
    cols = [C_BC, C_RL]
    parts = ax.violinplot(data, positions=[0, 1], widths=0.7,
                          showextrema=False)
    for body, col in zip(parts['bodies'], cols):
        body.set_facecolor(col)
        body.set_alpha(0.35)
        body.set_edgecolor(col)
    # jitter the 10 per-seed points so they do not stack
    rng = np.random.default_rng(0)
    for i, (vals, col) in enumerate(zip(data, cols)):
        jit = rng.uniform(-0.07, 0.07, len(vals))
        ax.scatter(i + jit, vals, s=16, color=col, edgecolor='black',
                   lw=0.4, zorder=5)
    for i, (p95, col) in enumerate(zip([bc_p95, rl_p95], cols)):
        ax.plot([i - 0.36, i + 0.36], [p95, p95], color=col, lw=2.0,
                zorder=6)
        ax.text(i, p95 + 0.06, f'P95 = {p95:.2f}', ha='center',
                fontsize=7, color=col, weight='bold')

    ax.axhline(1.0, ls=':', color='0.4', lw=1.0)
    ax.text(1.42, 1.03, 'inside\nobstacle', fontsize=6.3, color='0.4',
            ha='right', va='bottom')
    ax.annotate('', xy=(0.5, rl_p95), xytext=(0.5, bc_p95),
                arrowprops=dict(arrowstyle='-|>', color='0.2', lw=1.3))
    ax.text(0.58, (bc_p95 + rl_p95) / 2, f'−{reduction:.0f}% P95',
            fontsize=7.6, weight='bold', color='0.1', va='center')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Diffusion (BC)', '+ RL fine-tune'])
    ax.set_ylabel('decision-point peak obstacle field')
    ax.set_ylim(0, 1.6)
    ax.set_title('Decision-point safety: 10-seed peak-field distribution',
                 fontsize=8)
    ax.grid(True, axis='y', lw=0.3, color='0.88')
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    _save(fig, 'fig6_rl_safety.png')


if __name__ == '__main__':
    _FIG.mkdir(parents=True, exist_ok=True)
    print('rendering paper figures ->', _FIG)
    figure1()
    figure2()
    figure3()
    figure4()
    figure6()
    print('done (fig5 is an ffmpeg frame, see module docstring)')
