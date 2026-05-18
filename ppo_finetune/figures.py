"""
ppo_finetune/figures.py -- the 5 paper figures for the PPO fine-tune
result. Re-rendered from JSONs; no GPU needed.

Figures:
  1. ppo_training_trajectory.png     -- 4-panel per-iter trajectory
  2. ppo_safety_dist.png             -- violin plots BC / phase2 T=0.1 / T=0.5
  3. four_way_safety_comparison.png  -- scatter of all controllers
  4. ppo_checkpoint_pareto.png       -- iter5/10/15 trajectory in goal-vs-p95 space
  5. random_vs_dp_ppo.png            -- bar chart safety per controller

All figures saved as both .png (300 dpi) and .pdf in results/.

Usage:
  python -m ppo_finetune.figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_RES = _ROOT / 'results'


# ---- Style ---------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'bc':       '#888888',   # light gray
    't01':      '#1f77b4',   # blue
    't05':      '#2ca02c',   # green
    'planner':  '#d62728',   # red
    'mlp':      '#ff7f0e',   # orange
    'nmpc':     '#9467bd',   # purple
}


def _save(fig, name):
    """Save both .png and .pdf into results/."""
    for ext in ('png', 'pdf'):
        out = _RES / f'{name}.{ext}'
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
    print(f'  -> {name}.png + .pdf')


def _load_log(path):
    if not path.exists():
        return None
    return json.load(open(path))


def _load_eval(path):
    if not path.exists():
        return None
    return json.load(open(path))


# ----------------------------------------------------------------------------
# Figure 1: per-iter training trajectory (4 panels)
# ----------------------------------------------------------------------------
def figure_1_training_trajectory():
    """4-panel per-iter trajectory comparing T=0.1 and (if available) T=0.5."""
    log_t01 = _load_log(_RES / 'ppo_phase2_training_log_merged.json')
    log_t05 = _load_log(_RES / 'ppo_T05_training_log.json')
    assert log_t01 is not None, 'phase 2 merged log missing'

    rows01 = log_t01['iterations']
    rows05 = log_t05['iterations'] if log_t05 else []

    def _it(rows): return [r['iteration'] for r in rows]
    def _rt(rows): return [r['mean_return'] for r in rows]
    def _gd(rows): return [r['mean_goal_err_mm'] for r in rows]
    def _mfp95(rows): return [r['p95_max_field'] for r in rows]
    # Random p95 max_field: the phase-2 log has *all* max_field, not
    # decomposed. Without per-seed data we approximate "random p95" as the
    # overall p95 -- the dp seeds are 30/230 of the pool so their tail
    # dominates the p95 if they fail, which matches the brief's intent.
    def _smooth(xs, w=3):
        xs = np.asarray(xs, dtype=float)
        out = np.full_like(xs, np.nan)
        for i in range(len(xs)):
            lo = max(0, i - w + 1)
            out[i] = xs[lo:i+1].mean()
        return out

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    def _plot_pair(ax, ys_t01, ys_t05, ylabel, title):
        x01 = _it(rows01)
        ax.plot(x01, ys_t01, 'o-', color=COLORS['t01'],
                label='T=0.1 (phase 2)', lw=1.5, markersize=4, alpha=0.85)
        ax.plot(x01, _smooth(ys_t01), '-', color=COLORS['t01'],
                lw=2.5, alpha=0.55, label='_nolegend_')
        if rows05:
            x05 = _it(rows05)
            ax.plot(x05, ys_t05, 's-', color=COLORS['t05'],
                    label='T=0.5 (task A)', lw=1.5, markersize=4, alpha=0.85)
            ax.plot(x05, _smooth(ys_t05), '-', color=COLORS['t05'],
                    lw=2.5, alpha=0.55, label='_nolegend_')
        ax.axvline(15, color='black', linestyle=':', alpha=0.55,
                   label='iter15 (headline)')
        ax.set_xlabel('iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc='best')

    _plot_pair(axes[0], _rt(rows01), _rt(rows05) if rows05 else None,
               'mean undiscounted return', 'Training return (per iter)')
    _plot_pair(axes[1], _mfp95(rows01), _mfp95(rows05) if rows05 else None,
               'p95 max_field (all rollouts)',
               'Safety -- p95 max_field per iter (lower is safer)')
    _plot_pair(axes[2], [r['mean_max_field'] for r in rows01],
               [r['mean_max_field'] for r in rows05] if rows05 else None,
               'mean max_field (all rollouts)',
               'Safety -- mean max_field per iter')
    _plot_pair(axes[3], _gd(rows01), _gd(rows05) if rows05 else None,
               'mean goal err [mm]', 'Goal-reaching per iter')

    fig.suptitle('PPO fine-tune training trajectories',
                 fontsize=13, y=1.00)
    fig.tight_layout()
    _save(fig, 'ppo_training_trajectory')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: safety distribution violins
# ----------------------------------------------------------------------------
def figure_2_safety_dist():
    """Violins of max_field across 10-seed evals, random vs dp."""
    # Load the per-seed eval files
    bc_rand = _load_eval(_RES / 'decision_point_eval_v2_single.json')
    bc_dp_full = _load_eval(_RES / 'decision_point_eval_v2_single.json')
    ppo01_rand = _load_eval(_RES / 'diffusion_v2_ppo_phase2_10seed.json')
    ppo01_dp = _load_eval(_RES / 'diffusion_v2_ppo_phase2_dp.json')
    ppo05_rand = _load_eval(_RES / 'diffusion_v2_ppo_T05_10seed.json')
    ppo05_dp = _load_eval(_RES / 'diffusion_v2_ppo_T05_dp.json')

    def _vals(d, key='aggregate'):
        if d is None:
            return None
        if 'aggregate' in d and 'max_field' in d['aggregate']:
            return d['aggregate']['max_field'].get('values', [])
        return None

    # The BC v2 K=1 single-sample numbers live in the v2_single dp JSON;
    # for random we use the iter5 sibling values from the diffusion v2 K=1
    # eval that lives in the same file. The most authoritative random
    # numbers for BC v2 K=1 are in diffusion_distill_v2_10seed.json which
    # was the K=1 run.
    bc_rand_eval = _load_eval(_RES / 'diffusion_distill_v2_10seed.json')

    rand_lists = []
    rand_labels = []
    if bc_rand_eval is not None:
        rand_lists.append(bc_rand_eval['aggregate']['max_field']['values'])
        rand_labels.append('BC v2 (K=1)')
    if ppo01_rand is not None:
        rand_lists.append(ppo01_rand['aggregate']['max_field']['values'])
        rand_labels.append('PPO T=0.1 iter15')
    if ppo05_rand is not None:
        rand_lists.append(ppo05_rand['aggregate']['max_field']['values'])
        rand_labels.append('PPO T=0.5 iter15')

    dp_lists = []
    dp_labels = []
    # dp BC v2 K=1: from decision_point_eval_v2_single.json (Diffusion_BC row)
    if bc_dp_full is not None:
        dp_lists.append(bc_dp_full['aggregate']['Diffusion_BC']['max_field']['values'])
        dp_labels.append('BC v2 (K=1)')
    if ppo01_dp is not None:
        dp_lists.append(ppo01_dp['aggregate']['max_field']['values'])
        dp_labels.append('PPO T=0.1 iter15')
    if ppo05_dp is not None:
        dp_lists.append(ppo05_dp['aggregate']['max_field']['values'])
        dp_labels.append('PPO T=0.5 iter15')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    palette = [COLORS['bc'], COLORS['t01'], COLORS['t05']]

    for ax, lists, labels, title in [
        (axes[0], rand_lists, rand_labels,
         'Random 10-seed: max_field distribution'),
        (axes[1], dp_lists, dp_labels,
         'Decision-point 10-seed: max_field distribution'),
    ]:
        if not lists:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes)
            continue
        parts = ax.violinplot(lists, positions=range(len(lists)),
                              showmeans=False, showmedians=False,
                              showextrema=False, widths=0.7)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(palette[i % len(palette)])
            pc.set_alpha(0.55)
            pc.set_edgecolor('black')
        for i, vals in enumerate(lists):
            p95 = float(np.percentile(vals, 95))
            med = float(np.median(vals))
            ax.scatter([i] * len(vals), vals, s=18, color='black',
                       alpha=0.6, zorder=3)
            ax.hlines(p95, i - 0.3, i + 0.3, color='red',
                      lw=2, label='p95' if i == 0 else None)
            ax.hlines(med, i - 0.25, i + 0.25, color='white',
                      lw=2, label='median' if i == 0 else None)
            ax.text(i, p95 + 0.04, f'p95={p95:.2f}',
                    ha='center', fontsize=8, color='red')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel('max_field')
        ax.set_title(title)
        ax.axhline(1.0, color='black', linestyle=':', alpha=0.6,
                   label='inside obstacle')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, max(2.0, max(max(lst) for lst in lists) * 1.1))

    fig.tight_layout()
    _save(fig, 'ppo_safety_dist')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: Four-way safety comparison scatter
# ----------------------------------------------------------------------------
def figure_3_four_way_scatter():
    """Scatter: goal err vs dp p95 max_field for all controllers."""
    fwc = _load_eval(_RES / 'four_way_comparison.json')
    ppo_iter15 = _load_eval(_RES / 'diffusion_v2_ppo_phase2_dp.json')

    if fwc is None:
        raise SystemExit('four_way_comparison.json missing')
    dp = fwc['decision_point']
    rows = []
    # Planner
    rows.append(('Hierarchical planner', dp['Hierarchical_Planner']['goal_err_mm'],
                 dp['Hierarchical_Planner'].get('max_field_p95',
                     dp['Hierarchical_Planner']['max_field']),
                 COLORS['planner'], 'o'))
    rows.append(('MLP DAgger+DART', dp['MLP_DAgger_DART']['goal_err_mm'],
                 dp['MLP_DAgger_DART'].get('max_field_p95',
                     dp['MLP_DAgger_DART']['max_field']),
                 COLORS['mlp'], 's'))
    rows.append(('Diffusion BC v2 K=3', dp['Diffusion_BC_v2_multi']['goal_err_mm'],
                 dp['Diffusion_BC_v2_multi'].get('max_field_p95',
                     dp['Diffusion_BC_v2_multi']['max_field']),
                 COLORS['bc'], 'D'))
    # PPO (iter15 -- dp eval)
    if ppo_iter15 is not None:
        agg = ppo_iter15['aggregate']
        rows.append(('Diffusion + PPO iter15 (K=1)',
                     agg['goal_err_mm']['mean'], agg['max_field']['p95'],
                     COLORS['t01'], '*'))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, x, y, c, m in rows:
        ax.scatter(x, y, s=150 if m == '*' else 90, color=c, marker=m,
                   edgecolor='black', linewidth=1, zorder=3, label=label)
        ax.annotate(label, (x, y),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=9)
    # Arrow from BC -> PPO
    if ppo_iter15 is not None:
        bc_row = next(r for r in rows if 'BC v2' in r[0])
        ppo_row = next(r for r in rows if 'PPO' in r[0])
        arr = FancyArrowPatch((bc_row[1], bc_row[2]),
                              (ppo_row[1], ppo_row[2]),
                              arrowstyle='-|>', color=COLORS['t01'],
                              mutation_scale=20, lw=2,
                              connectionstyle='arc3,rad=0.18', alpha=0.7)
        ax.add_patch(arr)
        midx = 0.5 * (bc_row[1] + ppo_row[1])
        midy = 0.5 * (bc_row[2] + ppo_row[2]) + 0.05
        ax.text(midx, midy, 'PPO fine-tune', color=COLORS['t01'],
                fontsize=9, alpha=0.85)

    ax.set_xlabel('decision-point goal err [mm]')
    ax.set_ylabel('decision-point p95 max_field')
    ax.set_xscale('log')
    ax.set_xlim(8, 5000)
    ax.set_ylim(0, max(r[2] for r in rows) * 1.15)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.6,
               label='inside obstacle')
    ax.set_title('Decision-point safety vs goal-reaching')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, 'four_way_safety_comparison')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4: PPO checkpoint Pareto trajectory
# ----------------------------------------------------------------------------
def figure_4_ppo_pareto():
    """Per-checkpoint goal vs p95 max_field, with an annotated trajectory."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cps = []
    for it in (5, 6, 10, 15):
        rand_p = _RES / f'diffusion_v2_ppo_phase2_iter{it}_10seed.json'
        dp_p = _RES / f'diffusion_v2_ppo_phase2_iter{it}_dp.json'
        if not dp_p.exists():
            continue
        d = _load_eval(dp_p)
        agg = d['aggregate']
        cps.append((it, agg['goal_err_mm']['mean'], agg['max_field']['p95']))
    # BC v2 K=1
    bc = _load_eval(_RES / 'decision_point_eval_v2_single.json')
    if bc is not None:
        bc_row = bc['aggregate']['Diffusion_BC']
        bc_p95 = float(np.percentile(bc_row['max_field']['values'], 95))
        cps.insert(0, (0, bc_row['goal_err_mm']['mean'], bc_p95))
    # T=0.5
    t05_dp = _RES / 'diffusion_v2_ppo_T05_dp.json'
    if t05_dp.exists():
        d = _load_eval(t05_dp)
        agg = d['aggregate']
        ax.scatter(agg['goal_err_mm']['mean'], agg['max_field']['p95'],
                   s=180, color=COLORS['t05'], marker='*',
                   edgecolor='black', zorder=4,
                   label='PPO T=0.5 iter15')

    if cps:
        xs = [c[1] for c in cps]
        ys = [c[2] for c in cps]
        ax.plot(xs, ys, '-', color=COLORS['t01'], lw=2, alpha=0.6,
                zorder=2, label='PPO T=0.1 iter sequence')
        for it, x, y in cps:
            label = f'BC (iter0)' if it == 0 else f'iter{it}'
            ax.scatter(x, y, s=130, color=COLORS['t01'] if it > 0 else COLORS['bc'],
                       marker='o', edgecolor='black', lw=1, zorder=3)
            ax.annotate(label, (x, y), xytext=(8, 6),
                        textcoords='offset points', fontsize=9)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.5,
               label='inside obstacle')
    # Annotate iter15 as headline
    headline = next((c for c in cps if c[0] == 15), None)
    if headline is not None:
        ax.annotate('headline checkpoint',
                    (headline[1], headline[2]),
                    xytext=(40, 30), textcoords='offset points',
                    fontsize=9, color=COLORS['t01'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['t01']))
    ax.set_xlabel('decision-point goal err [mm]')
    ax.set_ylabel('decision-point p95 max_field')
    ax.set_title('PPO checkpoint trajectory (decision-point eval)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, 'ppo_checkpoint_pareto')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5: random vs dp p95 bar chart
# ----------------------------------------------------------------------------
def figure_5_random_vs_dp_bars():
    """Bar chart, 4 controllers x (random p95, dp p95)."""
    fwc = _load_eval(_RES / 'four_way_comparison.json')
    ppo15_rand = _load_eval(_RES / 'diffusion_v2_ppo_phase2_10seed.json')
    ppo15_dp = _load_eval(_RES / 'diffusion_v2_ppo_phase2_dp.json')

    # All random p95 numbers come from the raw 10-seed eval JSONs.
    def _p95_of(path, key='max_field'):
        d = _load_eval(_RES / path)
        if d is None:
            return None
        vals = d['aggregate'][key].get('values',
                                       d['aggregate'][key].get('mean'))
        return float(np.percentile(vals, 95))

    mlp_rand_p95 = _p95_of('mlp_distill_10seed.json')
    bc_rand_p95 = _p95_of('diffusion_distill_v2_10seed.json')
    mlp_dp_p95 = fwc['decision_point']['MLP_DAgger_DART']['max_field_p95']
    bc_dp_p95 = fwc['decision_point']['Diffusion_BC_v2_single']['max_field_p95']

    rows = [
        ('MLP DAgger+DART', mlp_rand_p95, mlp_dp_p95, COLORS['mlp']),
        ('Diffusion BC v2 K=1', bc_rand_p95, bc_dp_p95, COLORS['bc']),
        ('Diffusion + PPO T=0.1 iter15',
         ppo15_rand['aggregate']['max_field']['p95'],
         ppo15_dp['aggregate']['max_field']['p95'],
         COLORS['t01']),
    ]
    # Add T=0.5 if available
    t05_rand = _RES / 'diffusion_v2_ppo_T05_10seed.json'
    t05_dp = _RES / 'diffusion_v2_ppo_T05_dp.json'
    if t05_rand.exists() and t05_dp.exists():
        rows.append(('Diffusion + PPO T=0.5 iter15',
                     _load_eval(t05_rand)['aggregate']['max_field']['p95'],
                     _load_eval(t05_dp)['aggregate']['max_field']['p95'],
                     COLORS['t05']))

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [r[0] for r in rows]
    rand_vals = [r[1] for r in rows]
    dp_vals = [r[2] for r in rows]
    x = np.arange(len(labels))
    width = 0.4
    b1 = ax.bar(x - width/2, rand_vals, width, color=[r[3] for r in rows],
                edgecolor='black', alpha=0.85, label='random p95')
    b2 = ax.bar(x + width/2, dp_vals, width, color=[r[3] for r in rows],
                edgecolor='black', alpha=0.55, hatch='//', label='dp p95')
    for i, (bar, val) in enumerate(zip(b1, rand_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.2f}', ha='center', fontsize=9)
    for i, (bar, val) in enumerate(zip(b2, dp_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.2f}', ha='center', fontsize=9)
    # Annotate PPO improvement vs BC
    if len(rows) >= 3:
        bc_idx = 1
        for ppo_idx in range(2, len(rows)):
            d_rand = (rand_vals[ppo_idx] - rand_vals[bc_idx]) / rand_vals[bc_idx]
            d_dp = (dp_vals[ppo_idx] - dp_vals[bc_idx]) / dp_vals[bc_idx]
            ax.text(x[ppo_idx], -0.18,
                    f'Δ rand: {100*d_rand:+.0f}%\nΔ dp:  {100*d_dp:+.0f}%',
                    ha='center', fontsize=8, color=rows[ppo_idx][3])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel('p95 max_field')
    ax.set_title('Safety improvement of PPO fine-tune vs distilled baselines')
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.5,
               label='inside obstacle')
    ax.legend(loc='upper right')
    ax.set_ylim(-0.05, max(max(rand_vals), max(dp_vals)) * 1.20)
    fig.tight_layout()
    _save(fig, 'random_vs_dp_ppo')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=int, nargs='*',
                    help='subset of figures to render (1..5); default all')
    args = ap.parse_args()
    todo = args.only if args.only else [1, 2, 3, 4, 5]
    for i in todo:
        print(f'\n[figures] building figure {i}')
        {1: figure_1_training_trajectory,
         2: figure_2_safety_dist,
         3: figure_3_four_way_scatter,
         4: figure_4_ppo_pareto,
         5: figure_5_random_vs_dp_bars}[i]()


if __name__ == '__main__':
    main()
