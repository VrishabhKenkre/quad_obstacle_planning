"""
four_way_comparison.py -- merge the existing baseline results with the
two new student results, emit the four-row comparison table as JSON +
console, and produce results/comparison_plot.png (goal-err vs latency
scatter).

Inputs:
  results/planner_10seed.json                     hierarchical planner
  results/sdf_dagger_obstacle_10seed.json          NMPC teacher + old students
  results/mlp_distill_10seed.json                 MLP DAgger+DART (planner)
  results/mlp_bc_only_10seed.json                 MLP BC-only (planner)
  results/diffusion_distill_10seed.json           Diffusion BC (planner)
  results/decision_point_eval.json                Decision-point breakdown

Outputs:
  results/four_way_comparison.json
  results/comparison_plot.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def load(p):
    with open(p) as f:
        return json.load(f)


def _median_p95(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 95))


def main():
    pl = load(_ROOT / 'results' / 'planner_10seed.json')
    sdf = load(_ROOT / 'results' / 'sdf_dagger_obstacle_10seed.json')
    mlp = load(_ROOT / 'results' / 'mlp_distill_10seed.json')
    mlp_bc = load(_ROOT / 'results' / 'mlp_bc_only_10seed.json')
    # Three diffusion variants now available:
    diff_v1 = load(_ROOT / 'results' / 'diffusion_distill_10seed.json')
    diff_v2_single = load(_ROOT / 'results' / 'diffusion_distill_v2_10seed.json')
    diff_v2_multi = load(_ROOT / 'results' / 'diffusion_distill_v2_multi3_10seed.json')
    # Use v2-multi as the headline diffusion row for the random table.
    diff = diff_v2_multi
    dp = load(_ROOT / 'results' / 'decision_point_eval.json')
    dp_v2_single = load(_ROOT / 'results' / 'decision_point_eval_v2_single.json')
    dp_v2_multi = load(_ROOT / 'results' / 'decision_point_eval_v2_multi3.json')

    # NMPC teacher row from the existing baseline JSON
    nmpc = sdf['obstacle_course']['NMPC_teacher']

    # Random-seed rows.
    rows = [
        ('NMPC teacher (reactive)', nmpc['goal_err_mm'], nmpc['goal_std'],
         nmpc['max_field'], nmpc['field_std'], nmpc['speed_us'] / 1000),
        ('Hierarchical planner', pl['obstacle_course']['Hierarchical_Planner']['goal_err_mm'],
         pl['obstacle_course']['Hierarchical_Planner']['goal_std'],
         pl['obstacle_course']['Hierarchical_Planner']['max_field'],
         pl['obstacle_course']['Hierarchical_Planner']['field_std'],
         pl['obstacle_course']['Hierarchical_Planner']['speed_us'] / 1000),
        ('MLP DAgger+DART (planner)',
         mlp['obstacle_course']['MLP_DAgger_DART_planner']['goal_err_mm'],
         mlp['obstacle_course']['MLP_DAgger_DART_planner']['goal_std'],
         mlp['obstacle_course']['MLP_DAgger_DART_planner']['max_field'],
         mlp['obstacle_course']['MLP_DAgger_DART_planner']['field_std'],
         mlp['obstacle_course']['MLP_DAgger_DART_planner']['speed_us'] / 1000),
        ('Diffusion BC v2 K=3 (planner)',
         diff['obstacle_course']['Diffusion_BC_planner']['goal_err_mm'],
         diff['obstacle_course']['Diffusion_BC_planner']['goal_std'],
         diff['obstacle_course']['Diffusion_BC_planner']['max_field'],
         diff['obstacle_course']['Diffusion_BC_planner']['field_std'],
         diff['obstacle_course']['Diffusion_BC_planner']['speed_us'] / 1000),
    ]

    # Median + p95 max-field for the diffusion rows (random and dp).
    def med_p95(j, k='Diffusion_BC'):
        a = j.get('aggregate', {})
        if 'max_field' in a:
            vals = a['max_field']['values']
        else:
            vals = a[k]['max_field']['values']
        return _median_p95(vals)
    diff_random_summary = {
        'v1':       med_p95(diff_v1),
        'v2_single':med_p95(diff_v2_single),
        'v2_multi': med_p95(diff_v2_multi),
    }
    diff_dp_summary = {
        'v1':       med_p95(dp,             'Diffusion_BC'),
        'v2_single':med_p95(dp_v2_single,    'Diffusion_BC'),
        'v2_multi': med_p95(dp_v2_multi,     'Diffusion_BC'),
    }

    # Decision-point aggregate. Pull each model from its own most-current
    # eval JSON (v2 for the diffusion student; v1 numbers for the MLPs
    # which are bit-identical between v1 and v2 obstacle layouts).
    def dp_row(j, name):
        a = j['aggregate'].get(name, {})
        ge = a.get('goal_err_mm', {}); mf = a.get('max_field', {})
        med, p95 = ((_median_p95(mf['values']) if 'values' in mf
                     else (float('nan'), float('nan'))))
        return dict(
            goal_err_mm=ge.get('mean', float('nan')),
            goal_std=ge.get('std', float('nan')),
            max_field=mf.get('mean', float('nan')),
            field_std=mf.get('std', float('nan')),
            max_field_median=med,
            max_field_p95=p95,
        )

    decision_table = dict(
        # planner row comes from the v2-single eval (which has the planner
        # in the loop; the multi eval was run with --no-planner for speed)
        Hierarchical_Planner=dp_row(dp_v2_single, 'Hierarchical_Planner'),
        MLP_DAgger_DART=dp_row(dp, 'MLP_DAgger'),
        MLP_BC_only=dp_row(dp, 'MLP_BC'),
        Diffusion_BC_v1=dp_row(dp, 'Diffusion_BC'),
        Diffusion_BC_v2_single=dp_row(dp_v2_single, 'Diffusion_BC'),
        Diffusion_BC_v2_multi=dp_row(dp_v2_multi, 'Diffusion_BC'),
    )

    # Console print of the four-row table
    print('=' * 78)
    print(' Four-way comparison: 10-seed random obstacle course')
    print('=' * 78)
    hdr = f'  {"Controller":<32s} | {"Goal err":>16s} | {"Max field":>16s} | {"Latency":>10s}'
    print(hdr); print('  ' + '-' * (len(hdr) - 2))
    for name, ge, gs, mf, fs, lat_ms in rows:
        lat_s = (f'{lat_ms*1000:>5.0f} us' if lat_ms < 1.0
                 else f'{lat_ms:>5.1f} ms')
        print(f'  {name:<32s} | {ge:>7.0f} +/- {gs:>4.0f} mm | '
              f'{mf:.3f} +/- {fs:.3f} | {lat_s:>10s}')
    print()
    print('Plus the BC-only MLP-of-planner sanity:')
    bc = mlp_bc['obstacle_course']['MLP_DAgger_DART_planner']
    print(f'  {"MLP BC-only (planner)":<32s} | '
          f'{bc["goal_err_mm"]:>7.0f} +/- {bc["goal_std"]:>4.0f} mm | '
          f'{bc["max_field"]:.3f} +/- {bc["field_std"]:.3f} | '
          f'{bc["speed_us"]:>5.0f} us')

    print()
    print('=' * 100)
    print(' Decision-point seeds (10 bimodal seeds) — diffusion vs MLP gap')
    print(' Reporting mean(+-std), median, and p95 of max_field because the')
    print(' diffusion samplers produce heavy-tailed safety distributions.')
    print('=' * 100)
    hdr = f'  {"Controller":<28s} | {"Goal err":>16s} | {"Field mean+/-std":>18s} | {"Field med":>10s} | {"Field p95":>10s}'
    print(hdr); print('  ' + '-' * (len(hdr) - 2))
    for name in ['Hierarchical_Planner', 'MLP_BC_only', 'MLP_DAgger_DART',
                 'Diffusion_BC_v1', 'Diffusion_BC_v2_single',
                 'Diffusion_BC_v2_multi']:
        r = decision_table[name]
        print(f'  {name:<28s} | {r["goal_err_mm"]:>7.0f} +/- '
              f'{r["goal_std"]:>4.0f} mm | '
              f'{r["max_field"]:>6.3f} +/- {r["field_std"]:>6.3f}   | '
              f'{r["max_field_median"]:>10.3f} | '
              f'{r["max_field_p95"]:>10.3f}')

    print()
    print('=' * 100)
    print(' Random 10-seed obstacle course — diffusion variants (median + p95)')
    print('=' * 100)
    for tag, (med, p95) in diff_random_summary.items():
        print(f'  Diffusion_{tag:<11s} max_field median={med:.3f}, p95={p95:.3f}')

    # Combined JSON output
    out = dict(
        random_10seed=[dict(controller=name, goal_err_mm=ge, goal_std=gs,
                            max_field=mf, field_std=fs, latency_ms=lat_ms)
                       for (name, ge, gs, mf, fs, lat_ms) in rows]
                       + [dict(controller='MLP BC-only (planner)',
                                goal_err_mm=bc['goal_err_mm'],
                                goal_std=bc['goal_std'],
                                max_field=bc['max_field'],
                                field_std=bc['field_std'],
                                latency_ms=bc['speed_us'] / 1000)],
        decision_point=decision_table,
    )
    out_json = _ROOT / 'results' / 'four_way_comparison.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[comparison] -> {out_json}")

    # --- scatter plot: goal err vs inference latency ----------------
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    points = [
        ('NMPC teacher', 46, 59),
        ('Hierarchical planner', 16, 23),
        ('MLP DAgger+DART (planner)', rows[2][1], rows[2][5]),
        ('MLP BC-only (planner)', bc['goal_err_mm'], bc['speed_us'] / 1000),
        ('Diffusion BC (planner)', rows[3][1], rows[3][5]),
    ]
    colors = ['#4d4d4d', '#2980b9', '#e67e22', '#f39c12', '#8e44ad']
    markers = ['s', 'D', 'o', 'o', '^']
    for (name, ge, lat), c, m in zip(points, colors, markers):
        ax.scatter(lat, ge, color=c, marker=m, s=130, edgecolor='black',
                   label=name, zorder=3)
        ax.annotate(name, (lat, ge), textcoords='offset points',
                    xytext=(10, 4), fontsize=9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Inference latency (ms, log scale)')
    ax.set_ylabel('Goal error (mm, log scale)')
    ax.set_title('Quadrotor obstacle course (10 seeds): goal error vs inference latency')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    out_png = _ROOT / 'results' / 'comparison_plot.png'
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"[comparison] -> {out_png}")

    # --- bonus: bar chart comparing random vs decision-point --------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names_short = ['Planner', 'MLP DAgger', 'MLP BC', 'Diffusion']

    def ge_for(name_short, table):
        m = {'Planner': 'Hierarchical_Planner',
             'MLP DAgger': 'MLP_DAgger_DART',
             'MLP BC': 'MLP_BC_only',
             'Diffusion': 'Diffusion_BC_v2_multi'}
        return table[m[name_short]]['goal_err_mm']

    def std_for(name_short, table):
        m = {'Planner': 'Hierarchical_Planner',
             'MLP DAgger': 'MLP_DAgger_DART',
             'MLP BC': 'MLP_BC_only',
             'Diffusion': 'Diffusion_BC_v2_multi'}
        return table[m[name_short]]['goal_std']

    # Random seeds (from rows + bc)
    random_vals = [
        pl['obstacle_course']['Hierarchical_Planner']['goal_err_mm'],
        rows[2][1],
        bc['goal_err_mm'],
        rows[3][1],
    ]
    random_stds = [
        pl['obstacle_course']['Hierarchical_Planner']['goal_std'],
        rows[2][2],
        bc['goal_std'],
        rows[3][2],
    ]
    dp_vals = [ge_for(n, decision_table) for n in names_short]
    dp_stds = [std_for(n, decision_table) for n in names_short]

    axes[0].bar(names_short, random_vals, yerr=random_stds,
                color=['#2980b9', '#e67e22', '#f39c12', '#8e44ad'],
                edgecolor='black', alpha=0.85)
    axes[0].set_yscale('log')
    axes[0].set_title('Random 10-seed obstacle course')
    axes[0].set_ylabel('Goal error (mm, log scale)')
    axes[0].grid(True, axis='y', which='both', alpha=0.3)

    axes[1].bar(names_short, dp_vals, yerr=dp_stds,
                color=['#2980b9', '#e67e22', '#f39c12', '#8e44ad'],
                edgecolor='black', alpha=0.85)
    axes[1].set_yscale('log')
    axes[1].set_title('Decision-point 10-seed (bimodal)')
    axes[1].set_ylabel('Goal error (mm, log scale)')
    axes[1].grid(True, axis='y', which='both', alpha=0.3)

    plt.suptitle('MLP vs Diffusion: collapsing on multi-modal teacher')
    plt.tight_layout()
    out_png2 = _ROOT / 'results' / 'comparison_random_vs_dp.png'
    plt.savefig(out_png2, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"[comparison] -> {out_png2}")


if __name__ == '__main__':
    main()
