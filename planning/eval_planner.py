"""
eval_planner.py -- 10-seed evaluation of the hierarchical planner.

Reproduces the 10-seed canonical suite used by
results/sdf_dagger_obstacle_10seed.json (same obstacle seeds, same start/goal)
and saves results/planner_10seed.json in a structure that can be merged into
the existing comparison.

Run:
    python planning/eval_planner.py
"""
from __future__ import annotations

import sys, time, json
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / 'src'))

from hierarchical_ctrl import run_hierarchical
from obstacle_course import make_obstacles


SEEDS = [42, 7, 13, 99, 256, 128, 314, 2024, 777, 1337]
START = np.array([-1.5, -1.5, 1.0])
GOAL = np.array([1.5, 1.5, 1.0])


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _sem(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std() / np.sqrt(len(arr)))


def run_eval(seeds=SEEDS, save_path: str | None = None,
             save_first_seed: int | None = None, verbose: bool = True):
    rows = []
    detail = {}
    for seed in seeds:
        if verbose:
            print(f"\n[seed {seed}]")
        obstacles = make_obstacles(seed=seed)
        t0 = time.time()
        res = run_hierarchical(START, GOAL, obstacles, verbose=verbose,
                               return_trajectory=False)
        wall = time.time() - t0
        rows.append(res)
        detail[str(seed)] = {k: v for k, v in res.items()
                              if not isinstance(v, (np.ndarray, list))}
        if verbose:
            print(f"  [done] wall {wall:.1f} s")

    # Aggregate.
    keys = ['goal_err_mm', 'max_field', 'mean_field', 'path_len_m',
            'mean_speed_mps', 'median_solve_ms', 'mean_solve_ms',
            'planning_time_ms', 'astar_ms', 'smooth_ms', 'voxel_build_ms']
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows]
        m, s = _mean_std(vals)
        agg[k] = dict(mean=m, std=s, sem=_sem(vals), values=vals)

    # Match the SDF-baseline schema as closely as possible.
    canonical = dict(
        goal_err_mm=agg['goal_err_mm']['mean'],
        goal_std=agg['goal_err_mm']['std'],
        max_field=agg['max_field']['mean'],
        field_std=agg['max_field']['std'],
        teacher_dev_mm=None,
        dev_std=None,
        path_len_m=agg['path_len_m']['mean'],
        path_std=agg['path_len_m']['std'],
        speed_us=agg['median_solve_ms']['mean'] * 1000,  # us, mirroring baseline
    )

    out = dict(
        metadata=dict(
            n_seeds=len(seeds),
            obstacle_seeds=list(seeds),
            task='8-obstacle Gaussian course, start=(-1.5,-1.5,1) goal=(1.5,1.5,1)',
            controller='Hierarchical_Planner',
            planner='A* (26-conn, 5cm) + min-snap 7th-order + SE(3) NMPC tracker',
            voxel_resolution_m=0.05,
            safety_margin_m=0.15,
            target_avg_speed_mps=0.8,
            nmpc_horizon_N=15,
            nmpc_dt=0.02,
        ),
        obstacle_course=dict(Hierarchical_Planner=canonical),
        aggregate=agg,
        per_seed=detail,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(out, f, indent=2, default=lambda o: float(o)
                       if isinstance(o, (np.floating, np.integer)) else str(o))
        if verbose:
            print(f"\nSaved {save_path}")

    return out


def print_comparison_table(out, baseline_path=None):
    if baseline_path is None:
        baseline_path = str(_HERE.parent / 'results'
                            / 'sdf_dagger_obstacle_10seed.json')
    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
        oc = baseline.get('obstacle_course', {})
    except FileNotFoundError:
        oc = {}

    hp = out['obstacle_course']['Hierarchical_Planner']

    print('\n' + '=' * 78)
    print(' 10-seed obstacle-course comparison')
    print('=' * 78)
    hdr = f"  {'Controller':<22s} | {'Goal err':>13s} | {'Max field':>13s} | {'Path len':>12s} | {'Speed':>8s}"
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    def row(name, e):
        if 'speed_us' in e:
            sp = e['speed_us']
            if sp >= 1e4:
                sp_str = f"{sp/1000:.0f}ms"
            else:
                sp_str = f"{int(sp)}us"
        else:
            sp_str = '-'
        ge = e.get('goal_err_mm', float('nan'))
        gs = e.get('goal_std', 0)
        mf = e.get('max_field', float('nan'))
        fs = e.get('field_std', 0)
        pl = e.get('path_len_m', float('nan'))
        ps = e.get('path_std', 0)
        print(f"  {name:<22s} | {ge:>5.0f}+/-{gs:>4.0f} mm | "
              f"{mf:>5.3f}+/-{fs:.3f} | "
              f"{pl:>5.2f}+/-{ps:.2f} m | {sp_str:>8s}")

    for name in ('NMPC_teacher', 'Linear_MPC_blind', 'BC_SDF',
                 'DAgger_SDF', 'DAgger_DART_SDF'):
        if name in oc:
            row(name, oc[name])
    row('Hierarchical (ours)', hp)
    print('=' * 78)


if __name__ == '__main__':
    out = run_eval(
        save_path=str(_HERE.parent / 'results' / 'planner_10seed.json'),
        verbose=True)
    print_comparison_table(out)
