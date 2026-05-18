"""
ppo_finetune/plot_phase2_trajectory.py -- per-iter trajectory plot for
the phase-2 training log: mean_return / mean_goal_err / mean_max_field
vs iteration.

Usage:
  python -m ppo_finetune.plot_phase2_trajectory \\
      --log results/ppo_phase2_training_log.json \\
      --out results/ppo_phase2_trajectory.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', type=str,
                    default='results/ppo_phase2_training_log.json')
    ap.add_argument('--out', type=str,
                    default='results/ppo_phase2_trajectory.png')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    log = json.load(open(root / args.log))
    rows = log['iterations']
    if not rows:
        raise SystemExit('no iterations logged yet')

    it = [r['iteration'] for r in rows]
    ret = [r['mean_return'] for r in rows]
    goal = [r['mean_goal_err_mm'] for r in rows]
    goal_rand = [r.get('mean_goal_err_mm_random', r['mean_goal_err_mm'])
                 for r in rows]
    max_field_mean = [r['mean_max_field'] for r in rows]
    max_field_p95 = [r['p95_max_field'] for r in rows]
    drift_rel = [r.get('param_drift_relative', 0.0) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=120)
    axes = axes.flatten()

    axes[0].plot(it, ret, 'o-', color='#4c78a8', lw=2, markersize=6)
    axes[0].set_xlabel('iteration'); axes[0].set_ylabel('mean undiscounted return')
    axes[0].set_title('Mean rollout return')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(it, goal, 'o-', color='#54a24b', lw=2, markersize=6,
                 label='all seeds')
    axes[1].plot(it, goal_rand, 's--', color='#54a24b', lw=1.5, markersize=5,
                 alpha=0.7, label='random only')
    axes[1].axhline(93, color='gray', linestyle=':',
                    label='BC baseline (random) 93 mm')
    axes[1].set_xlabel('iteration'); axes[1].set_ylabel('mean goal err [mm]')
    axes[1].set_title('Goal reaching')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot(it, max_field_mean, 'o-', color='#e45756', lw=2,
                 markersize=6, label='mean')
    axes[2].plot(it, max_field_p95, 's--', color='#e45756', lw=1.5,
                 markersize=5, alpha=0.7, label='p95')
    axes[2].axhline(1.0, color='black', linestyle=':', alpha=0.5,
                    label='inside obstacle')
    axes[2].set_xlabel('iteration'); axes[2].set_ylabel('max_field')
    axes[2].set_title('Safety (max obstacle field across rollouts)')
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    axes[3].plot(it, drift_rel, 'o-', color='#9d755d', lw=2, markersize=6)
    axes[3].axhline(0.05, color='red', linestyle=':',
                    label='abort cap (5%)')
    axes[3].set_xlabel('iteration')
    axes[3].set_ylabel('||θ - θ_init|| / ||θ_init||')
    axes[3].set_title('Cumulative relative param drift')
    axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.3)

    aborted = log.get('aborted', False)
    title = f'Phase-2 AWR fine-tune ({len(rows)} iterations)'
    if aborted:
        title += f' -- ABORTED: {log.get("abort_reason", "?")}'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'-> {out_path}')


if __name__ == '__main__':
    main()
