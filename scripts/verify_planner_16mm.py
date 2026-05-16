#!/usr/bin/env python3
"""Verify the 16mm goal error claim by re-running 2 seeds and inspecting trajectories."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '.')

from planning.hierarchical_ctrl import run_hierarchical
from src.obstacle_course import make_obstacles    # <-- fixed name

BEST_SEED = 42    # 4.8 mm reported
WORST_SEED = 777  # 55.1 mm reported

START = np.array([-1.5, -1.5, 1.0])
GOAL = np.array([1.5, 1.5, 1.0])

results = {}

for seed in [BEST_SEED, WORST_SEED]:
    print(f"\n=== running seed {seed} ===")
    obstacles = make_obstacles(seed=seed, n=8)
    out = run_hierarchical(START, GOAL, obstacles, dt=0.01, T_max=10.0)

    # inspect what run_hierarchical actually returns
    print(f"  returned dict keys: {list(out.keys())}")

    # try common key names for the trajectory
    traj_key = None
    for candidate in ['executed_trajectory', 'trajectory', 'states',
                      'executed_states', 'x_traj', 'xs']:
        if candidate in out:
            traj_key = candidate
            break
    if traj_key is None:
        print("  ERROR: no trajectory key found. Full dict:")
        for k, v in out.items():
            if hasattr(v, 'shape'):
                print(f"    {k}: shape {v.shape}")
            else:
                print(f"    {k}: {type(v).__name__} = {v}")
        continue
    traj = np.array(out[traj_key])
    print(f"  found trajectory under key '{traj_key}', shape {traj.shape}")

    # normalize to (N, state_dim) — handle (state_dim, N) by transpose
    if traj.shape[0] < traj.shape[1]:
        traj = traj.T
        print(f"  transposed to {traj.shape}")

    final_pos = traj[-1, :3]
    goal_err = np.linalg.norm(final_pos - GOAL) * 1000

    print(f"  final position:   {final_pos}")
    print(f"  goal position:    {GOAL}")
    print(f"  L2 err (mm):      {goal_err:.2f}")
    reported = {42: 4.8, 777: 55.1}[seed]
    print(f"  reported in JSON: {reported} mm")
    print(f"  discrepancy:      {abs(goal_err - reported):.2f} mm")

    results[seed] = {'traj': traj, 'goal_err': goal_err, 'final_pos': final_pos}

# plot both
if len(results) == 2:
    fig = plt.figure(figsize=(14, 8))

    for idx, seed in enumerate([BEST_SEED, WORST_SEED]):
        r = results[seed]
        traj = r['traj']

        # 3D view
        ax1 = fig.add_subplot(2, 2, idx*2 + 1, projection='3d')
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=1)
        ax1.scatter(*START, c='green', s=120, marker='o', label='start')
        ax1.scatter(*GOAL, c='red', s=200, marker='*', label='goal')
        ax1.scatter(*traj[-1, :3], c='blue', s=120, marker='x',
                    label=f'final ({r["goal_err"]:.1f} mm)')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        ax1.legend(fontsize=8)
        ax1.set_title(f'seed {seed}: full 3D trajectory')

        # 2D zoom on final approach
        ax2 = fig.add_subplot(2, 2, idx*2 + 2)
        n_tail = min(100, len(traj))
        last_steps = traj[-n_tail:]
        ax2.plot(last_steps[:, 0], last_steps[:, 1], 'b-', linewidth=1.5)
        ax2.scatter(GOAL[0], GOAL[1], c='red', s=300, marker='*', label='goal')
        ax2.scatter(traj[-1, 0], traj[-1, 1], c='blue', s=120, marker='x', label='final')

        for r_m, lbl in [(0.005, '5 mm'), (0.020, '20 mm'), (0.060, '60 mm')]:
            c = plt.Circle((GOAL[0], GOAL[1]), r_m, fill=False, linewidth=1, label=lbl)
            ax2.add_patch(c)

        ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
        ax2.set_aspect('equal')
        ax2.legend(fontsize=8, loc='lower left')
        ax2.set_title(f'seed {seed}: final approach (last {n_tail*0.01:.1f}s)')

    plt.tight_layout()
    out_path = 'results/sanity_check_goal_err.png'
    plt.savefig(out_path, dpi=120)
    print(f"\nsaved plot to {out_path}")

    print("\n=== VERIFICATION SUMMARY ===")
    print(f"  seed {BEST_SEED} (best):  measured {results[BEST_SEED]['goal_err']:.1f} mm vs reported 4.8 mm")
    print(f"  seed {WORST_SEED} (worst): measured {results[WORST_SEED]['goal_err']:.1f} mm vs reported 55.1 mm")
    print("  If both match within ~2 mm, the 16 mm aggregate is real.")
else:
    print("\nERROR: not enough valid runs to compare. Check the dict-key output above.")