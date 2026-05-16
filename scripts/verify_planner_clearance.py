#!/usr/bin/env python3
"""Verify the planner keeps the drone outside obstacles by the criterion
the planner was actually designed for: Gaussian field value below threshold."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')

from planning.hierarchical_ctrl import run_hierarchical
from obstacle_course import make_obstacles, obstacle_field_value

SEEDS = [42, 7, 13, 99, 256, 128, 314, 2024, 777, 1337]
START = np.array([-1.5, -1.5, 1.0])
GOAL = np.array([1.5, 1.5, 1.0])

# the planner was tuned to keep field < 0.3 (corresponds to ~1.5 sigma)
PLANNER_THRESHOLD = 0.3
# anything above 0.6 (corresponds to ~1 sigma, i.e. "really close to center")
# is a genuine "near-collision" we'd worry about
NEAR_COLLISION = 0.6


def parse_obstacles(obstacles):
    centers = []
    sigmas = []
    for o in obstacles:
        c = np.array(o['center'], dtype=float)[:3]
        s_arr = np.asarray(o['sigma'], dtype=float)
        s = float(s_arr.mean())
        centers.append(c)
        sigmas.append(s)
    return np.array(centers), np.array(sigmas)


summary = []

for seed in SEEDS:
    obstacles = make_obstacles(seed=seed, n=8)
    out = run_hierarchical(START, GOAL, obstacles, dt=0.01, T_max=10.0)
    traj = np.array(out['xs'])
    positions = traj[:, :3]
    obstacle_centers, obstacle_sigmas = parse_obstacles(obstacles)

    # field value at every step
    field_per_step = np.array(
        [obstacle_field_value(positions[i], obstacles)
         for i in range(len(positions))]
    )

    # also compute: closest distance to any obstacle center,
    # and the "effective sigma units" of that closest approach
    diffs = positions[:, None, :] - obstacle_centers[None, :, :]
    eucl_dists = np.linalg.norm(diffs, axis=2)  # (N, M)
    # for each step, find the closest obstacle (in sigma-normalized units)
    sigma_dists = eucl_dists / obstacle_sigmas[None, :]  # (N, M)
    min_sigma_per_step = sigma_dists.min(axis=1)
    min_euclid_per_step = eucl_dists.min(axis=1)

    summary.append({
        'seed': seed,
        'max_field': float(field_per_step.max()),
        'min_sigma_dist': float(min_sigma_per_step.min()),
        'min_euclid_mm': float(min_euclid_per_step.min() * 1000),
        'n_above_threshold': int((field_per_step > PLANNER_THRESHOLD).sum()),
        'n_near_collision': int((field_per_step > NEAR_COLLISION).sum()),
        'field_per_step': field_per_step,
        'min_sigma_per_step': min_sigma_per_step,
        'min_euclid_per_step': min_euclid_per_step,
        'positions': positions,
        'obstacle_centers': obstacle_centers,
        'obstacle_sigmas': obstacle_sigmas,
    })

print(f"\n{'seed':>5} | {'max field':>10} | {'min σ-dist':>11} | "
      f"{'min m-dist':>11} | {'field>0.3':>10} | {'field>0.6':>10}")
print("-" * 85)
for r in summary:
    status = "  SAFE" if r['n_above_threshold'] == 0 else "  WARN"
    if r['n_near_collision'] > 0:
        status = "  COLLISION"
    print(f"{r['seed']:>5} | {r['max_field']:>10.4f} | "
          f"{r['min_sigma_dist']:>8.2f}σ  | "
          f"{r['min_euclid_mm']:>8.1f}mm | "
          f"{r['n_above_threshold']:>10d} | "
          f"{r['n_near_collision']:>10d}{status}")

# pick the seed with min σ-distance for the plot
worst = min(summary, key=lambda r: r['min_sigma_dist'])
print(f"\nworst seed (closest approach): {worst['seed']}")
print(f"  closest approach: {worst['min_sigma_dist']:.2f}σ = "
      f"{worst['min_euclid_mm']:.1f}mm from obstacle center")
print(f"  this corresponds to field value = "
      f"{np.exp(-worst['min_sigma_dist']**2 / 2):.4f}")
print(f"  planner threshold is 0.3, so this is "
      f"{'OK' if worst['max_field'] < 0.3 else 'OVER threshold'}")

# plot worst seed
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
positions = worst['positions']
ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1.5)
ax1.scatter(*START, c='green', s=100, marker='o', label='start')
ax1.scatter(*GOAL, c='red', s=200, marker='*', label='goal')

# draw 1.5σ surfaces (where field = 0.3, the planner's threshold)
u = np.linspace(0, 2*np.pi, 18)
v = np.linspace(0, np.pi, 18)
for c, s in zip(worst['obstacle_centers'], worst['obstacle_sigmas']):
    r = 1.5 * s  # the planner's actual safety surface
    sx = c[0] + r * np.outer(np.cos(u), np.sin(v))
    sy = c[1] + r * np.outer(np.sin(u), np.sin(v))
    sz = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax1.plot_wireframe(sx, sy, sz, color='red', alpha=0.3, linewidth=0.5)

ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.set_title(f"seed {worst['seed']}: trajectory + planner's 1.5σ safety surface")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(1, 3, 2)
t = np.arange(len(worst['field_per_step'])) * 0.01
ax2.plot(t, worst['field_per_step'], 'b-', linewidth=1)
ax2.axhline(PLANNER_THRESHOLD, color='red', linestyle='--',
            label=f'planner threshold ({PLANNER_THRESHOLD})')
ax2.axhline(NEAR_COLLISION, color='darkred', linestyle=':',
            label=f'near-collision ({NEAR_COLLISION})')
ax2.set_xlabel('time (s)'); ax2.set_ylabel('Gaussian field value')
ax2.set_title(f"seed {worst['seed']}: obstacle field over time")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 0.8)

ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(t, worst['min_sigma_per_step'], 'b-', linewidth=1)
ax3.axhline(1.5, color='red', linestyle='--', label='1.5σ (field=0.3)')
ax3.axhline(1.0, color='darkred', linestyle=':', label='1σ (field=0.6)')
ax3.set_xlabel('time (s)'); ax3.set_ylabel('distance to nearest obstacle (σ units)')
ax3.set_title(f"seed {worst['seed']}: σ-normalized clearance")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/clearance_check.png', dpi=120)
print('\nsaved plot to results/clearance_check.png')