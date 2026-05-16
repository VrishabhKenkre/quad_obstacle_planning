#!/usr/bin/env python3
"""Verify the planner never lets the drone touch obstacles."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')   # so obstacle_course can find its own imports

from planning.hierarchical_ctrl import run_hierarchical
from obstacle_course import make_obstacles, obstacle_field_value

SEEDS = [42, 7, 13, 99, 256, 128, 314, 2024, 777, 1337]
START = np.array([-1.5, -1.5, 1.0])
GOAL = np.array([1.5, 1.5, 1.0])
FIELD_THRESHOLD = 0.3
MIN_SAFE_DIST_M = 0.10


def parse_obstacles(obstacles):
    """Centers (M, 3) and isotropic sigmas (M,) from the dict format."""
    centers = []
    sigmas = []
    for o in obstacles:
        c = np.array(o['center'], dtype=float)[:3]
        s_raw = o['sigma']
        # sigma is stored as a 3-element list per axis, but always isotropic
        if isinstance(s_raw, (list, tuple, np.ndarray)):
            s_arr = np.asarray(s_raw, dtype=float)
            # verify isotropy (warn if not)
            if not np.allclose(s_arr, s_arr[0]):
                print(f"  WARNING: anisotropic sigma {s_arr}, using mean")
            s = float(s_arr.mean())
        else:
            s = float(s_raw)
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

    assert obstacle_centers.shape == (len(obstacles), 3)
    assert obstacle_sigmas.shape == (len(obstacles),)

    diffs = positions[:, None, :] - obstacle_centers[None, :, :]
    eucl_dists = np.linalg.norm(diffs, axis=2)
    surf_dists = eucl_dists - 3.0 * obstacle_sigmas[None, :]
    min_surf_dist_per_step = surf_dists.min(axis=1)

    field_per_step = np.array(
        [obstacle_field_value(positions[i], obstacles)
         for i in range(len(positions))]
    )

    n_inside_surface = int((min_surf_dist_per_step < 0).sum())
    n_close = int(((min_surf_dist_per_step >= 0) &
                   (min_surf_dist_per_step < MIN_SAFE_DIST_M)).sum())
    n_above_field = int((field_per_step > FIELD_THRESHOLD).sum())

    summary.append({
        'seed': seed,
        'min_surf_dist_mm': float(min_surf_dist_per_step.min() * 1000),
        'max_field': float(field_per_step.max()),
        'n_inside_surface': n_inside_surface,
        'n_close_under_10cm': n_close,
        'n_above_field_threshold': n_above_field,
        'positions': positions,
        'obstacle_centers': obstacle_centers,
        'obstacle_sigmas': obstacle_sigmas,
        'min_surf_dist_per_step': min_surf_dist_per_step,
        'field_per_step': field_per_step,
    })

print(f"\n{'seed':>5} | {'min surf':>10} | {'max field':>10} | "
      f"{'inside_surf':>12} | {'<10cm':>6} | {'field>0.3':>10}")
print("-" * 75)
for r in summary:
    flag = "  CLIPS" if r['n_inside_surface'] > 0 else "  clean"
    print(f"{r['seed']:>5} | {r['min_surf_dist_mm']:>7.1f}mm | "
          f"{r['max_field']:>10.4f} | {r['n_inside_surface']:>12d} | "
          f"{r['n_close_under_10cm']:>6d} | {r['n_above_field_threshold']:>10d}{flag}")

worst = min(summary, key=lambda r: r['min_surf_dist_mm'])
print(f"\nworst seed: {worst['seed']} (min surface dist {worst['min_surf_dist_mm']:.1f} mm)")

# plot worst seed
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
positions = worst['positions']
ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1.5)
ax1.scatter(*START, c='green', s=100, marker='o', label='start')
ax1.scatter(*GOAL, c='red', s=200, marker='*', label='goal')

u = np.linspace(0, 2*np.pi, 18)
v = np.linspace(0, np.pi, 18)
for c, s in zip(worst['obstacle_centers'], worst['obstacle_sigmas']):
    r = 3.0 * s
    sx = c[0] + r * np.outer(np.cos(u), np.sin(v))
    sy = c[1] + r * np.outer(np.sin(u), np.sin(v))
    sz = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax1.plot_wireframe(sx, sy, sz, color='red', alpha=0.2, linewidth=0.5)

ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.set_title(f"seed {worst['seed']}: trajectory + 3σ obstacle spheres")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(1, 3, 2)
t = np.arange(len(worst['min_surf_dist_per_step'])) * 0.01
ax2.plot(t, worst['min_surf_dist_per_step'] * 1000, 'b-', linewidth=1)
ax2.axhline(0, color='red', linestyle='--', label='3σ boundary')
ax2.axhline(100, color='orange', linestyle='--', label='10 cm margin')
ax2.fill_between(t, -100, 0, color='red', alpha=0.15)
ax2.set_xlabel('time (s)'); ax2.set_ylabel('min surface dist (mm)')
ax2.set_title(f"seed {worst['seed']}: distance to nearest obstacle")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(t, worst['field_per_step'], 'b-', linewidth=1)
ax3.axhline(FIELD_THRESHOLD, color='red', linestyle='--', label='field threshold 0.3')
ax3.set_xlabel('time (s)'); ax3.set_ylabel('Gaussian field value')
ax3.set_title(f"seed {worst['seed']}: field value over time")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/no_clipping_check.png', dpi=120)
print('\nsaved plot to results/no_clipping_check.png')