"""
render_multimodal_video.py -- side-by-side video showing what the multi-modal
collapse looks like.

Left panel  : MLP-BC trajectory (red trail) on a decision-point seed.
Right panel : Diffusion-BC trajectory (blue trail) on the same seed.

Both panels render the obstacle field as a heatmap in xy (the decision-
point obstacle is a tall column so a top-down projection is informative).
Frames are written with imageio at 30 fps; resolution 1280x720.

Trajectories are read from results/decision_point_eval.npz which was
populated by distillation/eval_decision_points.py.

Usage:
    python distillation/render_multimodal_video.py --seed 6 \
        --output videos/mlp_vs_diffusion_multimodal.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / 'src'))
from obstacle_course import obstacle_field_value  # noqa: E402

START = np.array([-1.5, -1.5, 1.0])
GOAL  = np.array([+1.5, +1.5, 1.0])


def gaussian_xy_heatmap(obstacles, nx=120, ny=120, bounds=((-2, 2), (-2, 2)),
                        z=1.0):
    """Eval the obstacle field on a 2D grid at fixed z, for display."""
    xs = np.linspace(bounds[0][0], bounds[0][1], nx)
    ys = np.linspace(bounds[1][0], bounds[1][1], ny)
    grid = np.zeros((ny, nx))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            grid[j, i] = obstacle_field_value(np.array([x, y, z]), obstacles)
    return xs, ys, grid


def render(seed: int, output: str, fps: int = 30,
           dt_ctrl: float = 0.02,
           data_path: str = 'results/decision_point_eval.npz',
           summary_path: str = 'results/decision_point_eval.json',
           dpi: int = 110):
    data = np.load(_ROOT / data_path, allow_pickle=True)
    summary = json.load(open(_ROOT / summary_path))
    pr = summary['per_seed_summary'][str(seed)]

    mlp_xs = data[f'seed_{seed}_MLP_BC_xs']
    diff_xs = data[f'seed_{seed}_Diffusion_BC_xs']
    obs_arr = data[f'seed_{seed}_obstacles']
    obstacles = [dict(center=list(map(float, c)),
                      sigma=list(map(float, s)),
                      weight=float(w)) for (c, s, w) in obs_arr]
    # Planner reference paths for context.
    planner_paths = []
    i = 0
    while f'seed_{seed}_planner_path_{i}' in data.files:
        planner_paths.append(data[f'seed_{seed}_planner_path_{i}'])
        i += 1
    print(f"[render] seed {seed}: MLP_BC traj {mlp_xs.shape}, "
          f"Diffusion traj {diff_xs.shape}, {len(obstacles)} obstacles, "
          f"{len(planner_paths)} planner alternatives")

    # 2D heatmap of obstacle field
    xs_g, ys_g, grid = gaussian_xy_heatmap(obstacles, nx=140, ny=140)

    # Frame count = trajectory length
    n_frames = max(len(mlp_xs), len(diff_xs))
    print(f"[render] {n_frames} frames at {fps} fps")

    out_path = Path(_ROOT / output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264',
                                quality=8, bitrate='2000k', macro_block_size=1)

    fig_w, fig_h = 1280, 720
    fig, axes = plt.subplots(1, 2, figsize=(fig_w/dpi, fig_h/dpi), dpi=dpi)
    titles = ['MLP-BC student (averages two valid plans)',
              'Diffusion-BC student (picks one plan, commits)']
    colors = ['#e74c3c', '#2980b9']
    suptitle = (f"Decision-point seed {seed}: planner has clear left/right "
                f"alternatives.\n"
                f"MLP goal err {pr['MLP_BC']['goal_err_mm']:.0f} mm "
                f"(max field {pr['MLP_BC']['max_field']:.2f}) "
                f" vs Diffusion goal err {pr['Diffusion_BC']['goal_err_mm']:.0f} mm "
                f"(max field {pr['Diffusion_BC']['max_field']:.2f})")

    fig.suptitle(suptitle, fontsize=11, y=0.99)

    # Stride frames so the video isn't insanely long: aim for ~10-12s at 30fps.
    target_n = 12 * fps
    stride = max(1, n_frames // target_n)
    print(f"[render] stride={stride} -> {n_frames // stride} frames "
          f"(~{(n_frames // stride) / fps:.1f} s)")

    try:
        for f_idx in range(0, n_frames, stride):
            for ax, traj, title, col in zip(axes, [mlp_xs, diff_xs], titles,
                                             colors):
                ax.clear()
                ax.imshow(grid, extent=[xs_g[0], xs_g[-1], ys_g[0], ys_g[-1]],
                          origin='lower', cmap='Reds', alpha=0.55, vmin=0,
                          vmax=1.0, aspect='equal')
                # planner alternatives in light grey
                for j, p in enumerate(planner_paths):
                    ax.plot(p[:, 0], p[:, 1], color='0.5', alpha=0.5, lw=1,
                             linestyle='--',
                             label=('planner alternative' if j == 0 else None))
                # past trail
                k = min(f_idx + 1, len(traj))
                ax.plot(traj[:k, 0], traj[:k, 1], color=col, lw=2.5,
                         alpha=0.95)
                # current drone position
                ax.plot(traj[k-1, 0], traj[k-1, 1], 'o', color=col,
                         markersize=10, markeredgecolor='black', zorder=4)
                # start/goal
                ax.plot(START[0], START[1], 'g^', markersize=12,
                         markeredgecolor='black', label='start')
                ax.plot(GOAL[0], GOAL[1], 'b*', markersize=16,
                         markeredgecolor='black', label='goal')
                ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
                ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
                ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
                t = f_idx * dt_ctrl
                ax.set_title(f'{title}\nt = {t:4.2f} s', fontsize=10)
                if f_idx == 0:
                    ax.legend(loc='lower right', fontsize=7)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(buf)
        # Hold the last frame for ~1 s
        for _ in range(fps):
            writer.append_data(buf)
    finally:
        writer.close()
        plt.close(fig)

    size_mb = out_path.stat().st_size / 1024**2
    print(f"[render] -> {out_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--output', type=str,
                        default='videos/mlp_vs_diffusion_multimodal.mp4')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    render(args.seed, args.output, fps=args.fps)


if __name__ == '__main__':
    main()
