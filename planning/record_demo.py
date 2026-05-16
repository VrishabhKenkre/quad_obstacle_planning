"""
record_demo.py -- record a single hierarchical-planner traversal as mp4.

Plans, smooths, then runs the NMPC tracker while rendering MuJoCo offscreen.
Burns a title overlay on each frame. The companion CSV (solve times per
step) is dropped next to the video for inspection.

Usage:
    python planning/record_demo.py --seed 99 --output videos/planner_demo.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / 'src'))

from voxelize import VoxelMap
from astar import astar_3d, voxel_path_to_world, prune_path
from min_snap import smooth_waypoints
from nonlinear_mpc import SE3_NMPC, rotors_to_mujoco
from quad_env import CrazyflieEnv
from obstacle_course import make_obstacles, obstacle_field_value
from hierarchical_ctrl import _mujoco_state_to_nmpc_state, _take_ref_window


_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size, _cache={}):
    if size not in _cache:
        try:
            _cache[size] = ImageFont.truetype(_FONT_PATH, size)
        except OSError:
            _cache[size] = ImageFont.load_default()
    return _cache[size]


def burn_overlay(img, title, t_now, goal_err_cm, max_field):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    white, black = (255, 255, 255), (0, 0, 0)
    draw.text((20, 18), title, fill=white, font=_font(26),
              stroke_width=2, stroke_fill=black)
    draw.text((20, 52), f"t = {t_now:5.2f} s", fill=white, font=_font(20),
              stroke_width=2, stroke_fill=black)
    draw.text((20, 78), f"goal err: {goal_err_cm:5.1f} cm", fill=white,
              font=_font(20), stroke_width=2, stroke_fill=black)
    draw.text((20, 104), f"max field: {max_field:.3f}", fill=white,
              font=_font(20), stroke_width=2, stroke_fill=black)
    return np.array(pil)


def render_run(seed: int, out_path: str, duration: float = 8.0, fps: int = 30,
               width: int = 1280, height: int = 720,
               overlay_text: str = "Hierarchical Planner: A* + min-snap + NMPC"):
    obstacles = make_obstacles(seed=seed)
    start = np.array([-1.5, -1.5, 1.0])
    goal = np.array([1.5, 1.5, 1.0])

    # ---- Plan once ----
    vm = VoxelMap()
    vm.from_obstacle_field(obstacles)
    vm.compute_esdf()
    path = astar_3d(start, goal, vm, safety_margin=0.15)
    if path is None:
        raise RuntimeError(f"A* failed on seed {seed}")
    pw = voxel_path_to_world(path, vm)
    pw = prune_path(pw, vm, safety_margin=0.15)
    pw[0] = start; pw[-1] = goal
    nmpc_dt = 0.02
    ref, meta = smooth_waypoints(pw, target_dt=nmpc_dt, target_avg_speed=0.8,
                                  return_meta=True)
    print(f"[plan] seed={seed} | {len(pw)} waypoints | smooth {meta['solver']}"
          f" | ref total {meta['total_time']:.2f} s")

    # ---- NMPC tracker ----
    nmpc = SE3_NMPC(N=15, dt=nmpc_dt, obstacles=obstacles,
                    q_pos=300, q_vel=10, q_quat=20, q_omega=0.1,
                    r_thrust=1e3, w_obs=800.0)

    # ---- MuJoCo offscreen renderer ----
    scene = str(_HERE.parent / "mujoco_menagerie" / "bitcraze_crazyflie_2"
                / "scene.xml")
    env = CrazyflieEnv(model_path=scene, dt_sim=0.002, dt_ctrl=nmpc_dt)
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.distance = 4.2
    cam.azimuth = 120
    cam.elevation = -28
    cam.lookat[:] = [0.0, 0.0, 1.0]

    # Add obstacles, reference polyline, start/goal, and drone trail by
    # APPENDING to the renderer's scene buffer. Note: we must NOT reset
    # ngeom to zero here -- update_scene() already populated the model
    # geoms (drone body, floor, lights) and we draw on top of them.
    drone_trail = []  # populated by render_run; we draw the trail markers

    def _add(scn, geom_type, size, pos, rgba, mat=None):
        if scn.ngeom >= scn.maxgeom - 1:
            return
        if mat is None:
            mat = np.eye(3).flatten()
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=geom_type,
            size=np.array(size, dtype=np.float64),
            pos=np.asarray(pos, dtype=np.float64),
            mat=np.asarray(mat, dtype=np.float64).flatten(),
            rgba=np.array(rgba, dtype=np.float32))
        scn.ngeom += 1

    def populate_scene(scn):
        # Obstacles (red, semi-transparent).
        for obs in obstacles:
            c = obs['center']
            s = float(obs['sigma'][0])
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [s, 0, 0], c,
                 [0.9, 0.2, 0.2, 0.35])
        # Reference trajectory polyline.
        step = max(1, ref.shape[1] // 80)
        for k in range(0, ref.shape[1], step):
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                 [0.018, 0, 0], ref[0:3, k],
                 [0.3, 0.85, 0.4, 0.85])
        # Start (blue) and goal (yellow) markers.
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0], start,
             [0.2, 0.5, 1.0, 0.95])
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0], goal,
             [1.0, 0.85, 0.1, 0.95])
        # Executed trail (cyan, fading).
        n_trail = len(drone_trail)
        for i, p in enumerate(drone_trail):
            alpha = 0.15 + 0.55 * (i / max(n_trail - 1, 1))
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.010, 0, 0], p,
                 [0.2, 0.9, 1.0, alpha])

    x_mj = env.reset(pos=start)
    n_steps = int(duration / nmpc_dt)
    frame_every = max(1, int(round((1.0 / fps) / nmpc_dt)))
    frames = []
    solve_times = []
    max_field = 0.0
    trail_every = max(1, int(round(0.05 / nmpc_dt)))  # one trail dot per 50 ms
    trail_max = 200

    for k in range(n_steps):
        rp_win, rv_win = _take_ref_window(ref, k, 15)
        x13 = _mujoco_state_to_nmpc_state(x_mj)
        t0 = time.perf_counter()
        u_rot, info = nmpc.solve(x13, rp_win, rv_win)
        solve_times.append(time.perf_counter() - t0)
        u_mj = rotors_to_mujoco(u_rot)
        x_mj = env.step(u_mj)
        fld = obstacle_field_value(x_mj[0:3], obstacles)
        max_field = max(max_field, fld)

        if k % trail_every == 0:
            drone_trail.append(x_mj[0:3].copy())
            if len(drone_trail) > trail_max:
                drone_trail.pop(0)

        if k % frame_every == 0:
            renderer.update_scene(env.data, cam)
            # Append obstacles, reference polyline, start/goal, and trail
            # on TOP of the existing model geoms (drone body + floor + lights).
            populate_scene(renderer.scene)
            img = renderer.render()
            goal_err_cm = float(np.linalg.norm(x_mj[0:3] - goal) * 100)
            img = burn_overlay(img, overlay_text, t_now=k * nmpc_dt,
                                goal_err_cm=goal_err_cm, max_field=max_field)
            frames.append(img)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix("").as_posix() + "_solvetimes.csv"
    with open(csv_path, "w") as f:
        f.write("step_index,solve_time_s\n")
        for i, t in enumerate(solve_times):
            f.write(f"{i},{t:.9f}\n")

    print(f"[record] {len(frames)} frames @ {fps} fps -> {out_path}; csv -> {csv_path}")
    imageio.mimsave(str(out_path), frames, fps=fps, codec="libx264", quality=7)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=256)
    ap.add_argument("--output", default="videos/planner_demo.mp4")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    render_run(seed=args.seed, out_path=args.output, duration=args.duration,
               fps=args.fps, width=args.width, height=args.height)
