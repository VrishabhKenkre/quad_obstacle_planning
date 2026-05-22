"""
hybrid/render_hybrid_videos.py -- render the hybrid controller's
verification videos.

Renders single-drone hybrid rollouts in MuJoCo with a 6-line overlay
(controller, seed, t, goal err, max field, tracking err), plus a
planner-vs-hybrid side-by-side. Frames extracted at t=5s and t=8s.

Videos:
  videos/hybrid_random_seed42.mp4
  videos/hybrid_dp_seed2.mp4
  videos/hybrid_dp_seed6.mp4
  videos/planner_vs_hybrid_dp_seed2.mp4   (hstack)

Usage:
  python -m hybrid.render_hybrid_videos
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'distillation'))
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))

from quad_env import CrazyflieEnv
from obstacle_course import make_obstacles, obstacle_field_value
from randomize_astar import decision_point_layout, randomized_astar_paths

from hybrid.planner_runner import PlannerRunner
from hybrid.mlp_tracker import MLPTracker
from hybrid.hybrid_ctrl import START, GOAL

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size, _cache={}):
    if size not in _cache:
        try:
            _cache[size] = ImageFont.truetype(_FONT, size)
        except OSError:
            _cache[size] = ImageFont.load_default()
    return _cache[size]


def _overlay(img, lines, max_field):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    white, black, red = (255, 255, 255), (0, 0, 0), (255, 90, 90)
    draw.text((20, 16), lines[0], fill=white, font=_font(23),
              stroke_width=2, stroke_fill=black)
    draw.text((20, 45), lines[1], fill=(220, 220, 220), font=_font(17),
              stroke_width=2, stroke_fill=black)
    y = 70
    for ln in lines[2:]:
        col = red if (ln.startswith('max field') and max_field >= 0.30) else white
        draw.text((20, y), ln, fill=col, font=_font(19),
                  stroke_width=2, stroke_fill=black)
        y += 23
    return np.array(pil)


def _add(scn, gtype, size, pos, rgba):
    if scn.ngeom >= scn.maxgeom - 1:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=gtype,
                        size=np.array(size, dtype=np.float64),
                        pos=np.asarray(pos, dtype=np.float64),
                        mat=np.eye(3).flatten(),
                        rgba=np.array(rgba, dtype=np.float32))
    scn.ngeom += 1


def _build_dp_paths(seed, safety_margin=0.30):
    obstacles, lb, rb = decision_point_layout(seed=int(seed))
    from voxelize import VoxelMap
    vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
    paths = randomized_astar_paths(
        START, GOAL, vm, k=2, safety_margin=safety_margin,
        length_ratio_max=1.30, forced_bias_pairs=[lb, rb],
        z_penalty_per_m=0.4, seed=20_000 + int(seed))
    return obstacles, paths


def render_hybrid(tracker, seed, seed_type, out_path,
                  dt_ctrl=0.005, dt_sim=0.001, duration=12.0, fps=30,
                  width=1280, height=720):
    """Render a hybrid rollout: plan once, track, draw, overlay."""
    if seed_type == 'random':
        obstacles = make_obstacles(seed=int(seed))
        safety_margin = 0.15
        planner_paths = []
    else:
        obstacles, planner_paths = _build_dp_paths(seed)
        safety_margin = 0.30

    planner = PlannerRunner(START, GOAL, obstacles,
                            safety_margin=safety_margin,
                            ref_dt=dt_ctrl, avg_speed=0.8)

    scene = str(_ROOT / 'mujoco_menagerie' / 'bitcraze_crazyflie_2'
                / 'scene.xml')
    env = CrazyflieEnv(model_path=scene, dt_sim=dt_sim, dt_ctrl=dt_ctrl)
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.distance = 4.2; cam.azimuth = 120; cam.elevation = -28
    cam.lookat[:] = [0.0, 0.0, 1.0]
    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0

    ref = planner.ref
    state_mj = env.reset(pos=START)
    n_steps = int(duration / dt_ctrl)
    frame_every = max(1, int(round((1.0 / fps) / dt_ctrl)))
    trail = []
    trail_every = max(1, int(round(0.05 / dt_ctrl)))
    max_field = 0.0
    frames, frame_t = [], []

    def populate(scn):
        for o in obstacles:
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                 [float(o['sigma'][0]), 0, 0], o['center'],
                 [0.9, 0.2, 0.2, 0.35])
        # planner reference polyline (cyan)
        step_s = max(1, ref.shape[1] // 120)
        for k in range(0, ref.shape[1], step_s):
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                 ref[0:3, k], [0.3, 0.85, 0.95, 0.8])
        for p in planner_paths:
            ss = max(1, len(p) // 60)
            for k in range(0, len(p), ss):
                _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.014, 0, 0],
                     p[k], [0.55, 0.75, 1.0, 0.5])
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0],
             [-1.5, -1.5, 1.0], [0.2, 0.5, 1.0, 0.95])
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0],
             [1.5, 1.5, 1.0], [1.0, 0.85, 0.1, 0.95])
        for i, p in enumerate(trail):
            a = 0.15 + 0.55 * (i / max(len(trail) - 1, 1))
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.010, 0, 0], p,
                 [1.0, 0.55, 0.15, a])

    for k in range(n_steps):
        ref_col = planner.query(k)
        a = tracker.predict(state_mj, ref_col)
        u = np.clip(u_mid + u_half * a, env.u_min, env.u_max)
        state_mj = env.step(u)
        max_field = max(max_field,
                        float(obstacle_field_value(state_mj[0:3], obstacles)))
        track_err_mm = float(np.linalg.norm(
            ref_col[0:3] - state_mj[0:3]) * 1000.0)
        if k % trail_every == 0:
            trail.append(state_mj[0:3].copy())
            if len(trail) > 240:
                trail.pop(0)
        if k % frame_every == 0:
            renderer.update_scene(env.data, cam)
            populate(renderer.scene)
            img = renderer.render()
            goal_cm = float(np.linalg.norm(state_mj[0:3] - GOAL) * 100)
            lines = [
                "Hybrid (planner 5Hz + MLP tracker 200Hz)",
                f"{seed_type} seed {seed}",
                f"t = {k*dt_ctrl:5.2f} s",
                f"goal err: {goal_cm:5.1f} cm",
                f"max field: {max_field:.3f}",
                f"tracking err: {track_err_mm:5.1f} mm",
            ]
            frames.append(_overlay(img, lines, max_field))
            frame_t.append(k * dt_ctrl)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, codec='libx264',
                    quality=7)
    ft = np.asarray(frame_t)
    for t_target, suf in ((5.0, 't05'), (8.0, 't08')):
        idx = int(np.argmin(np.abs(ft - t_target)))
        Image.fromarray(frames[idx]).save(
            out_path.with_name(out_path.stem + f'_{suf}.png'))
    print(f"  -> {out_path} ({out_path.stat().st_size/1024**2:.2f} MB)")
    return out_path


def hstack(in_paths, out_path, panel_width=640, height=720):
    fc = "".join(
        f"[{i}:v]scale={panel_width}:{height}:force_original_aspect_ratio=disable[v{i}];"
        for i in range(len(in_paths)))
    fc += "".join(f"[v{i}]" for i in range(len(in_paths)))
    fc += f"hstack=inputs={len(in_paths)}[out]"
    cmd = ["ffmpeg", "-y"]
    for p in in_paths:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", fc, "-map", "[out]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  -> {out_path} ({Path(out_path).stat().st_size/1024**2:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', type=str, default='data/mlp_tracker_v1.pt')
    ap.add_argument('--dt-ctrl', type=float, default=0.005)
    ap.add_argument('--dt-sim', type=float, default=0.001)
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    tracker = MLPTracker(args.tracker, device=args.device)
    vid = _ROOT / 'videos'

    targets = [
        ('hybrid_random_seed42', 42, 'random'),
        ('hybrid_dp_seed2', 2, 'dp'),
        ('hybrid_dp_seed6', 6, 'dp'),
    ]
    for name, seed, stype in targets:
        print(f"[render] {name}")
        t0 = time.time()
        render_hybrid(tracker, seed, stype, vid / f'{name}.mp4',
                      dt_ctrl=args.dt_ctrl, dt_sim=args.dt_sim)
        print(f"  ({time.time()-t0:.1f}s)")

    # planner-vs-hybrid side-by-side on dp seed 2:
    # reuse the existing hybrid dp_seed2 panel; render a planner panel
    # via the standard planner demo recorder.
    print("[render] planner_vs_hybrid_dp_seed2 (side-by-side)")
    tmp = vid / '_pvh_tmp'
    tmp.mkdir(exist_ok=True)
    # hybrid panel (640 wide)
    render_hybrid(tracker, 2, 'dp', tmp / 'hybrid.mp4',
                  dt_ctrl=args.dt_ctrl, dt_sim=args.dt_sim,
                  width=640, height=720)
    # planner panel: render the planner tracking the same dp layout
    _render_planner_panel(2, tmp / 'planner.mp4', width=640, height=720)
    hstack([tmp / 'planner.mp4', tmp / 'hybrid.mp4'],
           vid / 'planner_vs_hybrid_dp_seed2.mp4',
           panel_width=640, height=720)
    # frames
    for t_target, suf in ((5.0, 't05'), (8.0, 't08')):
        subprocess.run(["ffmpeg", "-ss", str(t_target), "-i",
                        str(vid / 'planner_vs_hybrid_dp_seed2.mp4'),
                        "-frames:v", "1", "-y",
                        str(vid / f'planner_vs_hybrid_dp_seed2_{suf}.png')],
                       check=True, capture_output=True)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def _render_planner_panel(seed, out_path, width=640, height=720,
                          duration=12.0, fps=30):
    """Render the hierarchical planner (A*+min-snap+NMPC) on a dp seed."""
    from nonlinear_mpc import SE3_NMPC, rotors_to_mujoco
    from hierarchical_ctrl import (plan_once, _take_ref_window,
                                    _mujoco_state_to_nmpc_state)
    obstacles, planner_paths = _build_dp_paths(seed)
    ref, _meta = plan_once(START, GOAL, obstacles, safety_margin=0.30,
                           ref_dt=0.02, avg_speed=0.8)
    nmpc = SE3_NMPC(N=15, dt=0.02, obstacles=obstacles,
                    q_pos=300, q_vel=10, q_quat=20, q_omega=0.1,
                    r_thrust=1e3, w_obs=800.0)
    scene = str(_ROOT / 'mujoco_menagerie' / 'bitcraze_crazyflie_2'
                / 'scene.xml')
    env = CrazyflieEnv(model_path=scene, dt_sim=0.002, dt_ctrl=0.02)
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.distance = 4.2; cam.azimuth = 120; cam.elevation = -28
    cam.lookat[:] = [0.0, 0.0, 1.0]
    x_mj = env.reset(pos=START)
    n_steps = int(duration / 0.02)
    frame_every = max(1, int(round((1.0 / fps) / 0.02)))
    trail = []
    max_field = 0.0
    frames = []

    def populate(scn):
        for o in obstacles:
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                 [float(o['sigma'][0]), 0, 0], o['center'],
                 [0.9, 0.2, 0.2, 0.35])
        step_s = max(1, ref.shape[1] // 120)
        for k in range(0, ref.shape[1], step_s):
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                 ref[0:3, k], [0.3, 0.85, 0.4, 0.8])
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0],
             [-1.5, -1.5, 1.0], [0.2, 0.5, 1.0, 0.95])
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0],
             [1.5, 1.5, 1.0], [1.0, 0.85, 0.1, 0.95])
        for i, p in enumerate(trail):
            a = 0.15 + 0.55 * (i / max(len(trail) - 1, 1))
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.010, 0, 0], p,
                 [0.2, 0.9, 1.0, a])

    for k in range(min(n_steps, ref.shape[1])):
        rp_win, rv_win = _take_ref_window(ref, k, 15)
        x13 = _mujoco_state_to_nmpc_state(x_mj)
        u_rot, _ = nmpc.solve(x13, rp_win, rv_win)
        x_mj = env.step(rotors_to_mujoco(u_rot))
        max_field = max(max_field,
                        float(obstacle_field_value(x_mj[0:3], obstacles)))
        if k % max(1, int(round(0.05 / 0.02))) == 0:
            trail.append(x_mj[0:3].copy())
            if len(trail) > 200:
                trail.pop(0)
        if k % frame_every == 0:
            renderer.update_scene(env.data, cam)
            populate(renderer.scene)
            img = renderer.render()
            goal_cm = float(np.linalg.norm(x_mj[0:3] - GOAL) * 100)
            lines = ["Hierarchical planner (A*+min-snap+NMPC)",
                     f"dp seed {seed}",
                     f"t = {k*0.02:5.2f} s",
                     f"goal err: {goal_cm:5.1f} cm",
                     f"max field: {max_field:.3f}"]
            frames.append(_overlay(img, lines, max_field))
    imageio.mimsave(str(out_path), frames, fps=fps, codec='libx264',
                    quality=7)
    print(f"  -> {out_path} (planner panel)")


if __name__ == '__main__':
    main()
