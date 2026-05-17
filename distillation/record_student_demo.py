"""
record_student_demo.py -- record a single learned-student rollout (MLP-BC,
MLP-DAgger, or diffusion-K) in MuJoCo with offscreen rendering. Reuses
record_demo.py's overlay/obstacle/trail drawing so the new videos match
the existing planner_demo.mp4 / mlp_vs_diffusion_multimodal.mp4 visual
language.

Three obstacle layouts are supported via `--layout`:
  random : the random Gaussian course (`make_obstacles(seed=...)`)
  dp     : the engineered decision-point layout

Usage:
  python distillation/record_student_demo.py \\
    --student diffusion --K 3 \\
    --layout dp --seed 6 \\
    --output videos/diffusion_clean_dp.mp4

Side-by-side videos are produced by recording each panel separately and
hstacking with ffmpeg outside of this script.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'external' / 'diffusion_policy'))

from voxelize import VoxelMap
from quad_env import CrazyflieEnv
from obstacle_course import make_obstacles, obstacle_field_value
from collect_planner_data import make_observation, OBS_DIM, ACT_DIM, START, GOAL
from randomize_astar import decision_point_layout, randomized_astar_paths
from mlp_student import MLPStudent
from diffusion_student import (build_diffusion_policy,
                                make_inference_scheduler,
                                build_normalizer_from_arrays,
                                DEFAULT_HORIZON)
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics


_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size, _cache={}):
    if size not in _cache:
        try:
            _cache[size] = ImageFont.truetype(_FONT_PATH, size)
        except OSError:
            _cache[size] = ImageFont.load_default()
    return _cache[size]


def burn_overlay(img, title, subtitle, t_now, goal_err_cm, max_field):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    white, black = (255, 255, 255), (0, 0, 0)
    draw.text((20, 18), title, fill=white, font=_font(26),
              stroke_width=2, stroke_fill=black)
    if subtitle:
        draw.text((20, 50), subtitle, fill=(220, 220, 220),
                  font=_font(18), stroke_width=2, stroke_fill=black)
    base_y = 80 if subtitle else 56
    draw.text((20, base_y), f"t = {t_now:5.2f} s", fill=white,
              font=_font(20), stroke_width=2, stroke_fill=black)
    draw.text((20, base_y + 26), f"goal err: {goal_err_cm:5.1f} cm", fill=white,
              font=_font(20), stroke_width=2, stroke_fill=black)
    color = (255, 110, 80) if max_field >= 1.0 else white
    draw.text((20, base_y + 52), f"max field: {max_field:.3f}", fill=color,
              font=_font(20), stroke_width=2, stroke_fill=black)
    return np.array(pil)


def build_obstacles_and_paths(layout: str, seed: int,
                              safety_margin: float = 0.30):
    if layout == 'random':
        obstacles = make_obstacles(seed=int(seed))
        paths = []  # no planner alternatives for the random course
    elif layout == 'dp':
        obstacles, lb, rb = decision_point_layout(seed=int(seed))
        vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
        paths = randomized_astar_paths(
            START, GOAL, vm, k=2, safety_margin=safety_margin,
            length_ratio_max=1.30,
            forced_bias_pairs=[lb, rb], z_penalty_per_m=0.4,
            seed=20_000 + int(seed))
        if len(paths) < 2:
            extra = randomized_astar_paths(
                START, GOAL, vm, k=2, safety_margin=safety_margin,
                length_ratio_max=1.30, z_penalty_per_m=0.4,
                seed=20_000 + int(seed))
            paths = (paths + extra)[:2]
    else:
        raise ValueError(f"layout must be 'random' or 'dp', got {layout!r}")
    return obstacles, paths


def load_student(student: str, model_path: str, data_path: str,
                 device: torch.device):
    if student == 'mlp_bc' or student == 'mlp_dagger':
        m = MLPStudent()
        m.load_state_dict(torch.load(model_path, map_location='cpu'))
        m = m.to(device).eval()
        return m, None
    if student == 'diffusion':
        data = np.load(_ROOT / data_path)
        norm = build_normalizer_from_arrays(data['observations'],
                                            data['actions'])
        policy = build_diffusion_policy(
            obs_dim=OBS_DIM, action_dim=ACT_DIM,
            horizon=DEFAULT_HORIZON, n_obs_steps=1,
            n_action_steps=DEFAULT_HORIZON,
            down_dims=(128, 256, 512),
            diffusion_step_embed_dim=128,
            num_inference_steps=8,
        )
        policy.set_normalizer(norm)
        policy.load_state_dict(torch.load(_ROOT / model_path,
                                          map_location='cpu'))
        policy = policy.to(device).eval()
        policy.noise_scheduler = make_inference_scheduler(num_inference_steps=8)
        policy.num_inference_steps = 8
        return policy, None
    raise ValueError(f"unknown student: {student}")


_QP = QuadParams()
_Ac, _Bc = linearize_at_hover(_QP)
_Ad_cache: dict = {}
def _ad_bd_for(dt: float):
    if dt not in _Ad_cache:
        _Ad_cache[dt] = discretize_dynamics(_Ac, _Bc, dt)
    return _Ad_cache[dt]


def _predict_traj_linearized(state_mj, action_seq, u_mid, u_half,
                             u_hover, dt):
    Ad, Bd = _ad_bd_for(dt)
    H = action_seq.shape[0]
    out = np.empty((H, 3), dtype=np.float32)
    x = state_mj.copy().astype(np.float64)
    for k in range(H):
        u_phys = u_mid + u_half * action_seq[k]
        x = Ad @ x + Bd @ (u_phys - u_hover)
        out[k] = x[0:3]
    return out


def step_student(student: str, model, vm, device, state_mj, K: int,
                 u_mid, u_half, u_hover, dt_ctrl,
                 diffusion_rng_state=None):
    """One control step. Returns the un-normalised action (4,)."""
    obs = make_observation(state_mj, GOAL, vm)
    if student in ('mlp_bc', 'mlp_dagger'):
        with torch.no_grad():
            inp = torch.from_numpy(obs).to(device).unsqueeze(0)
            a = model(inp).cpu().numpy()[0]
        return a
    # diffusion
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).to(device).reshape(1, 1, OBS_DIM)
        if K and K > 1:
            obs_t = obs_t.expand(K, 1, OBS_DIM).contiguous()
            result = model.predict_action({'obs': obs_t})
            actions_k = result['action'].cpu().numpy()
            best_k = 0; best_score = -float('inf')
            for k in range(K):
                traj = _predict_traj_linearized(
                    state_mj, actions_k[k], u_mid, u_half, u_hover, dt_ctrl)
                sdfs = [vm.query_esdf(p) for p in traj]
                score = float(min(sdfs))
                if score > best_score:
                    best_score = score
                    best_k = k
            return actions_k[best_k, 0]
        else:
            result = model.predict_action({'obs': obs_t})
            return result['action'][0, 0].cpu().numpy()


def render_run(student: str, layout: str, seed: int,
               model_path: str, data_path: str,
               out_path: str,
               K: int = 1,
               duration: float = 12.0, fps: int = 30,
               width: int = 1280, height: int = 720,
               overlay_text: str = "",
               overlay_subtitle: str = "",
               device_str: str = None,
               show_planner_paths: bool = True):
    device = (torch.device(device_str) if device_str else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[record-student] device={device}, student={student}, "
          f"layout={layout}, seed={seed}, K={K}")

    obstacles, paths = build_obstacles_and_paths(layout, seed)
    vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()

    model, _ = load_student(student, model_path, data_path, device)

    scene = str(_ROOT / "mujoco_menagerie" / "bitcraze_crazyflie_2"
                / "scene.xml")
    dt_ctrl = 0.02
    env = CrazyflieEnv(model_path=scene, dt_sim=0.002, dt_ctrl=dt_ctrl)
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.distance = 4.2
    cam.azimuth = 120
    cam.elevation = -28
    cam.lookat[:] = [0.0, 0.0, 1.0]

    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0
    u_hover = np.array([_QP.hover_thrust, 0.0, 0.0, 0.0])

    # diffusion seed for reproducibility
    if student == 'diffusion':
        torch.manual_seed(1234 + int(seed))
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(1234 + int(seed))

    drone_trail = []

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
        for obs in obstacles:
            c = obs['center']
            s = float(obs['sigma'][0])
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [s, 0, 0], c,
                 [0.9, 0.2, 0.2, 0.35])
        if show_planner_paths:
            for path in paths:
                step = max(1, len(path) // 80)
                for k in range(0, len(path), step):
                    _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                         [0.015, 0, 0], path[k],
                         [0.3, 0.85, 0.95, 0.85])
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0],
             [-1.5, -1.5, 1.0], [0.2, 0.5, 1.0, 0.95])
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0],
             [1.5, 1.5, 1.0], [1.0, 0.85, 0.1, 0.95])
        n_trail = len(drone_trail)
        for i, p in enumerate(drone_trail):
            alpha = 0.15 + 0.55 * (i / max(n_trail - 1, 1))
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.010, 0, 0], p,
                 [0.2, 0.9, 1.0, alpha])

    state_mj = env.reset(pos=START)
    n_steps = int(duration / dt_ctrl)
    frame_every = max(1, int(round((1.0 / fps) / dt_ctrl)))
    frames = []
    max_field = 0.0
    trail_every = max(1, int(round(0.05 / dt_ctrl)))
    trail_max = 240

    for k in range(n_steps):
        a = step_student(student, model, vm, device, state_mj, K,
                         u_mid, u_half, u_hover, dt_ctrl)
        u = u_mid + u_half * a
        u = np.clip(u, env.u_min, env.u_max)
        state_mj = env.step(u)
        fld = obstacle_field_value(state_mj[0:3], obstacles)
        max_field = max(max_field, fld)

        if k % trail_every == 0:
            drone_trail.append(state_mj[0:3].copy())
            if len(drone_trail) > trail_max:
                drone_trail.pop(0)

        if k % frame_every == 0:
            renderer.update_scene(env.data, cam)
            populate_scene(renderer.scene)
            img = renderer.render()
            goal_err_cm = float(np.linalg.norm(state_mj[0:3] - GOAL) * 100)
            img = burn_overlay(img, overlay_text, overlay_subtitle,
                               t_now=k * dt_ctrl,
                               goal_err_cm=goal_err_cm, max_field=max_field)
            frames.append(img)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[record-student] {len(frames)} frames @ {fps} fps -> {out_path}")
    imageio.mimsave(str(out_path), frames, fps=fps, codec="libx264",
                    quality=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--student', choices=['mlp_bc', 'mlp_dagger', 'diffusion'],
                    required=True)
    ap.add_argument('--layout', choices=['random', 'dp'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--output', type=str, required=True)
    ap.add_argument('--model', type=str, default=None,
                    help='checkpoint path; auto-selected if omitted')
    ap.add_argument('--data', type=str, default='data/planner_dataset_v2.npz')
    ap.add_argument('--K', type=int, default=1)
    ap.add_argument('--duration', type=float, default=12.0)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--overlay', type=str, default='')
    ap.add_argument('--subtitle', type=str, default='')
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--no-planner-paths', action='store_true')
    args = ap.parse_args()

    if args.model is None:
        if args.student == 'mlp_bc':
            args.model = 'data/mlp_student_bc_only.pt'
        elif args.student == 'mlp_dagger':
            args.model = 'data/mlp_student_v1.pt'
        elif args.student == 'diffusion':
            args.model = 'data/diffusion_student_v2_ema.pt'

    if not args.overlay:
        title = {'mlp_bc': 'MLP-BC student',
                 'mlp_dagger': 'MLP DAgger+DART student',
                 'diffusion': f'Diffusion v2 student (K={args.K})'}[args.student]
        args.overlay = title

    render_run(
        student=args.student, layout=args.layout, seed=args.seed,
        model_path=args.model, data_path=args.data,
        out_path=args.output, K=args.K,
        duration=args.duration, fps=args.fps,
        width=args.width, height=args.height,
        overlay_text=args.overlay, overlay_subtitle=args.subtitle,
        device_str=args.device,
        show_planner_paths=not args.no_planner_paths,
    )


if __name__ == '__main__':
    main()
