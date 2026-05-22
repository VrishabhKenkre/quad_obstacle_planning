"""
render_phase3_audit_videos.py -- render single-drone videos of selected
PPO phase-3 iter15 rollouts (cleanest 1/2/3, cleanest dp, ugliest) as
identified by ppo_phase3_visual_audit.py.

Live diffusion rollout (K=1, deterministic torch.manual_seed(1234+seed))
in MuJoCo, with the brief's exact 6-line overlay and the
red-if-max_field>0.60 rule (distinct from record_student_demo's
>=1.0 rule -- the audit's PASS criterion is "never red", i.e. < 0.60).

Usage:
  python distillation/render_phase3_audit_videos.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
from diffusion_student import (build_diffusion_policy, make_inference_scheduler,
                                build_normalizer_from_arrays, DEFAULT_HORIZON)

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
RED_THRESHOLD = 0.60   # max_field overlay turns red above this


def _font(size, _cache={}):
    if size not in _cache:
        try:
            _cache[size] = ImageFont.truetype(_FONT, size)
        except OSError:
            _cache[size] = ImageFont.load_default()
    return _cache[size]


def burn_overlay(img, seed, seed_type, t_now, goal_err_cm,
                 max_field, cleanness):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    white, black = (255, 255, 255), (0, 0, 0)
    red = (255, 80, 80)
    draw.text((20, 18), "PPO phase3 iter15", fill=white, font=_font(26),
              stroke_width=2, stroke_fill=black)
    draw.text((20, 50), f"seed {seed} ({seed_type})", fill=(220, 220, 220),
              font=_font(18), stroke_width=2, stroke_fill=black)
    y = 78
    draw.text((20, y), f"t = {t_now:5.2f} s", fill=white,
              font=_font(20), stroke_width=2, stroke_fill=black)
    draw.text((20, y + 24), f"goal err: {goal_err_cm:5.1f} cm", fill=white,
              font=_font(20), stroke_width=2, stroke_fill=black)
    mf_color = red if max_field > RED_THRESHOLD else white
    draw.text((20, y + 48), f"max field: {max_field:.3f}", fill=mf_color,
              font=_font(20), stroke_width=2, stroke_fill=black)
    draw.text((20, y + 72), f"cleanness: {cleanness:+.2f}", fill=white,
              font=_font(20), stroke_width=2, stroke_fill=black)
    return np.array(pil)


def load_policy(model_path, data_path, device):
    data = np.load(_ROOT / data_path)
    norm = build_normalizer_from_arrays(data['observations'], data['actions'])
    policy = build_diffusion_policy(
        obs_dim=OBS_DIM, action_dim=ACT_DIM, horizon=DEFAULT_HORIZON,
        n_obs_steps=1, n_action_steps=DEFAULT_HORIZON,
        down_dims=(128, 256, 512), diffusion_step_embed_dim=128,
        num_inference_steps=8)
    policy.set_normalizer(norm)
    policy.load_state_dict(torch.load(_ROOT / model_path, map_location='cpu'))
    policy = policy.to(device).eval()
    policy.noise_scheduler = make_inference_scheduler(num_inference_steps=8)
    policy.num_inference_steps = 8
    return policy


def build_layout(seed, seed_type, safety_margin=0.30):
    if seed_type == 'random':
        return make_obstacles(seed=int(seed)), []
    obstacles, lb, rb = decision_point_layout(seed=int(seed))
    vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
    paths = randomized_astar_paths(
        START, GOAL, vm, k=2, safety_margin=safety_margin,
        length_ratio_max=1.30, forced_bias_pairs=[lb, rb],
        z_penalty_per_m=0.4, seed=20_000 + int(seed))
    return obstacles, paths


def render_one(policy, seed, seed_type, cleanness, out_path, device,
               duration=12.0, fps=30, width=1280, height=720,
               dt_ctrl=0.02):
    obstacles, planner_paths = build_layout(seed, seed_type)
    vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()

    scene = str(_ROOT / 'mujoco_menagerie' / 'bitcraze_crazyflie_2'
                / 'scene.xml')
    env = CrazyflieEnv(model_path=scene, dt_sim=0.002, dt_ctrl=dt_ctrl)
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.distance = 4.2; cam.azimuth = 120; cam.elevation = -28
    cam.lookat[:] = [0.0, 0.0, 1.0]

    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0

    torch.manual_seed(1234 + int(seed))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(1234 + int(seed))

    state_mj = env.reset(pos=START)
    n_steps = int(duration / dt_ctrl)
    frame_every = max(1, int(round((1.0 / fps) / dt_ctrl)))
    trail = []
    trail_every = max(1, int(round(0.05 / dt_ctrl)))
    max_field = 0.0
    frames, frame_times = [], []

    def _add(scn, gtype, size, pos, rgba):
        if scn.ngeom >= scn.maxgeom - 1:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=gtype,
                            size=np.array(size, dtype=np.float64),
                            pos=np.asarray(pos, dtype=np.float64),
                            mat=np.eye(3).flatten(),
                            rgba=np.array(rgba, dtype=np.float32))
        scn.ngeom += 1

    def populate(scn):
        for obs in obstacles:
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                 [float(obs['sigma'][0]), 0, 0], obs['center'],
                 [0.9, 0.2, 0.2, 0.35])
        for path in planner_paths:
            step_s = max(1, len(path) // 80)
            for k in range(0, len(path), step_s):
                _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0, 0],
                     path[k], [0.3, 0.85, 0.95, 0.85])
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0],
             [-1.5, -1.5, 1.0], [0.2, 0.5, 1.0, 0.95])
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0],
             [1.5, 1.5, 1.0], [1.0, 0.85, 0.1, 0.95])
        for i, p in enumerate(trail):
            a = 0.15 + 0.55 * (i / max(len(trail) - 1, 1))
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.010, 0, 0], p,
                 [0.2, 0.9, 1.0, a])

    policy.eval()
    for k in range(n_steps):
        obs = make_observation(state_mj, GOAL, vm)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device).reshape(1, 1, OBS_DIM)
            a = policy.predict_action({'obs': obs_t})['action'][0, 0].cpu().numpy()
        u = np.clip(u_mid + u_half * a, env.u_min, env.u_max)
        state_mj = env.step(u)
        max_field = max(max_field,
                        float(obstacle_field_value(state_mj[0:3], obstacles)))
        if k % trail_every == 0:
            trail.append(state_mj[0:3].copy())
            if len(trail) > 240:
                trail.pop(0)
        if k % frame_every == 0:
            renderer.update_scene(env.data, cam)
            populate(renderer.scene)
            img = renderer.render()
            goal_err_cm = float(np.linalg.norm(state_mj[0:3] - GOAL) * 100)
            img = burn_overlay(img, seed, seed_type, k * dt_ctrl,
                               goal_err_cm, max_field, cleanness)
            frames.append(img)
            frame_times.append(k * dt_ctrl)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, codec='libx264',
                    quality=7)
    # frame extraction at t=5s and t=8s
    ft = np.asarray(frame_times)
    for t_target, suffix in ((5.0, 't05'), (8.0, 't08')):
        idx = int(np.argmin(np.abs(ft - t_target)))
        Image.fromarray(frames[idx]).save(
            out_path.with_name(out_path.stem + f'_{suffix}.png'))
    size_mb = out_path.stat().st_size / 1024**2
    print(f"  -> {out_path} ({size_mb:.2f} MB, {len(frames)} frames)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', type=str,
                    default='results/ppo_phase3_visual_audit.json')
    ap.add_argument('--model', type=str,
                    default='data/diffusion_v2_ppo_phase3_iter15.pt')
    ap.add_argument('--data', type=str,
                    default='data/planner_dataset_v2.npz')
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    audit = json.load(open(_ROOT / args.audit))
    ranked = audit['ranked']
    dp_ranked = [r for r in ranked if r['seed_type'] == 'dp']

    targets = [
        ('ppo_phase3_cleanest_1', ranked[0]),
        ('ppo_phase3_cleanest_2', ranked[1]),
        ('ppo_phase3_cleanest_3', ranked[2]),
        ('ppo_phase3_cleanest_dp', dp_ranked[0]),
        ('ppo_phase3_ugliest', ranked[-1]),
    ]
    print("[render] targets:")
    for name, r in targets:
        print(f"  {name}: seed {r['seed']} ({r['seed_type']}) "
              f"cleanness={r['cleanness_score']:+.3f}")

    policy = load_policy(args.model, args.data, device)
    print(f"[render] loaded {args.model}\n")

    for name, r in targets:
        print(f"[render] {name} -- seed {r['seed']} ({r['seed_type']})")
        t0 = time.time()
        render_one(policy, r['seed'], r['seed_type'],
                   r['cleanness_score'],
                   _ROOT / 'videos' / f'{name}.mp4', device)
        print(f"  ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
