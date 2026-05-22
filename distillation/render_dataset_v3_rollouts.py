"""
render_dataset_v3_rollouts.py -- replay-render the worst surviving / p95 /
p90 / p75 / median rollouts (and best-rejected) from the v3 curation, to
visually verify the filter criteria.

This does NOT re-run the planner -- it sets MuJoCo qpos directly to the
position sequence already stored in planner_dataset_v2.npz. The drone
body, obstacles, planner reference path (if present), start/goal markers
are drawn the same way as distillation/record_student_demo.py.

For each rollout:
  - render an mp4 (1280x720, 30 fps, libx264)
  - extract a frame at t=5s and at the end of the trajectory
  - overlay percentile label, seed/variant, t, goal err, max_field,
    efficiency, overshoot

Outputs to videos/v3_*.mp4 + v3_*_t05.png + v3_*_tEnd.png.

Usage:
  python distillation/render_dataset_v3_rollouts.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))

from voxelize import VoxelMap
from obstacle_course import make_obstacles, obstacle_field_value
from collect_planner_data import GOAL, START
from randomize_astar import decision_point_layout, randomized_astar_paths


_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size, _cache={}):
    if size not in _cache:
        try:
            _cache[size] = ImageFont.truetype(_FONT, size)
        except OSError:
            _cache[size] = ImageFont.load_default()
    return _cache[size]


def burn_overlay(img, lines: list, max_field_now: float):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    white, black = (255, 255, 255), (0, 0, 0)
    red = (255, 110, 80)
    y = 18
    draw.text((20, y), lines[0], fill=white, font=_font(26),
              stroke_width=2, stroke_fill=black)
    y += 32
    if len(lines) > 1:
        draw.text((20, y), lines[1], fill=(220, 220, 220),
                  font=_font(18), stroke_width=2, stroke_fill=black)
        y += 26
    for ln in lines[2:]:
        is_field = ln.startswith('max field') and max_field_now >= 1.0
        color = red if is_field else white
        draw.text((20, y), ln, fill=color, font=_font(20),
                  stroke_width=2, stroke_fill=black)
        y += 24
    return np.array(pil)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) Euler to wxyz quaternion."""
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    q = r.as_quat()   # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _build_obstacles(seed: int, is_dp: bool,
                      safety_margin: float = 0.30):
    if is_dp:
        obstacles, lb, rb = decision_point_layout(seed=int(seed))
        vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
        paths = randomized_astar_paths(
            START, GOAL, vm, k=2, safety_margin=safety_margin,
            length_ratio_max=1.30,
            forced_bias_pairs=[lb, rb], z_penalty_per_m=0.4,
            seed=20_000 + int(seed))
        return obstacles, paths
    else:
        return make_obstacles(seed=int(seed)), []


def render_rollout(positions: np.ndarray, eulers: np.ndarray,
                   obstacles, planner_paths: list,
                   label_line1: str, label_line2: str,
                   metrics: dict,
                   out_path: str,
                   target_duration: float = 12.0, fps: int = 30,
                   width: int = 1280, height: int = 720,
                   dt_ctrl: float = 0.02):
    """positions: (T, 3); eulers: (T, 3) roll/pitch/yaw or None."""
    scene = str(_ROOT / 'mujoco_menagerie' / 'bitcraze_crazyflie_2'
                / 'scene.xml')
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.distance = 4.2
    cam.azimuth = 120
    cam.elevation = -28
    cam.lookat[:] = [0.0, 0.0, 1.0]

    n = positions.shape[0]
    # Stride frames so the video lands close to target_duration.
    target_n = int(target_duration * fps)
    stride = max(1, n // target_n)

    drone_trail = []
    trail_every = max(1, int(round(0.05 / (stride * dt_ctrl))))
    trail_max = 200

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
        for path in planner_paths:
            step_s = max(1, len(path) // 80)
            for k in range(0, len(path), step_s):
                _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                     [0.015, 0, 0], path[k],
                     [0.3, 0.85, 0.95, 0.85])
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0],
             [-1.5, -1.5, 1.0], [0.2, 0.5, 1.0, 0.95])
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0],
             [1.5, 1.5, 1.0], [1.0, 0.85, 0.1, 0.95])
        for i, p in enumerate(drone_trail):
            alpha = 0.15 + 0.55 * (i / max(len(drone_trail) - 1, 1))
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.010, 0, 0], p,
                 [0.2, 0.9, 1.0, alpha])

    # max-field running tally
    max_field_running = 0.0
    frames = []
    frame_indices = []   # so we can locate t=5s and tEnd frames after
    for k in range(0, n, stride):
        # Set drone pose
        data.qpos[0:3] = positions[k]
        if eulers is not None:
            data.qpos[3:7] = euler_to_quat(*eulers[k])
        else:
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        # Update trail
        if (k // stride) % trail_every == 0:
            drone_trail.append(positions[k].copy())
            if len(drone_trail) > trail_max:
                drone_trail.pop(0)

        # Compute per-step diagnostics
        field_now = obstacle_field_value(positions[k], obstacles)
        max_field_running = max(max_field_running, float(field_now))
        t_now = k * dt_ctrl
        goal_err_cm = float(np.linalg.norm(positions[k]
                                            - np.asarray(GOAL)) * 100.0)

        # Render
        renderer.update_scene(data, cam)
        populate_scene(renderer.scene)
        img = renderer.render()

        # Overlay
        lines = [
            label_line1,
            label_line2,
            f"t = {t_now:5.2f} s",
            f"goal err: {goal_err_cm:5.1f} cm",
            f"max field: {max_field_running:.3f}",
            f"efficiency: {metrics['efficiency']:.2f}",
            f"overshoot: {metrics['overshoot_amount']*100:.1f} cm",
        ]
        img = burn_overlay(img, lines, max_field_now=max_field_running)
        frames.append(img)
        frame_indices.append(k)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  rendering {len(frames)} frames @ {fps} fps -> {out_path}")
    imageio.mimsave(str(out_path), frames, fps=fps, codec='libx264',
                    quality=7)

    # Frame extraction (t=5s + final)
    # We have `frames` list aligned with sim time `k * dt_ctrl` for each
    # frame's render index. Find the frame whose t is closest to 5s.
    sim_times = np.asarray([fi * dt_ctrl for fi in frame_indices])
    if sim_times.size:
        idx_5 = int(np.argmin(np.abs(sim_times - 5.0)))
        f5 = frames[idx_5]
        Image.fromarray(f5).save(
            out_path.with_name(out_path.stem + '_t05.png'))
        idx_end = len(frames) - 1
        Image.fromarray(frames[idx_end]).save(
            out_path.with_name(out_path.stem + '_tEnd.png'))
    return out_path


def select_six(audit: dict, surviving_keys: set,
               sorted_all: list) -> dict:
    """Pick worst, p95, p90, p75, median surviving + best_rejected.

    surviving_keys: set of (seed, variant_id) tuples that pass the
    final filter.
    sorted_all: all rollouts sorted by ugliness descending.
    """
    surviving = [r for r in sorted_all
                  if (int(r['seed']), int(r['variant_id'])) in surviving_keys]
    rejected = [r for r in sorted_all
                 if (int(r['seed']), int(r['variant_id'])) not in surviving_keys]
    n_s = len(surviving)
    if n_s == 0:
        raise SystemExit("No surviving rollouts to render.")
    # surviving is sorted descending by ugliness:
    # index 0 = worst, index n-1 = best (cleanest).
    def _pct_rank(p):
        # p = 5 means 5th-percentile WORST (close to top of surviving).
        return min(n_s - 1, int(round(p / 100.0 * (n_s - 1))))
    picks = dict(
        worst=surviving[0],
        p95=surviving[_pct_rank(5)],
        p90=surviving[_pct_rank(10)],
        p75=surviving[_pct_rank(25)],
        median=surviving[_pct_rank(50)],
    )
    # de-duplicate: if any two picks share (seed, variant_id), shift the
    # later one by +1
    seen = set()
    for label in ('worst', 'p95', 'p90', 'p75', 'median'):
        key = (int(picks[label]['seed']), int(picks[label]['variant_id']))
        if key in seen:
            # find next-distinct survivor after this index
            idx = surviving.index(picks[label])
            for j in range(idx + 1, n_s):
                k2 = (int(surviving[j]['seed']),
                      int(surviving[j]['variant_id']))
                if k2 not in seen:
                    picks[label] = surviving[j]
                    key = k2
                    break
        seen.add(key)
    # best rejected: rejected list is sorted descending by ugliness too;
    # so the LAST item in rejected is the lowest-ugliness rejected one.
    if rejected:
        picks['best_rejected'] = rejected[-1]
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', type=str,
                    default='data/planner_dataset_v2_rollout_audit.json')
    ap.add_argument('--stats', type=str,
                    default='data/planner_dataset_v3_stats.json')
    ap.add_argument('--data', type=str,
                    default='data/planner_dataset_v2.npz')
    ap.add_argument('--out-json', type=str,
                    default='results/dataset_v3_worst_rollouts.json')
    ap.add_argument('--video-dir', type=str, default='videos')
    ap.add_argument('--duration', type=float, default=12.0)
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()

    audit = json.load(open(_ROOT / args.audit))
    stats = json.load(open(_ROOT / args.stats))
    crit = stats['criteria_used']

    sorted_all = audit['rollouts']   # descending ugliness
    # Re-derive which rollouts pass the final criteria
    def _passes(r):
        return (r['final_goal_err_mm'] < crit['final_goal_err_mm_max']
                and r['max_field_along_rollout'] < crit['max_field_max']
                and r['terminal_speed_m_s'] < crit['terminal_speed_max']
                and r['efficiency'] > crit['efficiency_min']
                and r['overshoot_amount'] < crit['overshoot_max'])
    surviving_keys = set(
        (int(r['seed']), int(r['variant_id']))
        for r in sorted_all if _passes(r))
    print(f"[render] {len(surviving_keys)} surviving rollouts; "
          f"{len(sorted_all) - len(surviving_keys)} rejected")

    picks = select_six(audit, surviving_keys, sorted_all)

    # Save the selection JSON
    out_data = dict(
        n_surviving_rollouts=int(len(surviving_keys)),
        n_filtered_rollouts=int(len(sorted_all) - len(surviving_keys)),
        criteria_used=crit,
        relaxations_applied=stats.get('relaxations_applied', []),
    )
    for label, r in picks.items():
        out_data[label] = dict(
            seed=int(r['seed']),
            variant_id=int(r['variant_id']),
            variant=r['variant'],
            is_decision_pt=bool(r['is_decision_pt']),
            ugliness_score=float(r['ugliness_score']),
            metrics=dict(
                final_goal_err_mm=float(r['final_goal_err_mm']),
                max_field_along_rollout=float(r['max_field_along_rollout']),
                efficiency=float(r['efficiency']),
                terminal_speed_m_s=float(r['terminal_speed_m_s']),
                overshoot_amount=float(r['overshoot_amount']),
                path_length=float(r['path_length']),
                n_steps=int(r['n_steps']),
            ),
        )
    Path(_ROOT / args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(_ROOT / args.out_json, 'w'), indent=2)
    print(f"[render] selection -> {args.out_json}")

    # ---- Now render each ----
    print(f"\n[render] loading v2 sample arrays for replay")
    data = np.load(_ROOT / args.data)
    obs = data['observations']
    seeds = data['seeds']
    variant_ids = data['variant_ids']
    is_dp = data['is_decision_pt']
    step_indices = data['step_indices']
    goal = np.asarray(GOAL, dtype=np.float64)

    LABELS_DESC = dict(worst='WORST SURVIVING', p95='p95 SURVIVING',
                       p90='p90 SURVIVING', p75='p75 SURVIVING',
                       median='MEDIAN SURVIVING',
                       best_rejected='BEST REJECTED')

    for label, r in picks.items():
        seed = int(r['seed']); var = int(r['variant_id'])
        dp = bool(r['is_decision_pt'])
        mask = (seeds == seed) & (variant_ids == var)
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(step_indices[idx])]
        obs_r = obs[idx]
        positions = obs_r[:, 0:3].astype(np.float64) + goal[None, :]
        eulers = obs_r[:, 6:9].astype(np.float64)

        obstacles, planner_paths = _build_obstacles(seed, dp)

        label_line1 = LABELS_DESC[label]
        var_label = r['variant']
        label_line2 = f"seed {seed} variant {var_label}"
        if label == 'best_rejected':
            out_path = (_ROOT / args.video_dir / 'v3_best_rejected.mp4')
        else:
            out_path = (_ROOT / args.video_dir / f'v3_{label}_rollout.mp4')

        print(f"\n[render] === {label} ===  seed={seed} variant={var_label}  "
              f"ugliness={r['ugliness_score']:.3f}  "
              f"goal={r['final_goal_err_mm']:.0f}mm  "
              f"max_field={r['max_field_along_rollout']:.3f}  "
              f"eff={r['efficiency']:.3f}  "
              f"overshoot={r['overshoot_amount']*100:.1f}cm")
        render_rollout(positions, eulers, obstacles, planner_paths,
                       label_line1, label_line2,
                       metrics=r,
                       out_path=str(out_path),
                       target_duration=args.duration,
                       fps=args.fps)
        sz = out_path.stat().st_size / 1024**2
        print(f"  -> {out_path} ({sz:.2f} MB)")


if __name__ == '__main__':
    main()
