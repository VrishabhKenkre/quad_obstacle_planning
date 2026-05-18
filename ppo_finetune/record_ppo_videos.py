"""
ppo_finetune/record_ppo_videos.py -- the 2 PPO videos for the paper.

Video 1: videos/bc_vs_ppo_dp_seed6.mp4
  Side-by-side BC v2 vs PPO iter15 on decision-point seed 6.
  Each panel 640x720 -> hstack 1280x720, 12 sec, 30 fps.

Video 2: videos/ppo_iter_progression.mp4
  5 panels: BC (iter0), PPO iter5, iter10, iter15, T=0.5 iter15 (if
  available). Each panel 256x720 -> hstack 1280x720, 12 sec, 30 fps.

Reuses distillation/record_student_demo.py for individual panels and
ffmpeg for hstacking. Frame verification with ffmpeg -ss 5 ...

Usage:
  python -m ppo_finetune.record_ppo_videos
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'distillation'))

from record_student_demo import render_run


def hstack(in_paths, out_path, panel_width=640, height=720):
    panel_scale = f"scale={panel_width}:{height}:force_original_aspect_ratio=disable"
    fc = "".join(f"[{i}:v]{panel_scale}[v{i}];" for i in range(len(in_paths)))
    fc += "".join(f"[v{i}]" for i in range(len(in_paths)))
    fc += f"hstack=inputs={len(in_paths)}[out]"
    cmd = ["ffmpeg", "-y"]
    for p in in_paths:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", fc, "-map", "[out]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            str(out_path)]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


def verify_frame(video, t=5.0):
    frame = video.with_name(video.stem + '_frame.png')
    subprocess.run(["ffmpeg", "-ss", str(t), "-i", str(video),
                    "-frames:v", "1", "-y", str(frame)],
                   check=True, capture_output=True)
    print(f"  verify -> {frame}")


def make_video1(seed: int = 6, duration: float = 12.0, fps: int = 30):
    """BC vs PPO iter15 on dp seed 6."""
    out = _ROOT / 'videos' / 'bc_vs_ppo_dp_seed6.mp4'
    tmp = _ROOT / 'videos' / '_v1_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    bc_panel = tmp / 'bc.mp4'
    ppo_panel = tmp / 'ppo.mp4'

    render_run(student='diffusion', layout='dp', seed=seed,
               model_path='data/diffusion_student_v2_ema.pt',
               data_path='data/planner_dataset_v2.npz',
               out_path=str(bc_panel), K=1,
               duration=duration, fps=fps,
               width=640, height=720,
               overlay_text='Diffusion BC v2 (K=1)',
               overlay_subtitle=f'dp seed {seed}',
               show_planner_paths=True)
    render_run(student='diffusion', layout='dp', seed=seed,
               model_path='data/diffusion_v2_ppo_phase2_iter15.pt',
               data_path='data/planner_dataset_v2.npz',
               out_path=str(ppo_panel), K=1,
               duration=duration, fps=fps,
               width=640, height=720,
               overlay_text='Diffusion + PPO iter15',
               overlay_subtitle=f'dp seed {seed}',
               show_planner_paths=True)
    hstack([bc_panel, ppo_panel], out, panel_width=640, height=720)
    verify_frame(out, t=5.0)
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass
    print(f"\n[video1] -> {out}  ({out.stat().st_size / 1024**2:.2f} MB)")


def make_video2(seed: int = 6, duration: float = 12.0, fps: int = 30,
                include_t05: bool = True):
    """5-panel iter progression. Falls back to 4 panels if T=0.5 isn't ready."""
    out = _ROOT / 'videos' / 'ppo_iter_progression.mp4'
    tmp = _ROOT / 'videos' / '_v2_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    panels = [
        ('BC iter0', 'data/diffusion_student_v2_ema.pt'),
        ('PPO iter5 (T=0.1)', 'data/diffusion_v2_ppo_phase2_iter5.pt'),
        ('PPO iter10 (T=0.1)', 'data/diffusion_v2_ppo_phase2_iter10.pt'),
        ('PPO iter15 (T=0.1)', 'data/diffusion_v2_ppo_phase2_iter15.pt'),
    ]
    t05_path = _ROOT / 'data' / 'diffusion_v2_ppo_T05_iter15.pt'
    if include_t05 and t05_path.exists():
        panels.append(('PPO iter15 (T=0.5)',
                       'data/diffusion_v2_ppo_T05_iter15.pt'))

    panel_paths = []
    for i, (label, model) in enumerate(panels):
        p = tmp / f'panel_{i}.mp4'
        render_run(student='diffusion', layout='dp', seed=seed,
                   model_path=model,
                   data_path='data/planner_dataset_v2.npz',
                   out_path=str(p), K=1,
                   duration=duration, fps=fps,
                   width=400, height=720,   # fits 5*400 = 2000
                   overlay_text=label,
                   overlay_subtitle=f'dp seed {seed}',
                   show_planner_paths=True)
        panel_paths.append(p)

    panel_w = max(256, 2000 // max(1, len(panel_paths)))
    hstack(panel_paths, out, panel_width=panel_w, height=720)
    verify_frame(out, t=5.0)
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass
    print(f"\n[video2] -> {out}  ({out.stat().st_size / 1024**2:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=6)
    ap.add_argument('--duration', type=float, default=12.0)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--skip-v1', action='store_true')
    ap.add_argument('--skip-v2', action='store_true')
    args = ap.parse_args()
    if not args.skip_v1:
        print("\n[ppo-videos] === Video 1: BC vs PPO ===")
        make_video1(seed=args.seed, duration=args.duration, fps=args.fps)
    if not args.skip_v2:
        print("\n[ppo-videos] === Video 2: iter progression ===")
        make_video2(seed=args.seed, duration=args.duration, fps=args.fps)


if __name__ == '__main__':
    main()
