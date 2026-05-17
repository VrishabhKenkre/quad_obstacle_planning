"""
record_video_gallery.py -- record the three new portfolio videos for the
distillation paper. Re-uses record_student_demo.render_run for the
single-controller videos and stacks the three-up panel with ffmpeg.

Videos:
  videos/mlp_bc_unsafe.mp4
        MLP-BC on the random course seed with highest max_field --
        narrative: low goal err hides the obstacle violation.
  videos/diffusion_clean_dp.mp4
        Diffusion v2 (K from k_sweep) on the dp seed with lowest combined
        goal_err + max_field.
  videos/all_three_side_by_side.mp4
        MLP-BC | MLP-DAgger | Diffusion on dp seed 6 (the seed used by
        the existing mlp_vs_diffusion_multimodal.mp4 to keep narrative
        continuity).

Usage:
  python distillation/record_video_gallery.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from record_student_demo import render_run


def pick_mlp_bc_unsafe_seed(json_path: Path) -> int:
    """Highest max_field seed in mlp_bc_only_10seed.json."""
    with open(json_path) as f:
        d = json.load(f)
    per_seed = d['per_seed']
    seed = max(per_seed.items(), key=lambda kv: kv[1]['max_field'])
    print(f"  mlp_bc unsafe pick: seed={seed[0]} "
          f"goal_err={seed[1]['goal_err_mm']:.0f}mm "
          f"max_field={seed[1]['max_field']:.3f}")
    return int(seed[0])


def pick_diffusion_clean_dp_seed(json_path: Path) -> int:
    """Lowest combined (goal_err_mm + 1000*max_field) on Diffusion_BC
    rollouts in decision_point_eval_v2_multi3.json."""
    with open(json_path) as f:
        d = json.load(f)
    per_seed = d['per_seed_summary']
    best = None; best_score = float('inf')
    for s, rows in per_seed.items():
        r = rows['Diffusion_BC']
        score = r['goal_err_mm'] + 1000 * r['max_field']
        if score < best_score:
            best_score = score
            best = (int(s), r)
    print(f"  diffusion clean dp pick: seed={best[0]} "
          f"goal_err={best[1]['goal_err_mm']:.0f}mm "
          f"max_field={best[1]['max_field']:.3f}")
    return best[0]


def hstack_videos(in_paths, out_path, height=720, panel_width=640):
    """Use ffmpeg to scale each input to panel_width x height and hstack."""
    panel_scale = f"scale={panel_width}:{height}:force_original_aspect_ratio=disable"
    filter_complex = ""
    for i in range(len(in_paths)):
        filter_complex += f"[{i}:v]{panel_scale}[v{i}];"
    filter_complex += "".join(f"[v{i}]" for i in range(len(in_paths)))
    filter_complex += f"hstack=inputs={len(in_paths)}[out]"
    cmd = ["ffmpeg", "-y"]
    for p in in_paths:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            str(out_path)]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def verify_frame(video_path: Path, frame_t: float = 5.0) -> Path:
    """Extract a frame at frame_t seconds for visual verification."""
    frame_path = video_path.with_name(video_path.stem + '_frame.png')
    subprocess.run([
        "ffmpeg", "-ss", str(frame_t), "-i", str(video_path),
        "-frames:v", "1", "-y", str(frame_path)
    ], check=True, capture_output=True)
    print(f"  verify frame: {frame_path}")
    return frame_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=None,
                    help='Diffusion multi-sample K; defaults to '
                         'recommended_K from k_sweep result JSON if present, '
                         'else K=3')
    ap.add_argument('--video-dir', type=str, default='videos')
    ap.add_argument('--ksweep-json', type=str,
                    default='results/decision_point_eval_v2_kSweep.json')
    ap.add_argument('--side-by-side-seed', type=int, default=6)
    ap.add_argument('--duration', type=float, default=12.0)
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()

    # Decide K
    K = args.K
    if K is None:
        ks_path = _ROOT / args.ksweep_json
        if ks_path.exists():
            with open(ks_path) as f:
                ks = json.load(f)
            K = int(ks.get('recommended_K', 3))
            print(f"[gallery] using recommended K = {K} from {ks_path.name}")
        else:
            K = 3
            print(f"[gallery] {ks_path} not found, defaulting K = 3")

    video_dir = _ROOT / args.video_dir
    video_dir.mkdir(parents=True, exist_ok=True)

    # === Video 1: mlp_bc_unsafe.mp4 ===========================================
    print("\n[gallery] === Video 1: mlp_bc_unsafe.mp4 ===")
    seed_mlpbc = pick_mlp_bc_unsafe_seed(
        _ROOT / 'results/mlp_bc_only_10seed.json')
    out1 = video_dir / 'mlp_bc_unsafe.mp4'
    render_run(student='mlp_bc', layout='random', seed=seed_mlpbc,
               model_path='data/mlp_student_bc_only.pt',
               data_path='data/planner_dataset_v2.npz',
               out_path=str(out1), K=1,
               duration=args.duration, fps=args.fps,
               width=1280, height=720,
               overlay_text='MLP-BC student',
               overlay_subtitle=f'random seed {seed_mlpbc} (low goal err, '
                                 f'crashes through obstacle)',
               show_planner_paths=False)
    verify_frame(out1, frame_t=5.0)

    # === Video 2: diffusion_clean_dp.mp4 =====================================
    print("\n[gallery] === Video 2: diffusion_clean_dp.mp4 ===")
    seed_diff = pick_diffusion_clean_dp_seed(
        _ROOT / 'results/decision_point_eval_v2_multi3.json')
    out2 = video_dir / 'diffusion_clean_dp.mp4'
    render_run(student='diffusion', layout='dp', seed=seed_diff,
               model_path='data/diffusion_student_v2_ema.pt',
               data_path='data/planner_dataset_v2.npz',
               out_path=str(out2), K=K,
               duration=args.duration, fps=args.fps,
               width=1280, height=720,
               overlay_text=f'Diffusion v2 (K={K})',
               overlay_subtitle=f'decision-point seed {seed_diff} '
                                 f'(low max_field, clean arrival)',
               show_planner_paths=True)
    verify_frame(out2, frame_t=5.0)

    # === Video 3: all_three_side_by_side.mp4 =================================
    print("\n[gallery] === Video 3: all_three_side_by_side.mp4 ===")
    seed_sbs = args.side_by_side_seed
    print(f"  side-by-side seed: {seed_sbs}")
    tmp_dir = video_dir / '_sbs_tmp'
    tmp_dir.mkdir(exist_ok=True)
    panel_paths = []
    for student, mp, K_, sub in [
        ('mlp_bc', 'data/mlp_student_bc_only.pt', 1, 'MLP-BC (averaged)'),
        ('mlp_dagger', 'data/mlp_student_v1.pt', 1, 'MLP DAgger+DART'),
        ('diffusion', 'data/diffusion_student_v2_ema.pt', K,
         f'Diffusion v2 K={K}'),
    ]:
        p = tmp_dir / f'panel_{student}.mp4'
        render_run(student=student, layout='dp', seed=seed_sbs,
                   model_path=mp, data_path='data/planner_dataset_v2.npz',
                   out_path=str(p), K=K_,
                   duration=args.duration, fps=args.fps,
                   width=720, height=720,
                   overlay_text=sub,
                   overlay_subtitle=f'dp seed {seed_sbs}',
                   show_planner_paths=True)
        panel_paths.append(p)
    out3 = video_dir / 'all_three_side_by_side.mp4'
    hstack_videos(panel_paths, out3, height=720, panel_width=720)
    verify_frame(out3, frame_t=5.0)
    # Tidy
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    print("\n[gallery] done.")
    for p in [out1, out2, out3]:
        size_mb = p.stat().st_size / 1024**2
        print(f"  {p}: {size_mb:.2f} MB")


if __name__ == '__main__':
    main()
