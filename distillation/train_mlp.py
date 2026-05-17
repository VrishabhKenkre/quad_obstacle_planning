"""
train_mlp.py -- thin orchestrator: run dagger_planner and then evaluate the
final student on the canonical 10-seed obstacle course, dumping
results/mlp_distill_10seed.json in the same schema as
results/planner_10seed.json.

Usage:
    python distillation/train_mlp.py                 # full pipeline
    python distillation/train_mlp.py --eval-only     # skip training,
                                                       evaluate data/mlp_student_v1.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / 'planning'))
sys.path.insert(0, str(_HERE.parent / 'src'))

from voxelize import VoxelMap
from quad_env import CrazyflieEnv
from obstacle_course import make_obstacles, obstacle_field_value

from collect_planner_data import make_observation, OBS_DIM, ACT_DIM, START, GOAL
from mlp_student import MLPStudent
from dagger_planner import run_dagger


SEEDS = [42, 7, 13, 99, 256, 128, 314, 2024, 777, 1337]


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _sem(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std() / np.sqrt(len(arr)))


def run_student_seed(model: torch.nn.Module, seed: int,
                     device: torch.device,
                     dt_ctrl: float = 0.02,
                     T_max: float = 10.0) -> dict:
    """Roll out the learned policy on `seed`, no expert in the loop."""
    obstacles = make_obstacles(seed=seed)
    vm = VoxelMap()
    vm.from_obstacle_field(obstacles)
    vm.compute_esdf()
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=dt_ctrl)
    state_mj = env.reset(pos=START)
    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0

    n_steps = int(T_max / dt_ctrl)
    xs = [state_mj.copy()]
    field_vals = [obstacle_field_value(state_mj[0:3], obstacles)]
    inf_times = []
    model.eval()
    for i in range(n_steps):
        obs = make_observation(state_mj, GOAL, vm)
        t0 = time.perf_counter()
        with torch.no_grad():
            inp = torch.from_numpy(obs).to(device).unsqueeze(0)
            a = model(inp).cpu().numpy()[0]
        inf_times.append(time.perf_counter() - t0)
        u = u_mid + u_half * a
        u = np.clip(u, env.u_min, env.u_max)
        state_mj = env.step(u)
        xs.append(state_mj.copy())
        field_vals.append(obstacle_field_value(state_mj[0:3], obstacles))

    xs = np.asarray(xs)
    field_vals = np.asarray(field_vals)
    final_pos = xs[-1, 0:3]
    goal_err_mm = float(np.linalg.norm(final_pos - GOAL) * 1000)
    path_len_m = float(np.sum(np.linalg.norm(np.diff(xs[:, 0:3], axis=0),
                                              axis=1)))
    duration_s = n_steps * dt_ctrl
    return dict(
        goal_err_mm=goal_err_mm,
        max_field=float(np.max(field_vals)),
        mean_field=float(np.mean(field_vals)),
        path_len_m=path_len_m,
        mean_speed_mps=path_len_m / max(duration_s, 1e-6),
        median_inference_us=float(np.median(inf_times) * 1e6),
        mean_inference_us=float(np.mean(inf_times) * 1e6),
        n_steps=int(n_steps),
    )


def run_10seed_eval(model_path: str, save_path: str,
                    device: torch.device | None = None,
                    verbose: bool = True) -> dict:
    if device is None:
        device = torch.device('cpu')  # small MLP -> CPU is fastest
    model = MLPStudent().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    rows = []
    detail = {}
    for s in SEEDS:
        if verbose:
            print(f"[mlp eval] seed {s}")
        r = run_student_seed(model, s, device)
        rows.append(r)
        detail[str(s)] = r
        if verbose:
            print(f"  goal_err={r['goal_err_mm']:.0f} mm, "
                  f"max_field={r['max_field']:.3f}, "
                  f"speed={r['mean_speed_mps']:.2f} m/s, "
                  f"inference={r['median_inference_us']:.0f} us")

    keys = ['goal_err_mm', 'max_field', 'mean_field', 'path_len_m',
            'mean_speed_mps', 'median_inference_us', 'mean_inference_us']
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows]
        m, sd = _mean_std(vals)
        agg[k] = dict(mean=m, std=sd, sem=_sem(vals), values=vals)

    canonical = dict(
        goal_err_mm=agg['goal_err_mm']['mean'],
        goal_std=agg['goal_err_mm']['std'],
        max_field=agg['max_field']['mean'],
        field_std=agg['max_field']['std'],
        teacher_dev_mm=None,
        dev_std=None,
        path_len_m=agg['path_len_m']['mean'],
        path_std=agg['path_len_m']['std'],
        speed_us=agg['median_inference_us']['mean'],
    )
    out = dict(
        metadata=dict(
            n_seeds=len(SEEDS),
            obstacle_seeds=SEEDS,
            task='8-obstacle Gaussian course, start=(-1.5,-1.5,1) goal=(1.5,1.5,1)',
            controller='MLP_DAgger_DART_planner',
            obs_dim=OBS_DIM,
            obs_design='state(12) + sdf(1) + sdf_grad(3) + 8*sdf_lookahead',
            net='2-layer ReLU MLP, 64 hidden, tanh output',
            model_path=str(model_path),
        ),
        obstacle_course=dict(MLP_DAgger_DART_planner=canonical),
        aggregate=agg,
        per_seed=detail,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    if verbose:
        print(f"\n[mlp eval] -> {save_path}")
        print(f"  mean goal_err: {canonical['goal_err_mm']:.0f} +/- "
              f"{canonical['goal_std']:.0f} mm")
        print(f"  mean max_field: {canonical['max_field']:.3f}")
        print(f"  mean inference: {canonical['speed_us']:.0f} us")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--model', type=str, default='data/mlp_student_v1.pt')
    parser.add_argument('--out',   type=str,
                        default='results/mlp_distill_10seed.json')
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--eps-per-iter', type=int, default=40)
    parser.add_argument('--bc-epochs', type=int, default=80)
    parser.add_argument('--dagger-epochs', type=int, default=40)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if not args.eval_only:
        run_dagger(n_iters=args.iters,
                   episodes_per_iter=args.eps_per_iter,
                   bc_epochs=args.bc_epochs,
                   dagger_epochs=args.dagger_epochs,
                   device=args.device,
                   out_model=args.model)
    run_10seed_eval(model_path=str(_HERE.parent / args.model),
                    save_path=str(_HERE.parent / args.out))


if __name__ == '__main__':
    main()
