"""
eval_decision_points.py -- evaluate the three students (MLP-BC, MLP-DAgger,
Diffusion) on the decision-point seeds from Phase 1's decision_point_layout
generator. Saves per-model, per-seed:
  - goal_err_mm, max_field, path_len, mean_speed
  - the full position trajectory xs (N, 3) so Phase 4 can render videos

This is the eval that directly tests the multi-modality story: on
single-obstacle decision-point layouts, MLP averaging should hit the
obstacle whereas diffusion should pick one side and arrive cleanly.

Output: results/decision_point_eval.npz + results/decision_point_eval.json
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
sys.path.insert(0, str(_HERE.parent / 'external' / 'diffusion_policy'))

from voxelize import VoxelMap
from quad_env import CrazyflieEnv
from obstacle_course import obstacle_field_value
from hierarchical_ctrl import run_hierarchical

from collect_planner_data import make_observation, OBS_DIM, ACT_DIM, START, GOAL
from randomize_astar import decision_point_layout, randomized_astar_paths
from mlp_student import MLPStudent
from diffusion_student import (build_diffusion_policy,
                                make_inference_scheduler,
                                build_normalizer_from_arrays,
                                DEFAULT_HORIZON)
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics


DEFAULT_DP_SEEDS = list(range(10))   # the first 10 dp layouts


# ---- Rollout each student ------------------------------------------------

def _new_env(dt_ctrl: float = 0.02):
    return CrazyflieEnv(dt_sim=0.002, dt_ctrl=dt_ctrl)


def rollout_mlp(model: torch.nn.Module, vm: VoxelMap, obstacles: list,
                device: torch.device, dt_ctrl: float = 0.02,
                T_max: float = 10.0) -> dict:
    env = _new_env(dt_ctrl)
    state_mj = env.reset(pos=START)
    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0
    n_steps = int(T_max / dt_ctrl)
    xs = [state_mj.copy()]
    fields = [obstacle_field_value(state_mj[0:3], obstacles)]
    inf = []
    model.eval()
    for i in range(n_steps):
        obs = make_observation(state_mj, GOAL, vm)
        t0 = time.perf_counter()
        with torch.no_grad():
            inp = torch.from_numpy(obs).to(device).unsqueeze(0)
            a = model(inp).cpu().numpy()[0]
        inf.append(time.perf_counter() - t0)
        u = u_mid + u_half * a
        u = np.clip(u, env.u_min, env.u_max)
        state_mj = env.step(u)
        xs.append(state_mj.copy())
        fields.append(obstacle_field_value(state_mj[0:3], obstacles))
    xs = np.asarray(xs); fields = np.asarray(fields)
    return _summarize(xs, fields, inf, dt_ctrl)


# Cache linearized hover dynamics for the multi-sample safety predictor.
_QP = QuadParams()
_Ac, _Bc = linearize_at_hover(_QP)
_Ad_cache: dict = {}


def _ad_bd_for(dt: float):
    if dt not in _Ad_cache:
        _Ad_cache[dt] = discretize_dynamics(_Ac, _Bc, dt)
    return _Ad_cache[dt]


def _predict_traj_linearized(state_mj: np.ndarray, action_seq: np.ndarray,
                             u_mid: np.ndarray, u_half: np.ndarray,
                             u_hover: np.ndarray, dt: float) -> np.ndarray:
    """Predict the (H,3) world-frame position trajectory induced by
    `action_seq` (H, 4 normalised actions) starting from `state_mj`
    (12-D), using the hover-linearised Crazyflie dynamics.

    Cheap and approximate -- good enough as a safety scorer for
    multi-sample diffusion.
    """
    Ad, Bd = _ad_bd_for(dt)
    H = action_seq.shape[0]
    out = np.empty((H, 3), dtype=np.float32)
    x = state_mj.copy().astype(np.float64)
    for k in range(H):
        u_phys = u_mid + u_half * action_seq[k]
        x = Ad @ x + Bd @ (u_phys - u_hover)
        out[k] = x[0:3]
    return out


def rollout_diffusion_multisample(policy, vm: VoxelMap, obstacles: list,
                                  device: torch.device,
                                  dt_ctrl: float = 0.02,
                                  T_max: float = 10.0,
                                  diffusion_seed: int | None = None,
                                  K: int = 3,
                                  env_kwargs: dict | None = None) -> dict:
    """Diffusion rollout with K-sample safety scoring at every control step.

    At each step:
      1. Replicate the obs K times and call `predict_action` -> K different
         action sequences (each shape (H, 4)).
      2. For each sample, forward-predict the resulting 3-D position
         trajectory using the linearised hover dynamics; score the sample
         by `min ESDF along its predicted positions`.
      3. Execute the FIRST action of the highest-scoring sample.
    """
    if diffusion_seed is not None:
        torch.manual_seed(int(diffusion_seed))
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(int(diffusion_seed))
    env = _new_env(dt_ctrl)
    state_mj = env.reset(pos=START)
    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0
    u_hover = np.array([_QP.hover_thrust, 0.0, 0.0, 0.0])
    n_steps = int(T_max / dt_ctrl)
    xs = [state_mj.copy()]
    fields = [obstacle_field_value(state_mj[0:3], obstacles)]
    inf = []
    sample_choice_history = []
    policy.eval()
    for i in range(n_steps):
        obs = make_observation(state_mj, GOAL, vm)
        t0 = time.perf_counter()
        with torch.no_grad():
            # batch K replicas of the same observation
            obs_t = torch.from_numpy(obs).to(device).reshape(1, 1, OBS_DIM)
            obs_t = obs_t.expand(K, 1, OBS_DIM).contiguous()
            result = policy.predict_action({'obs': obs_t})
        # action_pred shape (K, n_action_steps, ACT_DIM)
        actions_k = result['action'].cpu().numpy()
        # Score each sample by min predicted ESDF along its 8-step traj
        best_k = 0
        best_score = -float('inf')
        for k in range(K):
            traj = _predict_traj_linearized(
                state_mj, actions_k[k], u_mid, u_half, u_hover, dt_ctrl)
            sdfs = [vm.query_esdf(p) for p in traj]
            score = float(min(sdfs))
            if score > best_score:
                best_score = score
                best_k = k
        sample_choice_history.append(best_k)
        a = actions_k[best_k, 0]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inf.append(time.perf_counter() - t0)
        u = u_mid + u_half * a
        u = np.clip(u, env.u_min, env.u_max)
        state_mj = env.step(u)
        xs.append(state_mj.copy())
        fields.append(obstacle_field_value(state_mj[0:3], obstacles))
    xs = np.asarray(xs); fields = np.asarray(fields)
    out = _summarize(xs, fields, inf, dt_ctrl)
    # Distribution over which sample was picked, for transparency.
    sample_choice_history = np.asarray(sample_choice_history)
    counts = np.bincount(sample_choice_history, minlength=K)
    out['sample_choice_counts'] = [int(c) for c in counts]
    return out


def rollout_diffusion(policy, vm: VoxelMap, obstacles: list,
                      device: torch.device, dt_ctrl: float = 0.02,
                      T_max: float = 10.0,
                      diffusion_seed: int | None = None) -> dict:
    """Roll out the diffusion student. `diffusion_seed`, if supplied,
    fixes the CUDA RNG so the stochastic DDIM trajectory is repeatable
    across eval runs."""
    if diffusion_seed is not None:
        torch.manual_seed(int(diffusion_seed))
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(int(diffusion_seed))
    env = _new_env(dt_ctrl)
    state_mj = env.reset(pos=START)
    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0
    n_steps = int(T_max / dt_ctrl)
    xs = [state_mj.copy()]
    fields = [obstacle_field_value(state_mj[0:3], obstacles)]
    inf = []
    policy.eval()
    for i in range(n_steps):
        obs = make_observation(state_mj, GOAL, vm)
        t0 = time.perf_counter()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device).reshape(1, 1, OBS_DIM)
            result = policy.predict_action({'obs': obs_t})
        a = result['action'][0, 0].cpu().numpy()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inf.append(time.perf_counter() - t0)
        u = u_mid + u_half * a
        u = np.clip(u, env.u_min, env.u_max)
        state_mj = env.step(u)
        xs.append(state_mj.copy())
        fields.append(obstacle_field_value(state_mj[0:3], obstacles))
    xs = np.asarray(xs); fields = np.asarray(fields)
    return _summarize(xs, fields, inf, dt_ctrl)


def rollout_planner(obstacles: list, T_max: float = 10.0,
                    dt_ctrl: float = 0.02,
                    safety_margin: float = 0.15) -> dict:
    res = run_hierarchical(START, GOAL, obstacles, dt=dt_ctrl,
                           T_max=T_max, return_trajectory=True,
                           safety_margin=safety_margin,
                           verbose=False)
    xs = res['xs']
    fields = res['field_vals']
    # Planner already reports goal_err_mm etc. Repackage with trajectory.
    return dict(
        goal_err_mm=res['goal_err_mm'],
        max_field=res['max_field'],
        mean_field=res['mean_field'],
        path_len_m=res['path_len_m'],
        mean_speed_mps=res['mean_speed_mps'],
        median_inference_us=res['median_solve_ms'] * 1000,
        n_steps=res['n_steps'],
        xs=xs[:, 0:3].astype(np.float32),
    )


def _summarize(xs: np.ndarray, fields: np.ndarray, inf: list,
               dt_ctrl: float) -> dict:
    final_pos = xs[-1, 0:3]
    goal_err_mm = float(np.linalg.norm(final_pos - GOAL) * 1000)
    path_len_m = float(np.sum(np.linalg.norm(np.diff(xs[:, 0:3], axis=0),
                                              axis=1)))
    return dict(
        goal_err_mm=goal_err_mm,
        max_field=float(np.max(fields)),
        mean_field=float(np.mean(fields)),
        path_len_m=path_len_m,
        mean_speed_mps=path_len_m / max(len(inf) * dt_ctrl, 1e-6),
        median_inference_us=float(np.median(inf) * 1e6),
        n_steps=int(len(inf)),
        xs=xs[:, 0:3].astype(np.float32),
    )


# ---- Build planner left/right reference paths for context ----------------

def planner_paths_for_seed(seed: int, safety_margin: float = 0.15) -> tuple:
    """Return (obstacles, left_pruned_path, right_pruned_path) for use as
    overlay context in the saved trajectories (so Phase 4 can show what
    the planner offered).

    `safety_margin` should match what generated the training dataset:
    0.15 for v1 dp seeds, 0.30 for v2.
    """
    obstacles, lb, rb = decision_point_layout(seed=seed)
    vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
    paths = randomized_astar_paths(
        START, GOAL, vm, k=2, safety_margin=safety_margin,
        length_ratio_max=1.30,
        forced_bias_pairs=[lb, rb], z_penalty_per_m=0.4,
        seed=20_000 + seed)
    if len(paths) < 2:
        extra = randomized_astar_paths(
            START, GOAL, vm, k=2, safety_margin=safety_margin,
            length_ratio_max=1.30, z_penalty_per_m=0.4,
            seed=20_000 + seed)
        paths = (paths + extra)[:2]
    return obstacles, vm, paths


# ---- Main ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=DEFAULT_DP_SEEDS)
    parser.add_argument('--mlp-bc', type=str,
                        default='data/mlp_student_bc_only.pt')
    parser.add_argument('--mlp-dagger', type=str,
                        default='data/mlp_student_v1.pt')
    parser.add_argument('--diffusion', type=str,
                        default='data/diffusion_student_v1_ema.pt')
    parser.add_argument('--out', type=str,
                        default='results/decision_point_eval')
    parser.add_argument('--t-max', type=float, default=10.0)
    parser.add_argument('--data', type=str,
                        default='data/planner_dataset_v1.npz',
                        help='used to build the diffusion normalizer')
    parser.add_argument('--no-planner', action='store_true',
                        help='skip the (slow) planner rollouts')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dp-safety-margin', type=float, default=0.15,
                        help='safety_margin for the planner ref + rollout. '
                             '0.15 matches v1 dp seeds; 0.30 matches v2.')
    parser.add_argument('--multi-K', type=int, default=1,
                        help='if >1, run diffusion with K-sample safety '
                             'filter and pick highest-min-ESDF sample')
    args = parser.parse_args()
    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[dp-eval] device={device}, seeds={args.seeds}")

    # --- load MLPs
    mlp_bc = MLPStudent()
    mlp_bc.load_state_dict(torch.load(args.mlp_bc, map_location='cpu'))
    mlp_bc = mlp_bc.to(device).eval()
    mlp_da = MLPStudent()
    mlp_da.load_state_dict(torch.load(args.mlp_dagger, map_location='cpu'))
    mlp_da = mlp_da.to(device).eval()

    # --- load diffusion
    data = np.load(_HERE.parent / args.data)
    obs_arr = data['observations']; act_arr = data['actions']
    norm = build_normalizer_from_arrays(obs_arr, act_arr)
    diff_policy = build_diffusion_policy(
        obs_dim=OBS_DIM, action_dim=ACT_DIM,
        horizon=DEFAULT_HORIZON, n_obs_steps=1,
        n_action_steps=DEFAULT_HORIZON,
        down_dims=(128, 256, 512),
        diffusion_step_embed_dim=128,
        num_inference_steps=8,
    )
    diff_policy.set_normalizer(norm)
    diff_policy.load_state_dict(
        torch.load(args.diffusion, map_location='cpu'))
    diff_policy = diff_policy.to(device).eval()
    diff_policy.noise_scheduler = make_inference_scheduler(num_inference_steps=8)
    diff_policy.num_inference_steps = 8

    # --- iterate seeds
    per_seed = {}
    trajs = {}
    summary_keys = ['goal_err_mm', 'max_field', 'mean_field', 'path_len_m',
                    'mean_speed_mps', 'median_inference_us', 'n_steps']
    student_results = {'MLP_BC': [], 'MLP_DAgger': [], 'Diffusion_BC': []}
    if not args.no_planner:
        student_results['Hierarchical_Planner'] = []

    for s in args.seeds:
        print(f"\n[dp-eval] seed {s}")
        obstacles, vm, paths = planner_paths_for_seed(
            s, safety_margin=args.dp_safety_margin)
        seed_traj = dict(start=START.tolist(), goal=GOAL.tolist(),
                          obstacles=obstacles,
                          planner_paths=[p.tolist() for p in paths])

        if not args.no_planner:
            print("  planner...", end=' ', flush=True)
            r = rollout_planner(obstacles, T_max=args.t_max,
                                safety_margin=args.dp_safety_margin)
            seed_traj['planner_xs'] = r['xs'].tolist()
            r_no_traj = {k: r[k] for k in summary_keys if k in r}
            student_results['Hierarchical_Planner'].append(r_no_traj)
            print(f"goal={r['goal_err_mm']:.0f}mm field={r['max_field']:.3f}")

        for name, model, kind in [
            ('MLP_BC', mlp_bc, 'mlp'),
            ('MLP_DAgger', mlp_da, 'mlp'),
            ('Diffusion_BC', diff_policy, 'diff'),
        ]:
            print(f"  {name}...", end=' ', flush=True)
            if kind == 'mlp':
                r = rollout_mlp(model, vm, obstacles, device,
                                T_max=args.t_max)
            else:
                if args.multi_K and args.multi_K > 1:
                    r = rollout_diffusion_multisample(
                        model, vm, obstacles, device,
                        T_max=args.t_max,
                        diffusion_seed=1234 + int(s),
                        K=int(args.multi_K))
                else:
                    r = rollout_diffusion(
                        model, vm, obstacles, device,
                        T_max=args.t_max,
                        diffusion_seed=1234 + int(s))
            seed_traj[f'{name}_xs'] = r['xs'].tolist()
            r_no_traj = {k: r[k] for k in summary_keys if k in r}
            student_results[name].append(r_no_traj)
            print(f"goal={r['goal_err_mm']:.0f}mm field={r['max_field']:.3f}")
        per_seed[str(s)] = {k: v for k, v in seed_traj.items()
                            if k.endswith('_xs') or k in ('start', 'goal')}
        trajs[str(s)] = seed_traj  # full info incl. obstacles+planner_paths

    # ---- aggregate
    aggregate = {}
    for name, rows in student_results.items():
        agg = {}
        for k in summary_keys:
            vals = [r[k] for r in rows if k in r]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            agg[k] = dict(mean=float(arr.mean()), std=float(arr.std()),
                           values=[float(v) for v in vals])
        aggregate[name] = agg

    # --- save
    out_json = Path(_HERE.parent / (args.out + '.json'))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(dict(
            metadata=dict(seeds=list(args.seeds), T_max=args.t_max),
            aggregate=aggregate,
            per_seed_summary={
                str(s): {name: student_results[name][i]
                         for name in student_results}
                for i, s in enumerate(args.seeds)
            },
        ), f, indent=2, default=lambda o: float(o)
        if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\n[dp-eval] summary -> {out_json}")

    # Save trajectories as compressed npz (large)
    out_npz = Path(_HERE.parent / (args.out + '.npz'))
    np.savez_compressed(
        out_npz,
        seeds=np.asarray(args.seeds),
        **{f'seed_{s}_{name}_xs': np.asarray(trajs[str(s)][f'{name}_xs'])
           for s in args.seeds
           for name in (['Hierarchical_Planner'] if not args.no_planner
                        else []) + ['MLP_BC', 'MLP_DAgger', 'Diffusion_BC']
           if f'{name}_xs' in trajs[str(s)]},
        **{f'seed_{s}_obstacles': np.asarray(
            [(o['center'], o['sigma'], o['weight']) for o in trajs[str(s)]['obstacles']],
            dtype=object)
           for s in args.seeds},
        **{f'seed_{s}_planner_path_{i}': np.asarray(p)
           for s in args.seeds
           for i, p in enumerate(trajs[str(s)]['planner_paths'])},
    )
    print(f"[dp-eval] trajectories -> {out_npz}")

    print("\n=== Decision-point aggregate ===")
    for name, agg in aggregate.items():
        ge = agg.get('goal_err_mm', {})
        mf = agg.get('max_field', {})
        if not ge:
            continue
        print(f"  {name:<22s}: goal_err = {ge['mean']:6.0f} +/- "
              f"{ge['std']:5.0f} mm  |  max_field = {mf['mean']:.3f} "
              f"+/- {mf['std']:.3f}")


if __name__ == '__main__':
    main()
