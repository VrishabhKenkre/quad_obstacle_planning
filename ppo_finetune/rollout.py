"""
ppo_finetune/rollout.py -- collect MuJoCo rollouts under the current
diffusion student. Phase 1 uses K=1 sampling (no multi-sample safety
filter) so the rollouts reflect the raw policy distribution that AWR
should update against.

Each rollout records, per control step:
  obs       (24,) -- the 24-D ESDF-aware observation
  action    (4,)  -- normalised [-1, 1] action actually executed
  reward    scalar -- shaped reward (see ppo_finetune.reward)
  done      bool -- True only on the last transition
  esdf      scalar -- ESDF at the post-step position (for diagnostics)
  state     (12,) -- MuJoCo state for diagnostics

Episode metadata: seed, layout, max_field, goal_err_mm, undiscounted return.

The compressed `.npz` written to `data/ppo_rollouts/round_N.npz` is a
flat record array (one row per transition, all episodes concatenated)
plus an `episode_ends` index array. This lets `advantage.py` slice per
episode without keeping a Python dict of variable-length arrays.

Seed pool: a fixed `ROLLOUT_SEED_POOL` of 60 obstacle seeds (40 random
+ 20 dp) disjoint from the eval seeds in `train_diffusion.SEEDS` and
`eval_decision_points.DEFAULT_DP_SEEDS`. Each call samples without
replacement from this pool to give a reproducible-but-diverse rollout
distribution.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / 'distillation'))
sys.path.insert(0, str(_ROOT / 'planning'))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'external' / 'diffusion_policy'))

from voxelize import VoxelMap
from quad_env import CrazyflieEnv
from obstacle_course import make_obstacles, obstacle_field_value

from collect_planner_data import make_observation, OBS_DIM, ACT_DIM, START, GOAL
from randomize_astar import decision_point_layout

from ppo_finetune.reward import compute_step_reward, compute_episode_undiscounted_return


# Held-out from the eval seeds: random eval uses {42, 7, 13, 99, 256, 128, 314,
# 2024, 777, 1337}; dp eval uses range(10). The pool below is disjoint from
# both.
ROLLOUT_RANDOM_SEEDS = list(range(50000, 50000 + 200))   # 200 random seeds
ROLLOUT_DP_SEEDS = list(range(60000, 60000 + 30))         # 30 dp seeds
ROLLOUT_SEED_POOL = [(s, 'random') for s in ROLLOUT_RANDOM_SEEDS] \
                    + [(s, 'dp') for s in ROLLOUT_DP_SEEDS]


@dataclass
class Episode:
    seed: int
    layout: str
    obs: np.ndarray            # (T, OBS_DIM)
    actions: np.ndarray        # (T, ACT_DIM)
    rewards: np.ndarray        # (T,)
    dones: np.ndarray          # (T,) bool
    states: np.ndarray         # (T, 12)
    esdf_vals: np.ndarray      # (T,)
    max_field: float
    goal_err_mm: float
    n_steps: int


def _build_obstacles(seed: int, layout: str):
    if layout == 'random':
        return make_obstacles(seed=int(seed))
    elif layout == 'dp':
        obstacles, _, _ = decision_point_layout(seed=int(seed))
        return obstacles
    raise ValueError(f"layout must be 'random' or 'dp': {layout!r}")


def rollout_one_episode(policy, seed: int, layout: str, device,
                        horizon: int = 300, dt_ctrl: float = 0.02,
                        reward_kwargs: Optional[dict] = None,
                        diffusion_torch_seed: int = 0) -> Episode:
    """Run a single rollout. `horizon` is the max number of control steps
    (= max episode length); we don't early-terminate on reach because the
    diffusion student hovers near the goal once it arrives, which is
    fine for the reward signal."""
    obstacles = _build_obstacles(seed, layout)
    vm = VoxelMap(); vm.from_obstacle_field(obstacles); vm.compute_esdf()
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=dt_ctrl)
    state_mj = env.reset(pos=START)
    u_mid = (env.u_max + env.u_min) / 2.0
    u_half = (env.u_max - env.u_min) / 2.0

    if diffusion_torch_seed is not None:
        torch.manual_seed(int(diffusion_torch_seed))
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(int(diffusion_torch_seed))

    reward_kwargs = reward_kwargs or {}
    goal = np.asarray(GOAL, dtype=np.float64)

    obs_list, act_list, rew_list = [], [], []
    state_list, esdf_list = [], []
    max_field = 0.0
    policy.eval()
    for t in range(horizon):
        obs = make_observation(state_mj, GOAL, vm)
        obs_list.append(obs.astype(np.float32).copy())
        state_list.append(state_mj.astype(np.float32).copy())

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device).reshape(1, 1, OBS_DIM)
            result = policy.predict_action({'obs': obs_t})
            a = result['action'][0, 0].cpu().numpy()
        act_list.append(a.astype(np.float32).copy())

        u = u_mid + u_half * a
        u = np.clip(u, env.u_min, env.u_max)
        state_mj = env.step(u)

        # Post-step observation properties for reward + diagnostics
        esdf_after = float(vm.query_esdf(state_mj[0:3]))
        field_after = obstacle_field_value(state_mj[0:3], obstacles)
        max_field = max(max_field, float(field_after))
        esdf_list.append(esdf_after)
        is_terminal = (t == horizon - 1)
        r = compute_step_reward(state_mj, a, goal, esdf_after,
                                is_terminal=is_terminal,
                                **reward_kwargs)
        rew_list.append(float(r))

    dones = np.zeros(len(rew_list), dtype=bool)
    if dones.size > 0:
        dones[-1] = True
    final_pos = state_mj[0:3]
    goal_err_mm = float(np.linalg.norm(final_pos - GOAL) * 1000)
    return Episode(
        seed=int(seed), layout=str(layout),
        obs=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(act_list, dtype=np.float32),
        rewards=np.asarray(rew_list, dtype=np.float32),
        dones=dones,
        states=np.asarray(state_list, dtype=np.float32),
        esdf_vals=np.asarray(esdf_list, dtype=np.float32),
        max_field=max_field, goal_err_mm=goal_err_mm,
        n_steps=len(rew_list),
    )


def collect_rollouts(policy, n_episodes: int = 50, horizon: int = 300,
                     device: Optional[torch.device] = None,
                     reward_kwargs: Optional[dict] = None,
                     rng_seed: int = 0,
                     verbose: bool = True) -> list:
    """Run `n_episodes` rollouts of the diffusion student.

    Sampling: without replacement from `ROLLOUT_SEED_POOL` until exhausted
    then we cycle. The diffusion RNG seed is `1234 + obstacle_seed` so
    repeating an episode is reproducible.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(int(rng_seed))
    pool = list(ROLLOUT_SEED_POOL)
    rng.shuffle(pool)
    pool = (pool * ((n_episodes + len(pool) - 1) // len(pool)))[:n_episodes]

    episodes: list = []
    t0 = time.time()
    for ep_i, (seed, layout) in enumerate(pool):
        ep = rollout_one_episode(
            policy, seed=seed, layout=layout, device=device,
            horizon=horizon, reward_kwargs=reward_kwargs,
            diffusion_torch_seed=1234 + int(seed))
        episodes.append(ep)
        if verbose and (ep_i % 5 == 0 or ep_i == n_episodes - 1):
            ret = compute_episode_undiscounted_return(ep.rewards)
            dt = time.time() - t0
            print(f"    [rollout] {ep_i+1}/{n_episodes} "
                  f"layout={layout} seed={seed:>6d} "
                  f"goal={ep.goal_err_mm:.0f}mm "
                  f"max_field={ep.max_field:.3f} "
                  f"return={ret:.2f}  ({dt:.1f}s elapsed)")
    return episodes


def save_rollouts_npz(episodes: list, out_path: str):
    """Flatten the list of Episode dataclasses into a single npz file."""
    obs = np.concatenate([e.obs for e in episodes], axis=0)
    acts = np.concatenate([e.actions for e in episodes], axis=0)
    rews = np.concatenate([e.rewards for e in episodes], axis=0)
    dones = np.concatenate([e.dones for e in episodes], axis=0)
    states = np.concatenate([e.states for e in episodes], axis=0)
    esdf = np.concatenate([e.esdf_vals for e in episodes], axis=0)
    episode_ends = np.cumsum([e.n_steps for e in episodes], dtype=np.int64)
    meta = np.asarray([(e.seed, e.layout, e.n_steps, e.max_field, e.goal_err_mm)
                       for e in episodes],
                      dtype=[('seed', 'i8'), ('layout', 'U10'),
                             ('n_steps', 'i8'), ('max_field', 'f4'),
                             ('goal_err_mm', 'f4')])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        obs=obs, actions=acts, rewards=rews, dones=dones,
                        states=states, esdf=esdf,
                        episode_ends=episode_ends, meta=meta)
    print(f"    [rollout] saved {len(episodes)} episodes "
          f"({obs.shape[0]} transitions) -> {out_path}")
