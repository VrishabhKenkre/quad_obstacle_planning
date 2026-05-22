"""
ppo_finetune/reward.py -- shortest-path reward redesign (phase 3).

Phase 2 retrospective: the policy learned to gain altitude beyond what
was needed to clear obstacles and then hover, because the per-step
safety penalty had no path-length opposing force and the terminal-only
goal cost provided no per-step gradient. This is classic reward
hacking — maximising local safety by going up and staying up.

Phase 3 rewards "shortest collision-free path to goal"
direction-agnostic. Going over an obstacle is fine if it's the
shortest path; going around is fine if it's the shortest path. What
phase 3 eliminates is gratuitous movement that doesn't earn progress
toward the goal.

Per-step reward at timestep t:

    r_t = + r_progress     # rewards shrinking distance to goal
          - r_safety       # penalty for being close to obstacle
          - r_path         # penalty for any movement (any direction)
          - r_control      # tiny penalty for control magnitude

Components:

  r_progress = lambda * (dist_to_goal(t-1) - dist_to_goal(t))
        POSITIVE when distance to goal shrinks. NEGATIVE when it grows.
        Direction-agnostic. lambda = 50.

  r_path = eta * step_length, where step_length = ||p_t - p_{t-1}||
        Penalises ANY movement. Combined with r_progress: a step that
        closes goal distance well earns net positive; a step that
        closes poorly earns net negative; a pure-vertical step when
        goal is to the side earns ~0 (no progress) minus the path
        penalty. eta = 5.

  r_safety = beta * max(0, 0.15 - esdf)  -- 0.15 m soft margin (vs
        phase 2's 0.10). beta = 8.

  r_control = gamma * ||u_t||^2 . gamma = 0.001 (unchanged from
        phase 2).

Terminal bonus (added at the final timestep only):

  terminal_bonus = kappa * max(0, success_threshold - final_goal_dist)
        success_threshold = 0.10 m, kappa = 200.
        At-goal arrival yields +20. 5 cm short yields +10. 10 cm+
        short yields 0.

This is a SHAPED reward: by Ng et al. 1999, adding a per-step
F(s,s') = -gamma * Phi(s') + Phi(s) potential-based shaping leaves
the optimal policy unchanged. With Phi(s) = -lambda * dist_to_goal(s)
the term lambda*(dist(t-1) - dist(t)) is a (slightly off) potential
shaping; the small off-by-gamma term is fine in practice for fine-
tuning a near-optimal policy. The r_path and r_safety terms ARE
real cost components (not shaping), so the optimal policy under
phase 3 is the shortest collision-free path to goal, weighted by
beta vs eta.

Two reward versions are supported here (selectable via the `version`
kwarg or train_ppo.py --reward-version flag):

  - 'phase2': the old terminal-only goal + per-step safety reward,
              kept for backward compatibility / direct comparison.
  - 'phase3': the new shortest-path reward described above. Default.

compute_step_reward always returns (float, dict) -- the scalar reward
and a per-component breakdown for diagnostics. The dict has keys
{r_progress, r_safety, r_path, r_control, r_terminal} for both
versions (unused components are 0.0).
"""
from __future__ import annotations

import numpy as np


# ---- Phase-3 defaults ---------------------------------------------------
P3_LAMBDA = 50.0       # progress weight
P3_ETA = 5.0           # path-length penalty weight
P3_BETA = 8.0          # safety penalty weight (vs phase 2's 5)
P3_GAMMA = 0.001       # control magnitude penalty
P3_SAFE_MARGIN = 0.15  # safety cliff (vs phase 2's 0.10)
P3_KAPPA = 200.0       # terminal bonus scaler
P3_SUCCESS_THRESHOLD = 0.10  # m

# ---- Phase-2 defaults (kept) --------------------------------------------
P2_BETA = 5.0
P2_GAMMA = 0.001
P2_KAPPA = 50.0
P2_SUCCESS_BONUS = 100.0
P2_SAFE_MARGIN = 0.10
P2_SUCCESS_THRESHOLD_M = 0.05


def _empty_components():
    return dict(r_progress=0.0, r_safety=0.0, r_path=0.0,
                r_control=0.0, r_terminal=0.0)


def _phase2_reward(state, action, goal, esdf_value,
                   beta=P2_BETA, gamma=P2_GAMMA,
                   safe_margin=P2_SAFE_MARGIN,
                   kappa=P2_KAPPA,
                   success_bonus=P2_SUCCESS_BONUS,
                   is_terminal=False, **_) -> tuple:
    pos = np.asarray(state[0:3], dtype=np.float64)
    safety_violation = max(0.0, safe_margin - float(esdf_value))
    u = np.asarray(action, dtype=np.float64)
    u_pen = float(np.dot(u, u))
    comp = _empty_components()
    comp['r_safety'] = -beta * safety_violation
    comp['r_control'] = -gamma * u_pen
    r = comp['r_safety'] + comp['r_control']
    if is_terminal:
        final_goal_dist = float(np.linalg.norm(
            pos - np.asarray(goal, dtype=np.float64)))
        term = -kappa * final_goal_dist
        if final_goal_dist < P2_SUCCESS_THRESHOLD_M:
            term += success_bonus
        comp['r_terminal'] = term
        r += term
    return float(r), comp


def _phase3_reward(state, action, goal, esdf_value, prev_state,
                   lam=P3_LAMBDA, eta=P3_ETA,
                   beta=P3_BETA, gamma=P3_GAMMA,
                   safe_margin=P3_SAFE_MARGIN,
                   kappa=P3_KAPPA,
                   success_threshold=P3_SUCCESS_THRESHOLD,
                   is_terminal=False, **_) -> tuple:
    pos = np.asarray(state[0:3], dtype=np.float64)
    prev_pos = np.asarray(prev_state[0:3], dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    dist_prev = float(np.linalg.norm(prev_pos - goal))
    dist_now = float(np.linalg.norm(pos - goal))
    progress = dist_prev - dist_now           # +ve when we get closer
    step_length = float(np.linalg.norm(pos - prev_pos))

    safety_violation = max(0.0, safe_margin - float(esdf_value))
    u = np.asarray(action, dtype=np.float64)
    u_pen = float(np.dot(u, u))

    comp = _empty_components()
    comp['r_progress'] = +lam * progress
    comp['r_path'] = -eta * step_length
    comp['r_safety'] = -beta * safety_violation
    comp['r_control'] = -gamma * u_pen
    r = comp['r_progress'] + comp['r_path'] + comp['r_safety'] + comp['r_control']
    if is_terminal:
        term = kappa * max(0.0, success_threshold - dist_now)
        comp['r_terminal'] = term
        r += term
    return float(r), comp


def compute_step_reward(state, action, goal, esdf_value,
                        prev_state=None, version: str = 'phase3',
                        is_terminal: bool = False, **kwargs) -> tuple:
    """Per-step reward. Returns (float, dict) -- scalar reward plus the
    per-component breakdown.

    `version` selects 'phase2' or 'phase3' shaping. Phase 3 REQUIRES
    `prev_state` (the 12-D MuJoCo state from the previous control
    step); phase 2 ignores it.
    """
    version = str(version).lower()
    if version == 'phase2':
        return _phase2_reward(state, action, goal, esdf_value,
                              is_terminal=is_terminal, **kwargs)
    if version == 'phase3':
        if prev_state is None:
            raise ValueError(
                "phase-3 reward requires prev_state (12-D MuJoCo state "
                "from the previous control step).")
        return _phase3_reward(state, action, goal, esdf_value, prev_state,
                              is_terminal=is_terminal, **kwargs)
    raise ValueError(f"unknown reward version: {version!r}")


def compute_episode_return(step_rewards: np.ndarray,
                           discount: float = 0.99) -> float:
    """Discounted sum sum_t gamma^t * r_t."""
    r = np.asarray(step_rewards, dtype=np.float64)
    if r.size == 0:
        return 0.0
    discounts = discount ** np.arange(r.size)
    return float(np.sum(discounts * r))


def compute_episode_undiscounted_return(step_rewards: np.ndarray) -> float:
    r = np.asarray(step_rewards, dtype=np.float64)
    return float(r.sum())


def aggregate_components(component_lists: list) -> dict:
    """Sum each component across a list of per-step dicts (one episode)."""
    if not component_lists:
        return _empty_components()
    keys = component_lists[0].keys()
    out = {k: float(sum(c[k] for c in component_lists)) for k in keys}
    return out
