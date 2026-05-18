"""
ppo_finetune/reward.py -- phase-2 reward shaping.

Phase-1 retrospective: the dense per-step `-alpha * goal_dist` term ate
~95 % of the episode return budget, drowning out the safety and action
signals. The advantage signal that AWR re-weights against therefore
reflected mostly seed-level goal-distance variance, not what the policy
could actually change.

Phase 2 moves goal-reaching to a *terminal* cost and keeps only the
safety penalty (ESDF cliff) and a small action-magnitude regulariser
on the per-step term. This lets the per-step advantage track what each
step can actually change (avoid the obstacle this step, smooth the
command); the policy still gets a strong scalar at episode end to keep
it pointing at the goal.

Per-step reward (every t):
    r_t = -beta  * max(0, safe_margin - esdf_t)     # safety cliff
          - gamma * ||action_t||_2^2                # action regulariser

Terminal bonus (added once at the last step):
    r_T += -kappa * final_goal_dist[m]              # goal cost
    r_T += success_bonus  if final_goal_dist < 0.05 else 0   # discrete win

Suggested phase-2 weights:
  beta = 5.0           # unchanged from phase 1
  gamma = 0.001        # 10x smaller than phase 1 (per-step now matters
                       # more without the goal term dominating)
  kappa = 50.0         # terminal goal cost; with typical 0.1-1 m final
                       # error this yields -5 to -50 added at episode end
  success_bonus = 100.0   # discrete 100 if within 5 cm of goal
  safe_margin = 0.10 m    # unchanged

Hyperparameter guidance:
  beta: increase if random p95 max_field stays high; decrease if the
        policy stalls in narrow corridors.
  gamma: increase if commands look jittery on the test videos; decrease
         if the policy hesitates.
  kappa: increase if the policy ignores the goal (e.g. stalls); decrease
         if it sprints through obstacles to get there.
  success_bonus: should be roughly 2x the typical |terminal_cost| to
                 make reaching the goal a clearly positive event.
"""
from __future__ import annotations

import numpy as np


# ---- Phase-2 defaults ----------------------------------------------------
DEFAULT_BETA = 5.0
DEFAULT_GAMMA = 0.001
DEFAULT_KAPPA = 50.0
DEFAULT_SUCCESS_BONUS = 100.0
DEFAULT_SAFE_MARGIN = 0.10
SUCCESS_THRESHOLD_M = 0.05


def compute_step_reward(state: np.ndarray, action: np.ndarray,
                        goal: np.ndarray, esdf_value: float,
                        beta: float = DEFAULT_BETA,
                        gamma: float = DEFAULT_GAMMA,
                        safe_margin: float = DEFAULT_SAFE_MARGIN,
                        is_terminal: bool = False,
                        kappa: float = DEFAULT_KAPPA,
                        success_bonus: float = DEFAULT_SUCCESS_BONUS,
                        ) -> float:
    """Phase-2 step reward.

    The `goal` argument is kept in the signature because we need it at
    the terminal step. Callers that pass `is_terminal=True` must also
    pass a `state` whose position is the final post-step pose so the
    terminal goal_dist is correct.
    """
    pos = np.asarray(state[0:3], dtype=np.float64)
    safety_violation = max(0.0, safe_margin - float(esdf_value))
    u = np.asarray(action, dtype=np.float64)
    u_pen = float(np.dot(u, u))
    r = -beta * safety_violation - gamma * u_pen

    if is_terminal:
        final_goal_dist = float(np.linalg.norm(
            pos - np.asarray(goal, dtype=np.float64)))
        r += -kappa * final_goal_dist
        if final_goal_dist < SUCCESS_THRESHOLD_M:
            r += success_bonus
    return r


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
