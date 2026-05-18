"""
ppo_finetune/advantage.py -- GAE on rollouts using a small value
network.

The value network is a 32->32 MLP on the 24-D observation. It is fit
from random init at the start of each AWR iteration via a few hundred
MSE-on-Monte-Carlo-returns steps. We re-fit each iteration rather than
maintain a long-running V across iterations: at this scale and with
small policy drift, the marginal benefit of carrying V forward is
small, and re-fitting gives a stronger sanity guarantee that V tracks
the most recent rollouts.

GAE notation:
  delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
  A_t = sum_{l=0..} (gamma * lambda)^l * delta_{t+l},
        clamped to within-episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ValueNet(nn.Module):
    """Tiny MLP: obs (Do,) -> scalar V(s).

    32->32 hidden, ReLU, linear output. Initialised so V(s) ~ 0 at the
    start of training (small last-layer weights). This is the only
    extra learned module aside from the diffusion student itself, and
    it is *not* shared across iterations -- we re-init each call.
    """
    def __init__(self, obs_dim: int = 24, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def fit_value_net(value_net: ValueNet, obs_all: np.ndarray,
                  returns_all: np.ndarray, device: torch.device,
                  epochs: int = 30, batch_size: int = 512,
                  lr: float = 1e-3, verbose: bool = False) -> dict:
    """Fit V(s) to Monte-Carlo returns by MSE."""
    obs_t = torch.from_numpy(obs_all.astype(np.float32)).to(device)
    ret_t = torch.from_numpy(returns_all.astype(np.float32)).to(device)
    N = obs_t.shape[0]
    opt = optim.AdamW(value_net.parameters(), lr=lr, weight_decay=1e-6)
    value_net.train()
    history = []
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        tot = 0.0; cnt = 0
        for s in range(0, N, batch_size):
            idx = perm[s:s+batch_size]
            v = value_net(obs_t[idx])
            loss = ((v - ret_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * idx.numel()
            cnt += idx.numel()
        history.append(tot / max(cnt, 1))
        if verbose and (ep == 0 or ep == epochs - 1 or ep % 10 == 0):
            print(f"    [value-fit] ep {ep:2d}: mse={history[-1]:.4f}")
    value_net.eval()
    return dict(loss_curve=history)


def compute_mc_returns(rewards: np.ndarray, dones: np.ndarray,
                       gamma: float) -> np.ndarray:
    """Per-step Monte-Carlo returns (sum_t gamma^t r_t within an episode)."""
    R = np.zeros_like(rewards, dtype=np.float64)
    G = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        if bool(dones[i]):
            G = 0.0
        G = float(rewards[i]) + gamma * G
        R[i] = G
    return R.astype(np.float32)


def compute_gae(values: np.ndarray, rewards: np.ndarray,
                dones: np.ndarray, episode_ends: np.ndarray,
                gamma: float, lam: float) -> np.ndarray:
    """GAE advantages, computed independently per episode."""
    A = np.zeros_like(rewards, dtype=np.float64)
    start = 0
    for end in episode_ends:
        end = int(end)
        # bootstrap V_{end} = 0 (episode terminates by horizon truncation)
        gae = 0.0
        for t in range(end - 1, start - 1, -1):
            v_t = float(values[t])
            v_tp1 = float(values[t + 1]) if (t + 1) < end else 0.0
            done_t = bool(dones[t]) and (t == end - 1)
            mask = 0.0 if done_t else 1.0
            delta = float(rewards[t]) + gamma * v_tp1 * mask - v_t
            gae = delta + gamma * lam * mask * gae
            A[t] = gae
        start = end
    return A.astype(np.float32)


def estimate_advantages(episodes: list, device: torch.device,
                        gamma: float = 0.99, lam: float = 0.95,
                        value_epochs: int = 30, verbose: bool = True,
                        value_net: 'ValueNet | None' = None,
                        ) -> dict:
    """Re-fit a value net and compute GAE per transition.

    Phase-2 change: if a `value_net` is supplied it is used as the warm
    start (no re-init); otherwise a fresh ValueNet is constructed. The
    returned `value_net` is the (now-trained) reference, suitable for
    carrying forward into the next iteration.

    Returns:
        dict with arrays of shape (sum_steps,) and an aligned `obs`
        tensor for the AWR update.
    """
    obs_all = np.concatenate([e.obs for e in episodes], axis=0)
    rew_all = np.concatenate([e.rewards for e in episodes], axis=0)
    done_all = np.concatenate([e.dones for e in episodes], axis=0)
    episode_ends = np.cumsum([e.n_steps for e in episodes], dtype=np.int64)

    mc_returns = compute_mc_returns(rew_all, done_all, gamma=gamma)

    if value_net is None:
        value_net = ValueNet(obs_dim=obs_all.shape[1]).to(device)
    else:
        value_net = value_net.to(device)
    fit_value_net(value_net, obs_all, mc_returns, device,
                  epochs=value_epochs, verbose=verbose)
    with torch.no_grad():
        values = value_net(
            torch.from_numpy(obs_all.astype(np.float32)).to(device)
        ).cpu().numpy()
    adv = compute_gae(values, rew_all, done_all, episode_ends,
                      gamma=gamma, lam=lam)
    # Normalise advantages (zero mean, unit std) -- standard GAE practice;
    # the temperature in AWR is then dataset-independent.
    a_mean = float(adv.mean()); a_std = float(adv.std()) + 1e-6
    adv_norm = (adv - a_mean) / a_std

    if verbose:
        print(f"    [advantage] returns mean={mc_returns.mean():.2f}, "
              f"V mean={values.mean():.2f}, "
              f"raw A std={adv.std():.3f}, "
              f"normalised A in [{adv_norm.min():.2f}, "
              f"{adv_norm.max():.2f}]")

    return dict(
        obs=obs_all, returns=mc_returns, values=values,
        advantages=adv, advantages_norm=adv_norm,
        episode_ends=episode_ends, value_net=value_net,
        adv_mean=a_mean, adv_std=a_std,
    )
