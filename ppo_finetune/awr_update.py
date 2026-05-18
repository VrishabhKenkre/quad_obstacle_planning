"""
ppo_finetune/awr_update.py -- one AWR optimisation step against the
collected rollouts.

AWR (advantage-weighted regression, Peng et al. 2019) replaces the BC
loss with an advantage-weighted BC loss:
    L = E[ exp(A/T) * BC_loss(s, a) ]
where BC_loss is the standard DDPM noise-prediction MSE used by the
existing diffusion student. T (temperature) controls the sharpness of
the re-weighting; small T pushes the update toward the highest-A
transitions, large T degenerates to plain BC.

Key differences from BC training:
  - Per-sample loss is weighted (we cannot use the policy's built-in
    compute_loss because it averages over the batch internally; instead
    we replicate its DDPM loss with a per-sample reduction).
  - Learning rate is 10x lower than BC (1e-5 vs 1e-4) -- we are
    fine-tuning, not training from scratch.
  - Exp-weights are clipped to a maximum (default 5.0) so a single
    outlier-advantage transition cannot dominate the gradient.
  - We report parameter L2 drift relative to the start-of-step policy
    so the training loop can abort on policy collapse.

Action windows: the DDPM loss expects (B, T_act, Da) action windows.
We extract sliding windows of length W=8 from each episode -- exactly
the format used by `distillation/train_diffusion.PlannerWindowDataset`.
The advantage at the *start* of the window weights the BC loss for
that window. Windows that would cross an episode boundary are dropped.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce


def _build_windows(obs_all: np.ndarray, actions_all: np.ndarray,
                   adv_all: np.ndarray, episode_ends: np.ndarray,
                   window: int = 8) -> dict:
    """Slide windows of length `window` within each episode."""
    starts = []
    start = 0
    for end in episode_ends:
        end = int(end)
        if end - start >= window:
            starts.extend(range(start, end - window + 1))
        start = end
    starts = np.asarray(starts, dtype=np.int64)
    # gather windows
    obs_win = obs_all[starts]                       # (N, Do) -- start obs
    act_win = np.stack([actions_all[s:s+window] for s in starts], axis=0)  # (N, W, Da)
    adv_win = adv_all[starts]                       # (N,) -- adv at window start
    return dict(obs=obs_win, action=act_win, adv=adv_win, starts=starts)


def _ddpm_per_sample_loss(policy, obs_t: torch.Tensor,
                          action_t: torch.Tensor) -> torch.Tensor:
    """Replicates DiffusionUnetLowdimPolicy.compute_loss but returns the
    per-sample loss (B,) rather than the scalar batch mean -- so we
    can weight it externally.
    """
    nbatch = policy.normalizer.normalize({'obs': obs_t, 'action': action_t})
    obs = nbatch['obs']
    action = nbatch['action']

    # obs_as_global_cond + pred_action_steps_only path (matches the
    # configuration in build_diffusion_policy)
    global_cond = obs[:, :policy.n_obs_steps, :].reshape(obs.shape[0], -1)
    To = policy.n_obs_steps
    start = To - 1 if policy.oa_step_convention else To
    end = start + policy.n_action_steps
    trajectory = action[:, start:end]

    # No conditioning mask when pred_action_steps_only=True
    noise = torch.randn(trajectory.shape, device=trajectory.device)
    bsz = trajectory.shape[0]
    timesteps = torch.randint(
        0, policy.noise_scheduler.config.num_train_timesteps,
        (bsz,), device=trajectory.device).long()
    noisy = policy.noise_scheduler.add_noise(trajectory, noise, timesteps)
    pred = policy.model(noisy, timesteps, local_cond=None,
                        global_cond=global_cond)
    if policy.noise_scheduler.config.prediction_type == 'epsilon':
        target = noise
    else:
        target = trajectory
    loss = F.mse_loss(pred, target, reduction='none')        # (B, T, Da)
    per_sample = reduce(loss, 'b ... -> b', 'mean')          # (B,)
    return per_sample


def _param_l2(policy_a, policy_b) -> float:
    """L2 distance between two flat parameter vectors."""
    sd_a = {k: v.detach() for k, v in policy_a.state_dict().items()}
    sd_b = {k: v.detach() for k, v in policy_b.state_dict().items()}
    total = 0.0
    for k, va in sd_a.items():
        if k not in sd_b:
            continue
        if va.dtype not in (torch.float32, torch.float16, torch.bfloat16,
                            torch.float64):
            continue
        vb = sd_b[k].to(va.device)
        total += float(((va - vb) ** 2).sum().item())
    return float(np.sqrt(total))


def _param_l2_norm(policy) -> float:
    """L2 norm of all trainable float parameters in a policy."""
    total = 0.0
    for v in policy.state_dict().values():
        if not torch.is_tensor(v):
            continue
        if v.dtype not in (torch.float32, torch.float16, torch.bfloat16,
                           torch.float64):
            continue
        total += float((v.detach() ** 2).sum().item())
    return float(np.sqrt(total))


def param_drift_relative(policy_now, policy_init) -> float:
    """||theta_now - theta_init|| / ||theta_init|| -- the scale-aware
    drift metric used by phase 2 to replace the absolute cap that was
    too tight for the 10.8M-param network."""
    init_norm = _param_l2_norm(policy_init)
    if init_norm < 1e-9:
        return float('inf')
    return _param_l2(policy_now, policy_init) / init_norm


def awr_step(diffusion_policy, rollouts_obs: np.ndarray,
             rollouts_actions: np.ndarray, rollouts_advantages: np.ndarray,
             episode_ends: np.ndarray, device: torch.device,
             n_grad_steps: int = 50, batch_size: int = 64,
             lr: float = 1e-5, temperature: float = 0.1,
             weight_clip: float = 5.0, window: int = 8,
             init_policy: Optional[object] = None,
             verbose: bool = True) -> dict:
    """Perform n_grad_steps AWR updates on `diffusion_policy` in-place.

    Returns:
        dict with avg loss, avg weighted loss, parameter L2 drift, and
        a rough KL-to-init proxy (the mean DDPM loss difference on a
        held-out batch -- not a true KL, but tracks the same direction).
    """
    # Build sliding windows once
    win = _build_windows(rollouts_obs, rollouts_actions, rollouts_advantages,
                         episode_ends, window=window)
    if len(win['starts']) == 0:
        raise RuntimeError(
            f"No valid {window}-length windows found in rollouts; "
            f"horizon too short relative to window?")
    obs_t = torch.from_numpy(win['obs'][:, None, :].astype(np.float32)).to(device)   # (N, 1, Do)
    act_t = torch.from_numpy(win['action'].astype(np.float32)).to(device)            # (N, W, Da)
    adv_t = torch.from_numpy(win['adv'].astype(np.float32)).to(device)               # (N,)
    N = obs_t.shape[0]

    # Snapshot for parameter drift measurement.
    snapshot = copy.deepcopy(diffusion_policy).to(device).eval()
    for p in snapshot.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(diffusion_policy.parameters(),
                            lr=lr, weight_decay=1e-6)
    diffusion_policy.train()

    losses, weighted_losses = [], []
    rng = np.random.default_rng(0)
    for step in range(n_grad_steps):
        idx = rng.choice(N, size=min(batch_size, N), replace=False)
        idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
        ob = obs_t[idx_t]; ac = act_t[idx_t]; ad = adv_t[idx_t]
        # exp(A/T) with clipping
        w = torch.exp(ad / float(temperature)).clamp(max=float(weight_clip))
        # per-sample DDPM loss
        per_loss = _ddpm_per_sample_loss(diffusion_policy, ob, ac)   # (B,)
        # IMPORTANT: stop-gradient through the weights -- they depend only
        # on rollout advantages, not on the policy.
        loss = (w.detach() * per_loss).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion_policy.parameters(), 5.0)
        opt.step()
        losses.append(float(per_loss.mean().item()))
        weighted_losses.append(float(loss.item()))
        if verbose and (step == 0 or step == n_grad_steps - 1 or step % 10 == 0):
            print(f"    [awr] step {step:2d}: per_loss={losses[-1]:.4f}  "
                  f"weighted={weighted_losses[-1]:.4f}  "
                  f"|w|_mean={float(w.mean().item()):.3f}  "
                  f"|w|_max={float(w.max().item()):.3f}")

    diffusion_policy.eval()
    # Parameter drift L2
    drift = _param_l2(diffusion_policy, snapshot)
    # KL-to-init proxy: DDPM-loss gap on a fixed held-out batch
    init_for_kl = init_policy if init_policy is not None else snapshot
    init_for_kl.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        idx = rng.choice(N, size=min(256, N), replace=False)
        idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
        ob = obs_t[idx_t]; ac = act_t[idx_t]
        torch.manual_seed(0)
        loss_now = _ddpm_per_sample_loss(diffusion_policy, ob, ac).mean()
        torch.manual_seed(0)
        loss_init = _ddpm_per_sample_loss(init_for_kl, ob, ac).mean()
        kl_proxy = float((loss_now - loss_init).abs().item())

    return dict(
        loss_curve=losses,
        weighted_loss_curve=weighted_losses,
        param_drift_l2=float(drift),
        kl_to_init=float(kl_proxy),
        n_windows=int(N),
        mean_weight=float(np.mean([0.0])) if not losses else float(np.mean([float(w.mean().item())])),
    )
