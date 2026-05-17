"""
diffusion_student.py -- the diffusion student. Wraps Cheng Chi's
DiffusionUnetLowdimPolicy (external/diffusion_policy) for our 24-D
SDF observation and 4-D Crazyflie action setup.

Key configuration:
  * obs_dim   = 24       (same as MLP student)
  * action_dim = 4        (normalised actuator commands in [-1, 1])
  * horizon    = 8        (predicts 8-step action sequence)
  * n_obs_steps    = 1    (single-step observation conditioning)
  * n_action_steps = 1    (executes first predicted action each step)
  * obs_as_global_cond = True   (FiLM conditioning on flattened obs)
  * pred_action_steps_only = True
  * 100 DDPM steps training, 8 DDIM steps at inference
  * ConditionalUnet1D with down_dims=[256, 512, 1024] -> ~70M params,
    too big for an 8GB 4070. We use [128, 256, 512] for ~1M params
    or [64, 128, 256] for ~250k. (Cheng Chi's defaults target images.)

We always train and infer using ACTION-space (4-D), not obs-space
inpainting -- with obs_as_global_cond=True the trajectory is purely
action.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / 'external' / 'diffusion_policy'))

from diffusion_policy.policy.diffusion_unet_lowdim_policy import (
    DiffusionUnetLowdimPolicy)
from diffusion_policy.model.diffusion.conditional_unet1d import (
    ConditionalUnet1D)
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


OBS_DIM = 24
ACT_DIM = 4
DEFAULT_HORIZON = 8
DEFAULT_DOWN_DIMS = (128, 256, 512)
DEFAULT_NUM_TRAIN_STEPS = 100
DEFAULT_NUM_INF_STEPS = 8


def build_diffusion_policy(
        obs_dim: int = OBS_DIM,
        action_dim: int = ACT_DIM,
        horizon: int = DEFAULT_HORIZON,
        n_obs_steps: int = 1,
        n_action_steps: int | None = None,
        down_dims=DEFAULT_DOWN_DIMS,
        num_train_timesteps: int = DEFAULT_NUM_TRAIN_STEPS,
        num_inference_steps: int = DEFAULT_NUM_INF_STEPS,
        diffusion_step_embed_dim: int = 128,
        ) -> DiffusionUnetLowdimPolicy:
    """Construct a DiffusionUnetLowdimPolicy in the global-cond +
    pred-action-only configuration.

    The U-Net needs the trajectory length to be a multiple of
    2^(len(down_dims) - 1). With down_dims=[128,256,512] (2 downsampling
    levels), the minimum legal length is 4. We default to 8 so the U-Net
    has comfortable receptive field.

    n_action_steps defaults to `horizon` so the U-Net sees a full
    `horizon`-long trajectory. At inference we receding-horizon: execute
    only the first action of the predicted sequence.
    """
    if n_action_steps is None:
        n_action_steps = horizon
    global_cond_dim = obs_dim * n_obs_steps
    model = ConditionalUnet1D(
        input_dim=action_dim,
        local_cond_dim=None,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=list(down_dims),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=True,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_inference_steps,
        obs_as_local_cond=False,
        obs_as_global_cond=True,
        pred_action_steps_only=True,
        oa_step_convention=True,
    )
    return policy


def make_inference_scheduler(num_inference_steps: int = DEFAULT_NUM_INF_STEPS,
                             num_train_timesteps: int = DEFAULT_NUM_TRAIN_STEPS
                             ) -> DDIMScheduler:
    """Build a DDIM scheduler matching the DDPM config but with fewer
    inference steps. Swap into `policy.noise_scheduler` for fast sampling.
    """
    return DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
        set_alpha_to_one=True,
        steps_offset=0,
    )


def build_normalizer_from_arrays(observations, actions) -> LinearNormalizer:
    """LinearNormalizer fit from dataset arrays (numpy). Both arrays are
    fit independently so per-dim scaling is well-posed."""
    import torch
    obs_t = torch.as_tensor(observations, dtype=torch.float32)
    act_t = torch.as_tensor(actions, dtype=torch.float32)
    norm = LinearNormalizer()
    norm.fit({'obs': obs_t, 'action': act_t},
             last_n_dims=1, mode='limits', output_max=1.0,
             output_min=-1.0, range_eps=1e-4)
    return norm


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    policy = build_diffusion_policy()
    print(f"DiffusionUnetLowdimPolicy params: {n_params(policy):,}")
    # Smoke: fake batch through compute_loss
    B, T, Do, Da = 2, 8, OBS_DIM, ACT_DIM
    import numpy as np
    obs = np.random.randn(B, 1, Do).astype(np.float32)
    act = np.random.randn(B, T, Da).astype(np.float32) * 0.3
    norm = build_normalizer_from_arrays(obs.reshape(-1, Do),
                                        act.reshape(-1, Da))
    policy.set_normalizer(norm)
    batch = dict(
        obs=torch.from_numpy(obs),
        action=torch.from_numpy(act),
    )
    loss = policy.compute_loss(batch)
    print(f"smoke loss: {float(loss):.4f}")
    # Smoke inference
    policy.eval()
    with torch.no_grad():
        out = policy.predict_action({'obs': torch.from_numpy(obs)})
    print(f"action shape: {out['action'].shape}  "
          f"pred shape: {out['action_pred'].shape}")
