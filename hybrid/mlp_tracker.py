"""
hybrid/mlp_tracker.py -- the learned reference-tracking policy of the
hybrid controller.

A 2-layer MLP (27-D observation, 64 hidden, ~6.3k params) trained by
`distillation/train_tracker.py` to behavioural-clone the NMPC's
reference-tracking actions. Reference tracking is unimodal, so the MLP
-- which collapses on the multi-modal *navigation* task -- is exactly
the right tool here, and at ~30 us inference it closes a 200 Hz loop
with three orders of magnitude of CPU headroom.

Observation (27-D), matching collect_tracking_data.py:
  obs[0:12]  = current 12-D MuJoCo state
  obs[12:24] = reference column ref[:, i]
  obs[24:27] = reference velocity ref[3:6, i]

The MLP output is a normalised action in [-1, 1]; convert to a physical
actuator command with CrazyflieEnv.u_mid + u_half * action.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / 'distillation'))

from mlp_student import MLPStudent

TRACK_OBS_DIM = 27
ACT_DIM = 4


def make_tracking_obs(state_mj: np.ndarray, ref_col: np.ndarray) -> np.ndarray:
    """Assemble the 27-D tracker observation from the current MuJoCo
    state (12-D) and a min-snap reference column (12-D)."""
    obs = np.empty(TRACK_OBS_DIM, dtype=np.float32)
    obs[0:12] = np.asarray(state_mj[0:12], dtype=np.float32)
    obs[12:24] = np.asarray(ref_col[0:12], dtype=np.float32)
    obs[24:27] = np.asarray(ref_col[3:6], dtype=np.float32)
    return obs


class MLPTracker:
    """Wraps the trained MLP reference tracker."""

    def __init__(self, model_path: str, device: str | None = None,
                 hidden: int = 64):
        self.device = torch.device(
            device if device else
            ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = MLPStudent(obs_dim=TRACK_OBS_DIM, act_dim=ACT_DIM,
                                hidden=hidden).to(self.device)
        self.model.load_state_dict(
            torch.load(_ROOT / model_path, map_location=self.device))
        self.model.eval()
        self.n_params = self.model.n_params()
        # Last measured single-step inference latency (seconds).
        self.last_inference_s = 0.0

    @torch.no_grad()
    def predict(self, state_mj: np.ndarray,
                ref_col: np.ndarray) -> np.ndarray:
        """One tracking step: returns a normalised action in [-1, 1]."""
        obs = make_tracking_obs(state_mj, ref_col)
        t0 = time.perf_counter()
        inp = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        a = self.model(inp).cpu().numpy()[0]
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.last_inference_s = time.perf_counter() - t0
        return a
