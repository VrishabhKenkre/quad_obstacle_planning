"""
mlp_student.py -- the reactive MLP student that distills the hierarchical
planner. Architecture matches the existing IL student exactly (2-layer MLP,
64 hidden, ReLU, tanh output) so the MLP-of-planner vs MLP-of-NMPC
comparison is apples-to-apples.

obs (24) -> Linear(24, 64) -> ReLU
         -> Linear(64, 64) -> ReLU
         -> Linear(64, 4)  -> tanh -> action in [-1, 1]

Parameter count: 24*64+64 + 64*64+64 + 64*4+4 = 6,020. The existing
20-D student lists 5,764; the 256 extra params come from the 4 added
obs dims (the SDF-based observation we adopted in Phase 1).

The action is in normalised actuator space and is converted to physical
units by CrazyflieEnv.u_mid + u_half * action.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# Observation / action dims; kept in lockstep with collect_planner_data.py.
OBS_DIM = 24
ACT_DIM = 4
HIDDEN = 64


class MLPStudent(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = HIDDEN):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Convenience inference path. Returns numpy action in [-1, 1]."""
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        a = self.forward(x).cpu().numpy()
        return a[0] if single else a


if __name__ == '__main__':
    model = MLPStudent()
    n = model.n_params()
    print(f"MLPStudent: {n:,} parameters "
          f"(target 5,764 for 20-D obs; +256 for our 24-D obs)")
    print(model)
    # Smoke test forward pass.
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    a = model.predict(obs)
    print(f"input obs shape: {obs.shape}")
    print(f"output action  : {a}   (range [-1, 1])")
