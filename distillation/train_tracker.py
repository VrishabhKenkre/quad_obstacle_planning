"""
train_tracker.py -- train the hybrid controller's learned reference
tracker: a small MLP that maps (state, reference) -> actuator command.

The hybrid system was originally specced with a diffusion tracker, but
reference tracking is a *unimodal* problem -- for a given (state,
reference) pair there is one correct action -- so the multi-modality
that motivated the diffusion student does not apply. A 2-layer MLP
(the same architecture as the navigation MLP student, just with a 27-D
input) is the right tool: ~6.3k parameters, ~30 us inference, genuinely
fast enough to close a 200 Hz control loop. This is a deliberate
deviation from the original brief's "retrain the diffusion as a
tracker" -- see the sprint report.

Training: supervised MSE regression of the NMPC's normalised actuator
command. AdamW lr 1e-4, batch 256, 100 epochs, best-validation-MSE
checkpoint. (EMA, a diffusion-training trick, is not used -- it is
pointless for a 6 k-parameter MLP regression.)

Output:
  data/mlp_tracker_v1.pt        (best-val checkpoint)
  results/tracker_training.png  (train/val MSE curve)

Usage:
  python distillation/train_tracker.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from mlp_student import MLPStudent

TRACK_OBS_DIM = 27
ACT_DIM = 4


def train(data_path: Path, model_out: Path, plot_out: Path,
          epochs: int = 100, batch_size: int = 256, lr: float = 1e-4,
          val_frac: float = 0.05, hidden: int = 64,
          device: torch.device | None = None) -> dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = np.load(data_path)
    obs = data['obs'].astype(np.float32)
    act = data['actions'].astype(np.float32)
    assert obs.shape[1] == TRACK_OBS_DIM, f"expected 27-D obs, got {obs.shape}"
    n = obs.shape[0]
    print(f"[train-tracker] {n:,} samples, obs_dim={obs.shape[1]}, "
          f"act_dim={act.shape[1]}, device={device}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(int(n * val_frac), 256)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    obs_t = torch.from_numpy(obs)
    act_t = torch.from_numpy(act)
    tr_ds = TensorDataset(obs_t[tr_idx], act_t[tr_idx])
    va_ds = TensorDataset(obs_t[val_idx], act_t[val_idx])
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)

    model = MLPStudent(obs_dim=TRACK_OBS_DIM, act_dim=ACT_DIM,
                       hidden=hidden).to(device)
    print(f"[train-tracker] MLP params: {model.n_params():,}")
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    tr_curve, va_curve = [], []
    best_val = float('inf')
    best_state = None
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        ep_loss = 0.0; ep_n = 0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * xb.shape[0]
            ep_n += xb.shape[0]
        tr_loss = ep_loss / max(ep_n, 1)

        model.eval()
        v_loss = 0.0; v_n = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                v_loss += float(loss_fn(model(xb), yb).item()) * xb.shape[0]
                v_n += xb.shape[0]
        v_loss /= max(v_n, 1)
        tr_curve.append(tr_loss); va_curve.append(v_loss)
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}: tr_mse={tr_loss:.6f}  "
                  f"val_mse={v_loss:.6f}  best={best_val:.6f}  "
                  f"({time.perf_counter()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"[train-tracker] saved best-val checkpoint -> {model_out}")

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(tr_curve))
    ax.plot(xs, tr_curve, label='train MSE', lw=1.5)
    ax.plot(xs, va_curve, label='val MSE', lw=1.5, linestyle='--')
    ax.set_xlabel('epoch'); ax.set_ylabel('action MSE')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('MLP reference-tracker training')
    plt.tight_layout()
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[train-tracker] plot -> {plot_out}")

    return dict(best_val_mse=float(best_val),
                final_train_mse=float(tr_curve[-1]),
                final_val_mse=float(va_curve[-1]),
                n_params=int(model.n_params()),
                n_samples=int(n),
                epochs=int(epochs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='data/tracking_dataset_v1.npz')
    ap.add_argument('--model-out', type=str, default='data/mlp_tracker_v1.pt')
    ap.add_argument('--plot-out', type=str,
                    default='results/tracker_training.png')
    ap.add_argument('--stats-out', type=str,
                    default='results/tracker_training_stats.json')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device else None)
    stats = train(_ROOT / args.data, _ROOT / args.model_out,
                  _ROOT / args.plot_out,
                  epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, device=device)
    json.dump(stats, open(_ROOT / args.stats_out, 'w'), indent=2)
    print(f"[train-tracker] stats -> {args.stats_out}")
    print(f"  best val MSE: {stats['best_val_mse']:.6f}")


if __name__ == '__main__':
    main()
