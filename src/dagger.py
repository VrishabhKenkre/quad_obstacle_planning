"""
dagger.py -- MPC-to-RL distillation via DAgger for the Crazyflie.

DAgger pipeline: run the MPC expert (C ADMM) to collect (state, action)
pairs, train an MLP via behavioral cloning, roll the learned policy out
and relabel visited states with the expert, then retrain. The student
runs ~10x faster than the expert at deployment and is dynamics-free.

Run from the repository root:
    python3 src/dagger.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from gymnasium import spaces
import time
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))
from quad_env import CrazyflieEnv
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
from solver_admm_c import CADMMSolver


# ---- Gymnasium environment -----------------------------------------------

def gen_fig8_ff(c, r, h, per, dur, dt):
    N = int(dur / dt); t = np.arange(N) * dt; w = 2*np.pi/per; g = 9.81
    ref = np.zeros((12, N))
    ref[0] = c[0] + r * np.sin(w * t)
    ref[1] = c[1] + r * np.sin(2*w*t) / 2
    ref[2] = h
    ref[3] = r*w * np.cos(w*t)
    ref[4] = r*w * np.cos(2*w*t)
    ax = -r*w**2 * np.sin(w*t); ay = -r*w**2*2 * np.sin(2*w*t)
    ref[6] = -ay / g; ref[7] = ax / g
    return ref


class CrazyflieTrackingEnv(gym.Env):
    """Gym wrapper for Crazyflie figure-8 tracking."""
    
    def __init__(self, dt=0.01, episode_length=10.0):
        super().__init__()
        self.p = QuadParams()
        self.dt = dt
        self.episode_length = episode_length
        self.max_steps = int(episode_length / dt)
        self.N_horizon = 20
        
        model_path = str(Path(__file__).parent.parent /
                        "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml")
        self.env = CrazyflieEnv(model_path=model_path, dt_sim=0.002, dt_ctrl=dt)
        
        # Reference trajectory
        ref_dur = episode_length + self.N_horizon * dt + 2.0
        self.ref = gen_fig8_ff(np.array([0.0, 0.0]), 0.5, 1.0, 4.0, ref_dur, dt)
        self.N_ref = self.ref.shape[1]
        
        # Observation: [state(12), tracking_error(3), ref_velocity(3), ref_angles(2)] = 20
        self.obs_dim = 20
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # Action: [thrust, tau_x, tau_y, tau_z] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-np.ones(4), high=np.ones(4), dtype=np.float32)
        
        # Action scaling
        self.u_mid = (self.p.u_max + self.p.u_min) / 2
        self.u_half = (self.p.u_max - self.p.u_min) / 2
        
        self.step_count = 0
        self.state = np.zeros(12)
    
    def _get_obs(self):
        k = min(self.step_count, self.N_ref - 1)
        ref_k = self.ref[:, k]
        tracking_err = self.state[0:3] - ref_k[0:3]
        ref_vel = ref_k[3:6]
        ref_angles = ref_k[6:8]
        obs = np.concatenate([self.state, tracking_err, ref_vel, ref_angles])
        return obs.astype(np.float32)
    
    def _action_to_control(self, action):
        """Map [-1,1] action to physical control."""
        return np.clip(self.u_mid + self.u_half * action, self.p.u_min, self.p.u_max)
    
    def _control_to_action(self, control):
        """Map physical control to [-1,1] action."""
        return np.clip((control - self.u_mid) / self.u_half, -1, 1)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = self.env.reset(pos=self.ref[0:3, 0])
        return self._get_obs(), {}
    
    def step(self, action):
        control = self._action_to_control(action)
        self.state = self.env.step(control)
        self.step_count += 1
        
        # Reward: negative tracking error
        k = min(self.step_count, self.N_ref - 1)
        pos_err = np.linalg.norm(self.state[0:3] - self.ref[0:3, k])
        angle_penalty = np.sum(self.state[6:8]**2) * 0.1
        control_penalty = np.sum((control - np.array([self.p.hover_thrust, 0, 0, 0]))**2) * 10
        
        reward = -pos_err * 100 - angle_penalty - control_penalty
        
        terminated = pos_err > 1.0 or self.state[2] < 0.1 or np.any(np.abs(self.state[6:8]) > np.radians(60))
        truncated = self.step_count >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {'pos_err': pos_err}
    
    def get_ref_window(self):
        """Get reference window for MPC expert."""
        rw = np.zeros((12, self.N_horizon + 1))
        for k in range(self.N_horizon + 1):
            rw[:, k] = self.ref[:, min(self.step_count + k, self.N_ref - 1)]
        return rw


# ---- Policy network ------------------------------------------------------

class PolicyNet(nn.Module):
    """Small MLP policy: obs(20) -> action(4)."""
    
    def __init__(self, obs_dim=20, act_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh()  # output in [-1, 1]
        )
    
    def forward(self, x):
        return self.net(x)


# ---- MPC expert ----------------------------------------------------------

class MPCExpert:
    """C ADMM MPC as the expert teacher."""
    
    def __init__(self, dt=0.01):
        p = QuadParams()
        N = 20
        Ac, Bc = linearize_at_hover(p)
        Ad, Bd = discretize_dynamics(Ac, Bc, dt)
        Q_d = np.array([300,300,300, 10,10,10, 3,3,1, 0.1,0.1,0.1])
        R_d = np.array([30, 1.5e3, 1.5e3, 1.5e3])
        uh = np.array([p.hover_thrust, 0, 0, 0])
        dg = (np.eye(12) - Ad) @ np.zeros(12) - Bd @ uh
        INF = 1e10
        xlo = np.array([-INF]*3+[-INF]*3+[-np.radians(35)]*2+[-INF]*4)
        xhi = np.array([INF]*3+[INF]*3+[np.radians(35)]*2+[INF]*4)
        
        self.solver = CADMMSolver(Ad, Bd, Q_d, R_d, N,
                                  p.u_min, p.u_max, xlo, xhi,
                                  uh, dg, rho=1.0, max_iter=200)
        self.p = p
    
    def get_action(self, state, ref_window):
        u, info = self.solver.solve(state, ref_window)
        self.solver.warm_shift()
        return u


# ---- DAgger pipeline -----------------------------------------------------

def collect_expert_data(env, expert, n_episodes=5):
    """Roll out MPC expert, collect (obs, action) pairs."""
    obs_list, act_list = [], []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            ref_w = env.get_ref_window()
            u_expert = expert.get_action(env.state, ref_w)
            action = env._control_to_action(u_expert)
            
            obs_list.append(obs.copy())
            act_list.append(action.copy().astype(np.float32))
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    
    return np.array(obs_list), np.array(act_list)


def collect_dagger_data(env, expert, policy, device, n_episodes=3):
    """Roll out LEARNED policy, query MPC expert at visited states."""
    obs_list, act_list = [], []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # Run LEARNED policy
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = policy(obs_t).cpu().numpy()[0]
            
            # But LABEL with MPC expert
            ref_w = env.get_ref_window()
            u_expert = expert.get_action(env.state, ref_w)
            expert_action = env._control_to_action(u_expert).astype(np.float32)
            
            obs_list.append(obs.copy())
            act_list.append(expert_action)
            
            # Step with LEARNED policy's action (not expert)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    
    return np.array(obs_list), np.array(act_list)


def train_policy(policy, optimizer, obs_data, act_data, device, epochs=50, batch_size=256):
    """Train policy via supervised learning (behavioral cloning)."""
    dataset = TensorDataset(
        torch.FloatTensor(obs_data).to(device),
        torch.FloatTensor(act_data).to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for obs_batch, act_batch in loader:
            pred = policy(obs_batch)
            loss = nn.MSELoss()(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
    
    return losses


def evaluate_policy(env, policy, expert, device, label=""):
    """Evaluate learned policy vs MPC expert."""
    # Learned policy
    obs, _ = env.reset()
    errors_policy = []
    times_policy = []
    done = False
    while not done:
        t0 = time.perf_counter()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = policy(obs_t).cpu().numpy()[0]
        times_policy.append(time.perf_counter() - t0)
        obs, _, terminated, truncated, info = env.step(action)
        errors_policy.append(info['pos_err'])
        done = terminated or truncated
    
    # MPC expert
    obs, _ = env.reset()
    errors_expert = []
    times_expert = []
    done = False
    while not done:
        ref_w = env.get_ref_window()
        t0 = time.perf_counter()
        u_expert = expert.get_action(env.state, ref_w)
        times_expert.append(time.perf_counter() - t0)
        action = env._control_to_action(u_expert)
        obs, _, terminated, truncated, info = env.step(action)
        errors_expert.append(info['pos_err'])
        done = terminated or truncated
    
    skip = int(2.0 / env.dt)
    
    def ss_rmse(errs):
        e = np.array(errs)
        if len(e) > skip:
            return np.sqrt(np.mean(e[skip:]**2))
        return np.sqrt(np.mean(e**2))
    
    ss_p = ss_rmse(errors_policy)
    ss_e = ss_rmse(errors_expert)
    t_p = np.median(times_policy) * 1e6
    t_e = np.median(times_expert) * 1e6
    
    print(f"  {label:<20s} | Policy: {ss_p*1000:6.1f}mm @ {t_p:6.0f}us | "
          f"MPC: {ss_e*1000:6.1f}mm @ {t_e:6.0f}us | "
          f"Speedup: {t_e/max(t_p,1):.0f}x")
    
    return errors_policy, errors_expert, times_policy, times_expert


# ---- Main pipeline -------------------------------------------------------

def run_dagger(n_dagger_iters=5, n_expert_episodes=5, n_dagger_episodes=3,
               bc_epochs=100, dagger_epochs=50):
    
    device = torch.device('cpu')  # small network, CPU is faster
    
    print("[DAgger] MPC -> RL distillation")
    print("  Teacher: C ADMM (38us, 3.7mm)")
    print("  Student: 2-layer MLP (64 hidden, ~2K params)")
    
    env = CrazyflieTrackingEnv(dt=0.01, episode_length=10.0)
    expert = MPCExpert(dt=0.01)
    
    policy = PolicyNet(obs_dim=20, act_dim=4, hidden=64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy: {n_params} parameters ({n_params*4/1024:.1f} KB)\n")
    
    # Step 1: collect expert demonstrations.
    print("Step 1: Collecting MPC expert demonstrations...")
    t0 = time.time()
    obs_data, act_data = collect_expert_data(env, expert, n_episodes=n_expert_episodes)
    print(f"  Collected {len(obs_data)} samples from {n_expert_episodes} episodes "
          f"({time.time()-t0:.1f}s)")
    
    # ---- Step 2: behavioral cloning ----------------------------------
    print("\nStep 2: Behavioral cloning (supervised learning on expert data)...")
    losses_bc = train_policy(policy, optimizer, obs_data, act_data, device,
                             epochs=bc_epochs, batch_size=256)
    print(f"  BC loss: {losses_bc[0]:.4f} -> {losses_bc[-1]:.4f}")
    
    print("\n  Evaluation after BC:")
    eval_results = []
    errs_p, errs_e, _, _ = evaluate_policy(env, policy, expert, device, "After BC")
    eval_results.append(('BC', np.array(errs_p), np.array(errs_e)))
    
    # ---- Step 3: DAgger iterations -----------------------------------
    all_losses = [losses_bc[-1]]
    
    for dagger_iter in range(n_dagger_iters):
        print(f"\nDAgger iteration {dagger_iter+1}/{n_dagger_iters}:")
        
        # Collect on-policy data with expert labels
        print(f"  Collecting on-policy data ({n_dagger_episodes} episodes)...")
        new_obs, new_act = collect_dagger_data(env, expert, policy, device,
                                                n_episodes=n_dagger_episodes)
        print(f"  Got {len(new_obs)} new samples")
        
        # Aggregate
        obs_data = np.concatenate([obs_data, new_obs])
        act_data = np.concatenate([act_data, new_act])
        print(f"  Total dataset: {len(obs_data)} samples")
        
        # Retrain
        losses = train_policy(policy, optimizer, obs_data, act_data, device,
                              epochs=dagger_epochs, batch_size=256)
        all_losses.append(losses[-1])
        print(f"  Loss: {losses[-1]:.4f}")
        
        # Evaluate
        errs_p, errs_e, _, _ = evaluate_policy(env, policy, expert, device,
                                                f"DAgger iter {dagger_iter+1}")
        eval_results.append((f'DAgger-{dagger_iter+1}', np.array(errs_p), np.array(errs_e)))
    
    # ---- Final evaluation ---------
    print("\n  Final Comparison")
    errs_final_p, errs_final_e, times_p, times_e = evaluate_policy(
        env, policy, expert, device, "FINAL")
    
    # ---- Plots -------------------------------------------------------
    t = np.arange(len(errs_final_p)) * env.dt
    t_e = np.arange(len(errs_final_e)) * env.dt
    skip = int(2.0 / env.dt)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('DAgger: MPC -> RL Policy Distillation\n'
                 'Teacher: C ADMM (3.7mm, 38us) -> Student: 2-layer MLP',
                 fontsize=13, fontweight='bold')
    
    # 1) Tracking error comparison
    ax = axes[0, 0]
    ax.plot(t_e, np.array(errs_final_e)*1000, 'b-', lw=1.5,
            label=f'MPC (SS={np.sqrt(np.mean(np.array(errs_final_e)[skip:]**2))*1000:.1f}mm)')
    ax.plot(t, np.array(errs_final_p)*1000, 'r-', lw=1.5, alpha=0.8,
            label=f'Policy (SS={np.sqrt(np.mean(np.array(errs_final_p)[skip:]**2))*1000:.1f}mm)')
    ax.set_title('Tracking Error'); ax.set_ylabel('[mm]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim([0, 50])
    
    # 2) DAgger loss progression
    ax = axes[0, 1]
    ax.plot(range(len(all_losses)), all_losses, 'bo-', lw=2)
    ax.set_title('DAgger Loss Progression'); ax.set_ylabel('MSE Loss')
    ax.set_xlabel('Iteration (0=BC)'); ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(all_losses)))
    ax.set_xticklabels(['BC'] + [f'D{i+1}' for i in range(n_dagger_iters)])
    
    # 3) Solve time comparison
    ax = axes[0, 2]
    labels_bar = ['MPC\n(C ADMM)', 'Learned\nPolicy']
    times_bar = [np.median(times_e)*1e6, np.median(times_p)*1e6]
    colors_bar = ['#2980b9', '#e74c3c']
    bars = ax.bar(labels_bar, times_bar, color=colors_bar, alpha=0.8)
    ax.set_ylabel('Inference Time [us]'); ax.set_title('Speed Comparison')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, times_bar):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.2,
                f'{val:.0f}us', ha='center', fontsize=11, fontweight='bold')
    
    # 4) Error over DAgger iterations
    ax = axes[1, 0]
    names = []; ss_vals_p = []; ss_vals_e = []
    for name, ep, ee in eval_results:
        names.append(name)
        ss_vals_p.append(np.sqrt(np.mean(ep[skip:]**2))*1000 if len(ep) > skip else np.sqrt(np.mean(ep**2))*1000)
        ss_vals_e.append(np.sqrt(np.mean(ee[skip:]**2))*1000 if len(ee) > skip else np.sqrt(np.mean(ee**2))*1000)
    x_pos = np.arange(len(names))
    ax.bar(x_pos - 0.15, ss_vals_p, 0.3, color='#e74c3c', alpha=0.8, label='Policy')
    ax.axhline(ss_vals_e[0], color='#2980b9', ls='--', lw=2, label='MPC baseline')
    ax.set_ylabel('SS-RMSE [mm]'); ax.set_title('Policy Improvement')
    ax.set_xticks(x_pos); ax.set_xticklabels(names, fontsize=8)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    
    # 5) Training data growth
    ax = axes[1, 1]
    data_sizes = [n_expert_episodes * env.max_steps]
    for i in range(n_dagger_iters):
        last = data_sizes[-1]
        # Each DAgger episode may be shorter (crash), estimate
        data_sizes.append(last + n_dagger_episodes * env.max_steps)
    ax.bar(range(len(data_sizes)), [d/1000 for d in data_sizes], color='#27ae60', alpha=0.8)
    ax.set_ylabel('Dataset Size [K samples]'); ax.set_title('Data Aggregation')
    ax.set_xlabel('Iteration'); ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(range(len(data_sizes)))
    ax.set_xticklabels(['BC'] + [f'D{i+1}' for i in range(n_dagger_iters)])
    
    # 6) Summary
    ax = axes[1, 2]; ax.axis('off')
    ss_mpc = np.sqrt(np.mean(np.array(errs_final_e)[skip:]**2))*1000
    ss_pol = np.sqrt(np.mean(np.array(errs_final_p)[skip:]**2))*1000
    t_mpc = np.median(times_e)*1e6
    t_pol = np.median(times_p)*1e6
    summary = (
        f"{'DAgger Results':^40s}\n{'-'*40}\n"
        f"{'':20s}{'MPC':>10s}{'Policy':>10s}\n{'-'*40}\n"
        f"{'SS-RMSE [mm]':20s}{ss_mpc:>9.1f} {ss_pol:>9.1f}\n"
        f"{'Inference [us]':20s}{t_mpc:>9.0f} {t_pol:>9.0f}\n"
        f"{'Speedup':20s}{'1x':>9s} {t_mpc/max(t_pol,1):>8.0f}x\n"
        f"{'Parameters':20s}{'N/A':>9s} {n_params:>9d}\n"
        f"{'Model size':20s}{'-':>9s} {n_params*4/1024:>7.1f}KB\n"
        f"{'-'*40}\n"
        f"DAgger iterations: {n_dagger_iters}\n"
        f"Expert episodes: {n_expert_episodes}\n"
        f"Total training samples: {len(obs_data)}\n"
    )
    ax.text(0.05, 0.95, summary, fontsize=10, fontfamily='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(str(Path(__file__).parent.parent / 'results' / 'dagger_results.png'),
                dpi=150, bbox_inches='tight')
    print(f"\n  Saved results/dagger_results.png")
    
    # Save policy
    torch.save(policy.state_dict(),
               str(Path(__file__).parent.parent / 'results' / 'dagger_policy.pt'))
    print(f"  Saved results/dagger_policy.pt ({n_params*4/1024:.1f} KB)")
    
    return policy, eval_results


if __name__ == '__main__':
    run_dagger(
        n_dagger_iters=5,
        n_expert_episodes=5,
        n_dagger_episodes=3,
        bc_epochs=100,
        dagger_epochs=50
    )
