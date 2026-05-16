"""
obstacle_course.py — Linear MPC vs Nonlinear MPC on a Gaussian obstacle field.

Setup:
  - 8 random soft obstacles (3D Gaussian penalty fields) inside a 3x3x1 m box
  - Start: (-1.5, -1.5, 1.0)
  - Goal:  (+1.5, +1.5, 1.0)
  - Reference: straight-line motion from start to goal over 5 s
  - Both controllers track the reference. NMPC also pays a cost for entering
    obstacles. Linear MPC doesn't see them.

Measures:
  - SS-RMSE to reference
  - Minimum obstacle clearance (= 1/max obstacle "field value" along path)
  - Path arc length
  - Wall solve time
"""
import sys, time, os
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

from nonlinear_mpc import SE3_NMPC, rotors_to_mujoco, M, G
from quad_env import CrazyflieEnv
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
from solver_admm_c import CADMMSolver


# ─── Deterministic obstacle course ───
def make_obstacles(seed=42, n=8):
    rng = np.random.RandomState(seed)
    obstacles = []
    # Restrict to corridor between start (-1.5,-1.5,1) and goal (1.5,1.5,1)
    # so they actually interfere with the path
    for _ in range(n):
        cx = rng.uniform(-1.2, 1.2)
        cy = rng.uniform(-1.2, 1.2)
        cz = rng.uniform(0.7, 1.3)
        sig = rng.uniform(0.18, 0.30)
        obstacles.append(dict(center=[cx, cy, cz], sigma=[sig, sig, sig], weight=1.0))
    return obstacles


def obstacle_field_value(p, obstacles):
    """Sum of obstacle Gaussian values at point p — higher = more dangerous."""
    val = 0.0
    for obs in obstacles:
        c = np.array(obs['center']); s = np.array(obs['sigma'])
        d = ((p - c) / s) ** 2
        val += obs['weight'] * np.exp(-0.5 * np.sum(d))
    return val


def make_reference(start, goal, duration, dt):
    """Straight-line reference from start to goal."""
    N = int(duration / dt)
    t = np.arange(N) * dt / duration
    p = start[:, None] * (1 - t)[None, :] + goal[:, None] * t[None, :]
    v = ((goal - start) / duration)[:, None] * np.ones((1, N))
    return p, v   # shape (3, N) each


# ─── Run NMPC ───
def run_nmpc(obstacles, start, goal, duration=5.0, dt=0.04):
    nmpc = SE3_NMPC(N=15, dt=dt, obstacles=obstacles,
                    q_pos=300, q_vel=10, q_quat=20, q_omega=0.1,
                    r_thrust=1e3, w_obs=800.0)
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=dt)
    x_mj = env.reset(pos=start)

    # Reference
    ref_p, ref_v = make_reference(start, goal, duration, dt)
    total = ref_p.shape[1]

    xs = [x_mj.copy()]
    us = []
    times = []
    field_vals = []
    for i in range(total):
        # MuJoCo (12D rpy) -> NMPC (13D quat)
        p, v, rpy, w = x_mj[0:3], x_mj[3:6], x_mj[6:9], x_mj[9:12]
        phi, theta, psi = rpy
        cy, sy = np.cos(psi/2), np.sin(psi/2); cp, sp = np.cos(theta/2), np.sin(theta/2)
        cr, sr = np.cos(phi/2), np.sin(phi/2)
        q = np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                       cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])
        x13 = np.concatenate([p, v, q, w])

        # Window: pad with goal
        rp_win = np.zeros((3, 16)); rv_win = np.zeros((3, 16))
        for k in range(16):
            j = min(i + k, total - 1)
            rp_win[:, k] = ref_p[:, j]; rv_win[:, k] = ref_v[:, j]

        t0 = time.time()
        u_rot, info = nmpc.solve(x13, rp_win, rv_win)
        times.append(time.time() - t0)

        u_mj = rotors_to_mujoco(u_rot)
        x_mj = env.step(u_mj)
        xs.append(x_mj.copy())
        us.append(u_mj.copy())
        field_vals.append(obstacle_field_value(x_mj[0:3], obstacles))

    return dict(xs=np.array(xs), us=np.array(us), times=np.array(times),
                field_vals=np.array(field_vals), ref_p=ref_p, ref_v=ref_v,
                total=total, dt=dt)


# ─── Run linear MPC (obstacle-blind) ───
def run_linear(obstacles, start, goal, duration=5.0, dt=0.04):
    p_ = QuadParams()
    Ac, Bc = linearize_at_hover(p_)
    Ad, Bd = discretize_dynamics(Ac, Bc, dt)
    Q_d = np.array([300, 300, 300, 10, 10, 10, 3, 3, 1, 0.1, 0.1, 0.1])
    R_d = np.array([30, 1.5e3, 1.5e3, 1.5e3])
    uh = np.array([p_.hover_thrust, 0, 0, 0])
    d = (np.eye(12) - Ad) @ np.zeros(12) - Bd @ uh
    INF = 1e10
    xlo = np.array([-INF]*3 + [-INF]*3 + [-np.radians(35)]*2 + [-INF]*4)
    xhi = np.array([INF]*3 + [INF]*3 + [np.radians(35)]*2 + [INF]*4)

    solver = CADMMSolver(Ad, Bd, Q_d, R_d, 20, p_.u_min, p_.u_max, xlo, xhi,
                         uh, d, rho=1.0, max_iter=200, eps_abs=1e-4)

    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=dt)
    x_mj = env.reset(pos=start)

    ref_p, ref_v = make_reference(start, goal, duration, dt)
    total = ref_p.shape[1]

    xs = [x_mj.copy()]; us = []; times = []; field_vals = []
    for i in range(total):
        # Build 12D reference window (with feedforward attitudes from accel)
        win = np.zeros((12, 21))
        for k in range(21):
            j = min(i + k, total - 1)
            win[0:3, k] = ref_p[:, j]
            win[3:6, k] = ref_v[:, j]
            # No accel feedforward (it's a constant-velocity straight line)

        t0 = time.time()
        u, info = solver.solve(x_mj, win)
        times.append(time.time() - t0)
        solver.warm_shift()

        x_mj = env.step(u)
        xs.append(x_mj.copy()); us.append(u.copy())
        field_vals.append(obstacle_field_value(x_mj[0:3], obstacles))

    return dict(xs=np.array(xs), us=np.array(us), times=np.array(times),
                field_vals=np.array(field_vals), ref_p=ref_p, ref_v=ref_v,
                total=total, dt=dt)


# ─── Main ───
if __name__ == '__main__':
    start = np.array([-1.5, -1.5, 1.0])
    goal  = np.array([ 1.5,  1.5, 1.0])
    obstacles = make_obstacles(seed=42, n=8)

    print(f"\n=== Obstacle course: {len(obstacles)} Gaussian obstacles ===")
    for i, obs in enumerate(obstacles):
        print(f"  obs {i}: center {obs['center']}, sigma {obs['sigma'][0]:.3f}")

    print("\n[Linear MPC (obstacle-blind)] running...")
    t0 = time.time()
    r_lin = run_linear(obstacles, start, goal)
    print(f"  done in {time.time()-t0:.1f}s | "
          f"solve median {np.median(r_lin['times'])*1e6:.0f}us")

    print("\n[NMPC SE(3) (obstacle-aware)] running...")
    t0 = time.time()
    r_nmpc = run_nmpc(obstacles, start, goal)
    print(f"  done in {time.time()-t0:.1f}s | "
          f"solve median {np.median(r_nmpc['times'])*1000:.0f}ms")

    # ─── Metrics ───
    def metrics(r, label):
        xs = r['xs']
        ref = r['ref_p']
        path_err = np.linalg.norm(xs[1:, 0:3].T - ref, axis=0)
        # exclude warm-up
        warmup = int(1.0 / r['dt'])
        rmse = np.sqrt(np.mean(path_err[warmup:]**2)) * 1000
        # Path length
        path_len = np.sum(np.linalg.norm(np.diff(xs[:, 0:3], axis=0), axis=1))
        # Obstacle interaction
        max_field = np.max(r['field_vals'])
        mean_field = np.mean(r['field_vals'])
        # Goal proximity at end
        final_err = np.linalg.norm(xs[-1, 0:3] - goal) * 1000
        return dict(label=label, rmse_to_ref_mm=rmse, path_len_m=path_len,
                    max_field=max_field, mean_field=mean_field,
                    final_err_mm=final_err, solve_us=np.median(r['times'])*1e6)

    m_lin = metrics(r_lin, 'Linear MPC')
    m_nmpc = metrics(r_nmpc, 'SE(3) NMPC')

    print(f"\n=== Metrics ===")
    print(f"  {'metric':<22s} | {'Linear MPC':>12s} | {'SE(3) NMPC':>12s}")
    print("  " + "-"*55)
    for k in ['rmse_to_ref_mm', 'path_len_m', 'max_field', 'mean_field', 'final_err_mm', 'solve_us']:
        lv = m_lin[k]; nv = m_nmpc[k]
        if k == 'solve_us':
            print(f"  {k:<22s} | {lv:>12.1f} | {nv:>12.0f}")
        else:
            print(f"  {k:<22s} | {lv:>12.3f} | {nv:>12.3f}")

    # ─── Plot ───
    out = Path('../results'); out.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(14, 6))

    # 3D path comparison
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # Show obstacles as wireframe spheres at sigma
    u_sph, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    for obs in obstacles:
        c = obs['center']; s = obs['sigma'][0]
        xs_s = c[0] + s * np.cos(u_sph) * np.sin(v_sph)
        ys_s = c[1] + s * np.sin(u_sph) * np.sin(v_sph)
        zs_s = c[2] + s * np.cos(v_sph)
        ax.plot_wireframe(xs_s, ys_s, zs_s, color='red', alpha=0.2, lw=0.4)
    # Paths
    ax.plot(r_lin['xs'][:, 0], r_lin['xs'][:, 1], r_lin['xs'][:, 2], 'C1', lw=2,
            label=f'Linear MPC (max field {m_lin["max_field"]:.2f})')
    ax.plot(r_nmpc['xs'][:, 0], r_nmpc['xs'][:, 1], r_nmpc['xs'][:, 2], 'C0', lw=2,
            label=f'SE(3) NMPC (max field {m_nmpc["max_field"]:.2f})')
    # Reference (straight line)
    ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]],
            'k--', lw=1.0, alpha=0.5, label='reference (straight)')
    ax.scatter(*start, color='green', s=60, label='start')
    ax.scatter(*goal, color='blue', s=60, label='goal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_title('Path through 8-obstacle Gaussian field')
    ax.legend(fontsize=8, loc='upper left')

    # Obstacle field along path
    ax2 = fig.add_subplot(1, 2, 2)
    t_lin = np.arange(len(r_lin['field_vals'])) * r_lin['dt']
    t_nmpc = np.arange(len(r_nmpc['field_vals'])) * r_nmpc['dt']
    ax2.plot(t_lin, r_lin['field_vals'], 'C1', lw=1.5,
             label=f'Linear MPC (peak {m_lin["max_field"]:.2f})')
    ax2.plot(t_nmpc, r_nmpc['field_vals'], 'C0', lw=1.5,
             label=f'SE(3) NMPC (peak {m_nmpc["max_field"]:.2f})')
    ax2.axhline(1.0, color='red', ls=':', lw=1, alpha=0.6, label='deep obstacle interior')
    ax2.set_xlabel('time [s]'); ax2.set_ylabel('Obstacle field value (sum of Gaussians)')
    ax2.set_title('Obstacle field encountered along path')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / 'obstacle_course.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out / 'obstacle_course.png'}")

    # Save data
    np.savez(out / 'obstacle_course.npz',
             obstacles=obstacles, start=start, goal=goal,
             linear_xs=r_lin['xs'], nmpc_xs=r_nmpc['xs'],
             linear_field=r_lin['field_vals'], nmpc_field=r_nmpc['field_vals'],
             linear_times=r_lin['times'], nmpc_times=r_nmpc['times'])
