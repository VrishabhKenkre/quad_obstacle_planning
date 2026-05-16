"""
nonlinear_mpc.py — SE(3) nonlinear MPC for Crazyflie 2 with obstacle avoidance.

State (13D):  x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
Input (4D):   u = [T1, T2, T3, T4]   (rotor thrusts, N)

Dynamics
--------
  ṗ = v
  v̇ = (1/m) * R(q) e3 * (T1+T2+T3+T4) - g e3
  q̇ = 0.5 * H(omega) * q                 (Hamilton convention quaternion product)
  ω̇ = J^-1 * (tau_body - omega x J*omega)

where the mixer (X-config rotors at +/- l on body x,y axes):

  tau_body_x =  l_arm * (T1 - T2 - T3 + T4)
  tau_body_y =  l_arm * (T1 + T2 - T3 - T4)
  tau_body_z =  c_t   * (T1 - T2 + T3 - T4)

(rotor 1 = front-right, 2 = back-right, 3 = back-left, 4 = front-left;
 alternating CCW/CW/CCW/CW so opposite rotors share spin direction.)

For applying to MuJoCo, we use the reverse mapping to (T_total, tau_xyz):
  T_total = T1 + T2 + T3 + T4
  tau_x   = l_arm * (T1 - T2 - T3 + T4)
  tau_y   = l_arm * (T1 + T2 - T3 - T4)
  tau_z   = c_t   * (T1 - T2 + T3 - T4)

then MuJoCo's _u_to_ctrl converts (T_total, tau_xyz) -> ctrl array.

Obstacle cost
-------------
Sum of axis-aligned 3D Gaussians:
  J_obs(p) = sum_i  w_i * exp( -((p - c_i) / sigma_i)^2 / 2 )

Differentiable, no hard constraints. Pushed into stage cost.
"""
import sys, os, time
sys.path.insert(0, '.')
import numpy as np
import casadi as ca
from pathlib import Path

# Physical parameters (Crazyflie 2)
M = 0.027                  # kg
G = 9.81                   # m/s^2
IXX = IYY = 2.3951e-5      # kg m^2
IZZ = 3.2347e-5
L_ARM = 0.046              # m (rotor arm)
C_T = 0.005964552          # drag-thrust ratio (Crazyflie 2, from Mellinger ICRA 2011)
T_MAX_ROTOR = 0.15         # N per rotor (so 4 rotors x 0.15 = 0.6 N total)


def quat_mul_mat(q):
    """Left-multiplication matrix for quaternion q = [w, x, y, z]."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(w, -x, -y, -z),
        ca.horzcat(x,  w, -z,  y),
        ca.horzcat(y,  z,  w, -x),
        ca.horzcat(z, -y,  x,  w),
    )


def quat_to_R_e3(q):
    """Third column of rotation matrix R(q): body z-axis in world frame."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(2*(x*z + w*y),
                      2*(y*z - w*x),
                      1 - 2*(x*x + y*y))


def dynamics_continuous(x, u):
    """ẋ = f(x, u). x is 13D, u is 4D rotor thrusts."""
    p = x[0:3]
    v = x[3:6]
    q = x[6:10]
    w = x[10:13]
    T1, T2, T3, T4 = u[0], u[1], u[2], u[3]
    T_total = T1 + T2 + T3 + T4
    tau_x = L_ARM * (T1 - T2 - T3 + T4)
    tau_y = L_ARM * (T1 + T2 - T3 - T4)
    tau_z = C_T   * (T1 - T2 + T3 - T4)

    # Accelerations
    pdot = v
    vdot = (T_total / M) * quat_to_R_e3(q) - ca.vertcat(0, 0, G)

    # Quaternion derivative: q̇ = 0.5 * q ⊗ [0, w]
    omega_q = ca.vertcat(0, w[0], w[1], w[2])
    qdot = 0.5 * quat_mul_mat(q) @ omega_q

    # Rotational dynamics
    J = ca.diag(ca.vertcat(IXX, IYY, IZZ))
    Jw = J @ w
    wdot = ca.solve(J, ca.vertcat(tau_x, tau_y, tau_z) - ca.cross(w, Jw))

    return ca.vertcat(pdot, vdot, qdot, wdot)


def rk4_step(x, u, dt):
    """4th-order Runge-Kutta integration step."""
    k1 = dynamics_continuous(x, u)
    k2 = dynamics_continuous(x + dt/2 * k1, u)
    k3 = dynamics_continuous(x + dt/2 * k2, u)
    k4 = dynamics_continuous(x + dt   * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def gaussian_obstacle_cost(p, obstacles):
    """Sum of axis-aligned 3D Gaussian costs. p is (3,), obstacles is list of dicts."""
    cost = 0.0
    for obs in obstacles:
        c = ca.DM(obs['center'])
        sigma = ca.DM(obs['sigma'])
        w = obs['weight']
        delta = (p - c) / sigma
        cost = cost + w * ca.exp(-0.5 * ca.sumsqr(delta))
    return cost


class SE3_NMPC:
    """Closed-loop SE(3) NMPC with obstacles and reference tracking."""

    def __init__(self, N=20, dt=0.02, obstacles=None,
                 q_pos=300.0, q_vel=10.0, q_quat=20.0, q_omega=0.1,
                 r_thrust=1e3, w_obs=200.0):
        self.N = N
        self.dt = dt
        self.obstacles = obstacles or []
        self.nx = 13
        self.nu = 4

        # Decision variables
        opti = ca.Opti()
        X = opti.variable(13, N + 1)
        U = opti.variable(4, N)
        X0 = opti.parameter(13)
        XREF = opti.parameter(3, N + 1)   # only position+velocity reference
        VREF = opti.parameter(3, N + 1)

        # Cost
        J = 0
        u_hover_per_rotor = M * G / 4  # ~0.066 N per rotor at hover
        for k in range(N):
            pos_err = X[0:3, k] - XREF[:, k]
            vel_err = X[3:6, k] - VREF[:, k]
            # Quaternion regularization: prefer hover orientation [1,0,0,0]
            # Use 1 - qw^2 - qz^2 as a soft attitude regularizer (penalize tilt only)
            q_w = X[6, k]; q_x = X[7, k]; q_y = X[8, k]; q_z = X[9, k]
            tilt = q_x*q_x + q_y*q_y
            J += q_pos  * ca.sumsqr(pos_err)
            J += q_vel  * ca.sumsqr(vel_err)
            J += q_quat * tilt
            J += q_omega * ca.sumsqr(X[10:13, k])
            J += r_thrust * ca.sumsqr(U[:, k] - u_hover_per_rotor)
            # Obstacle cost
            J += w_obs * gaussian_obstacle_cost(X[0:3, k], self.obstacles)
        # Terminal cost (no terminal obs to allow goal inside obstacle field)
        pos_err_N = X[0:3, N] - XREF[:, N]
        J += q_pos  * 10 * ca.sumsqr(pos_err_N)
        J += q_vel  * 10 * ca.sumsqr(X[3:6, N] - VREF[:, N])

        opti.minimize(J)

        # Dynamics constraints (RK4)
        for k in range(N):
            x_next = rk4_step(X[:, k], U[:, k], dt)
            opti.subject_to(X[:, k+1] == x_next)
        opti.subject_to(X[:, 0] == X0)

        # Input bounds
        opti.subject_to(opti.bounded(0.0, U, T_MAX_ROTOR))

        # Quaternion unit norm (soft via terminal cost — IPOPT will keep it near 1)
        for k in range(N + 1):
            qnorm = ca.sumsqr(X[6:10, k]) - 1.0
            J += 100.0 * qnorm * qnorm
        opti.minimize(J)

        # IPOPT options
        opts = {
            'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0,
            'ipopt.max_iter': 100, 'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.warm_start_init_point': 'yes',
        }
        opti.solver('ipopt', opts)

        self.opti = opti
        self.X = X; self.U = U; self.X0 = X0; self.XREF = XREF; self.VREF = VREF
        self.prev_X = None; self.prev_U = None

    def solve(self, x0, ref_p, ref_v):
        """Solve one NMPC step. ref_p, ref_v are (3, N+1) arrays."""
        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.XREF, ref_p)
        self.opti.set_value(self.VREF, ref_v)

        # Warm start
        if self.prev_X is not None:
            X_init = np.zeros((13, self.N + 1))
            X_init[:, :-1] = self.prev_X[:, 1:]
            X_init[:, -1] = self.prev_X[:, -1]
            U_init = np.zeros((4, self.N))
            U_init[:, :-1] = self.prev_U[:, 1:]
            U_init[:, -1] = self.prev_U[:, -1]
            self.opti.set_initial(self.X, X_init)
            self.opti.set_initial(self.U, U_init)
        else:
            X_init = np.tile(x0[:, None], (1, self.N + 1))
            self.opti.set_initial(self.X, X_init)
            self.opti.set_initial(self.U, M * G / 4 * np.ones((4, self.N)))

        t0 = time.time()
        try:
            sol = self.opti.solve()
            U_opt = sol.value(self.U)
            X_opt = sol.value(self.X)
            t_solve = time.time() - t0
            self.prev_X = X_opt; self.prev_U = U_opt
            return U_opt[:, 0], dict(solve_time=t_solve, status='solved',
                                     X_pred=X_opt, U_seq=U_opt)
        except Exception as e:
            t_solve = time.time() - t0
            # Take debug values from IPOPT
            try:
                U_opt = self.opti.debug.value(self.U)
                X_opt = self.opti.debug.value(self.X)
                self.prev_X = X_opt; self.prev_U = U_opt
                return U_opt[:, 0], dict(solve_time=t_solve, status='inaccurate',
                                         X_pred=X_opt, U_seq=U_opt)
            except:
                return np.array([M*G/4]*4), dict(solve_time=t_solve, status='failed',
                                                 X_pred=None, U_seq=None, error=str(e))


def rotors_to_mujoco(u_rotors):
    """Convert (T1, T2, T3, T4) -> (T_total, tau_x, tau_y, tau_z) for MuJoCo CrazyflieEnv."""
    T1, T2, T3, T4 = u_rotors
    T_total = T1 + T2 + T3 + T4
    tau_x = L_ARM * (T1 - T2 - T3 + T4)
    tau_y = L_ARM * (T1 + T2 - T3 - T4)
    tau_z = C_T   * (T1 - T2 + T3 - T4)
    return np.array([T_total, tau_x, tau_y, tau_z])


if __name__ == '__main__':
    # Quick smoke test: hover
    nmpc = SE3_NMPC(N=20, dt=0.02, obstacles=[])
    x0 = np.array([0,0,1, 0,0,0, 1,0,0,0, 0,0,0], dtype=float)
    ref_p = np.tile(np.array([0,0,1.0])[:, None], (1, 21))
    ref_v = np.zeros((3, 21))
    print("Compiling NMPC and solving first step...")
    u, info = nmpc.solve(x0, ref_p, ref_v)
    print(f"  status={info['status']} time={info['solve_time']*1000:.1f}ms")
    print(f"  u (rotor thrusts) = {u}")
    print(f"  u_total = {sum(u):.4f} N (should be ~{M*G:.4f} N for hover)")
    print(f"  MuJoCo input: {rotors_to_mujoco(u)}")
