"""quad_dynamics.py -- 12-state quadrotor dynamics with CasADi.

Nonlinear Newton-Euler model with state x = [pos, vel, rpy, omega]
(world-frame translation, body-frame angular velocity, ZYX Euler) and
control u = [thrust, tau_xyz]. Provides analytical linearization at
hover and discretization for MPC formulations.
"""

import numpy as np
import casadi as ca
from scipy.linalg import solve_discrete_are, expm
from dataclasses import dataclass


@dataclass
class QuadParams:
    """Crazyflie 2 physical parameters.
    
    Source: MuJoCo Menagerie bitcraze_crazyflie_2/cf2.xml
    Validated against: Landry (2015), Förster (2015) system ID
    """
    mass: float = 0.027                  # kg
    Ixx: float = 2.3951e-5               # kg·m²
    Iyy: float = 2.3951e-5               # kg·m²
    Izz: float = 3.2347e-5               # kg·m²
    g: float = 9.81                      # m/s²
    
    # Actuator limits
    thrust_min: float = 0.0              # N
    thrust_max: float = 0.60             # N (TWR ~ 2.27, realistic)
    torque_rp_max: float = 0.0069        # N*m (roll/pitch: arm * motor_thrust)
    torque_yaw_max: float = 0.0036       # N*m (yaw: reactive torque)
    
    # Arm length (for reference)
    arm_length: float = 0.046            # m
    
    @property
    def hover_thrust(self) -> float:
        return self.mass * self.g
    
    @property
    def I_diag(self) -> np.ndarray:
        return np.array([self.Ixx, self.Iyy, self.Izz])
    
    @property
    def u_min(self) -> np.ndarray:
        return np.array([self.thrust_min, -self.torque_rp_max, 
                         -self.torque_rp_max, -self.torque_yaw_max])
    
    @property
    def u_max(self) -> np.ndarray:
        return np.array([self.thrust_max, self.torque_rp_max, 
                         self.torque_rp_max, self.torque_yaw_max])


# ---- Nonlinear dynamics (CasADi symbolic) --------------------------------

def build_nonlinear_dynamics(p: QuadParams) -> ca.Function:
    """Build CasADi function for continuous-time nonlinear dynamics.
    
    ẋ = f(x, u)
    
    Returns a CasADi Function: f(x[12], u[4]) -> x_dot[12].
    """
    # Symbolic variables
    x = ca.SX.sym('x', 12)
    u = ca.SX.sym('u', 4)
    
    # Unpack state
    # pos = x[0:3]   (not needed for dynamics)
    vel = x[3:6]
    phi, theta, psi = x[6], x[7], x[8]
    wx, wy, wz = x[9], x[10], x[11]
    
    # Unpack control
    T = u[0]          # thrust
    tau_x = u[1]      # roll torque
    tau_y = u[2]      # pitch torque
    tau_z = u[3]      # yaw torque
    
    # ---- Rotation matrix R_body_to_world (ZYX Euler) -----------------
    cphi = ca.cos(phi);   sphi = ca.sin(phi)
    cth  = ca.cos(theta); sth  = ca.sin(theta)
    cpsi = ca.cos(psi);   spsi = ca.sin(psi)
    
    R = ca.vertcat(
        ca.horzcat(cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi),
        ca.horzcat(spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi),
        ca.horzcat(-sth,     cth*sphi,                   cth*cphi)
    )
    
    # ---- Translational dynamics --------------------------------------
    # v̇ = R · [0; 0; T/m] + [0; 0; -g]
    thrust_body = ca.vertcat(0, 0, T / p.mass)
    thrust_world = R @ thrust_body
    gravity = ca.vertcat(0, 0, -p.g)
    
    vel_dot = thrust_world + gravity
    
    # ---- Euler rate matrix W -----------------------------------------
    # Maps body angular velocity to Euler angle rates
    # [φ̇]   [1   sinφ·tanθ   cosφ·tanθ] [ωx]
    # [θ̇] = [0   cosφ        -sinφ     ] [ωy]
    # [ψ̇]   [0   sinφ/cosθ   cosφ/cosθ ] [ωz]
    #
    # Note: singular at θ = ±π/2 (gimbal lock)
    # For small angles (quadrotor MPC), this is fine.
    
    W = ca.vertcat(
        ca.horzcat(1, sphi*sth/cth, cphi*sth/cth),
        ca.horzcat(0, cphi,         -sphi),
        ca.horzcat(0, sphi/cth,     cphi/cth)
    )
    
    omega = ca.vertcat(wx, wy, wz)
    euler_dot = W @ omega
    
    # ---- Rotational dynamics (Euler's equation) ----------------------
    # I·ω̇ = τ - ω × (I·ω)
    tau = ca.vertcat(tau_x, tau_y, tau_z)
    I_omega = ca.vertcat(p.Ixx * wx, p.Iyy * wy, p.Izz * wz)
    omega_cross_Iomega = ca.cross(omega, I_omega)
    
    omega_dot = ca.vertcat(
        (tau_x - (p.Iyy - p.Izz) * wy * wz) / p.Ixx,
        (tau_y - (p.Izz - p.Ixx) * wz * wx) / p.Iyy,
        (tau_z - (p.Ixx - p.Iyy) * wx * wy) / p.Izz
    )
    
    # ---- Full state derivative ---------------------------------------
    x_dot = ca.vertcat(
        vel,           # ṗ = v
        vel_dot,       # v̇ = R·[0;0;T/m] + g
        euler_dot,     # Euleṙ = W·ω
        omega_dot      # ω̇ = I⁻¹(τ - ω×Iω)
    )
    
    return ca.Function('f_continuous', [x, u], [x_dot],
                       ['x', 'u'], ['x_dot'])


def build_rk4_integrator(p: QuadParams, dt: float) -> ca.Function:
    """Build RK4 discrete-time integrator.
    
    x[k+1] = F(x[k], u[k])
    
    Args:
        p: quadrotor parameters
        dt: integration timestep [s]
    
    Returns:
        CasADi Function: F(x[12], u[4]) -> x_next[12].
    """
    f_cont = build_nonlinear_dynamics(p)
    
    x = ca.SX.sym('x', 12)
    u = ca.SX.sym('u', 4)
    
    # RK4 with M sub-steps for accuracy
    M = 4  # sub-steps per integration step
    h = dt / M
    
    x_next = x
    for _ in range(M):
        k1 = f_cont(x_next, u)
        k2 = f_cont(x_next + h/2 * k1, u)
        k3 = f_cont(x_next + h/2 * k2, u)
        k4 = f_cont(x_next + h * k3, u)
        x_next = x_next + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return ca.Function('F_rk4', [x, u], [x_next],
                       ['x', 'u'], ['x_next'])


# ---- Linearization at hover ----------------------------------------------

def linearize_at_hover(p: QuadParams) -> tuple:
    """Analytical linearization of quadrotor dynamics at hover.
    
    At hover: x₀ = [*, *, *, 0,0,0, 0,0,0, 0,0,0]
              u₀ = [mg, 0, 0, 0]
    
    The system decomposes into 4 decoupled subsystems:
      1. Z-altitude:  z ↔ vz ↔ δT         (double integrator)
      2. X-position:  x ↔ vx ↔ θ ↔ ωy ↔ τy (4th order chain via pitch)
      3. Y-position:  y ↔ vy ↔ φ ↔ ωx ↔ τx (4th order chain via roll)
      4. Yaw:         ψ ↔ ωz ↔ τz          (double integrator)
    
    Returns:
        Ac: [12×12] continuous-time A matrix
        Bc: [12×4]  continuous-time B matrix
    """
    g = p.g
    m = p.mass
    
    # State ordering: [px, py, pz, vx, vy, vz, φ, θ, ψ, ωx, ωy, ωz]
    #                   0   1   2   3   4   5  6  7  8   9  10  11
    
    Ac = np.zeros((12, 12))
    Bc = np.zeros((12, 4))
    
    # Position from velocity: p_dot = v.
    Ac[0, 3] = 1.0   # dx/dvx
    Ac[1, 4] = 1.0   # dy/dvy
    Ac[2, 5] = 1.0   # dz/dvz
    
    # Velocity from orientation (gravity torque coupling).
    # At hover, R ~ I + skew(euler), so:
    #   vx_dot ~  g*theta  (pitch forward -> accelerate in x)
    #   vy_dot ~ -g*phi    (roll right    -> accelerate in -y)
    #   vz_dot ~ (T-mg)/m = dT/m
    Ac[3, 7] = g      # dvx/dθ = g
    Ac[4, 6] = -g     # dvy/dφ = -g
    
    # Euler rates from angular velocity.
    # At hover (φ=θ=0): W = I, so euler_dot = omega
    Ac[6, 9]  = 1.0   # dφ/dωx
    Ac[7, 10] = 1.0   # dθ/dωy
    Ac[8, 11] = 1.0   # dψ/dωz
    
    # Control input matrix.
    # Thrust -> vz: B(vz, T) = 1/m
    Bc[5, 0] = 1.0 / m

    # Torques -> angular accelerations: B(omega_i, tau_i) = 1/Iii
    Bc[9,  1] = 1.0 / p.Ixx
    Bc[10, 2] = 1.0 / p.Iyy
    Bc[11, 3] = 1.0 / p.Izz
    
    return Ac, Bc


def discretize_dynamics(Ac: np.ndarray, Bc: np.ndarray, 
                        dt: float, method: str = 'expm') -> tuple:
    """Discretize continuous-time LTI system.
    
    x[k+1] = Ad·x[k] + Bd·u[k]
    
    Args:
        Ac: continuous A matrix [n x n]
        Bc: continuous B matrix [n x m]
        dt: timestep [s]
        method: 'expm' (exact) or 'euler' (first-order)

    Returns:
        Ad: discrete A matrix [n x n]
        Bd: discrete B matrix [n x m]
    """
    n = Ac.shape[0]
    m = Bc.shape[1]
    
    if method == 'expm':
        # Exact discretization via matrix exponential
        # [Ad Bd] = expm([Ac Bc; 0 0] * dt)[0:n, :]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = Ac
        M[:n, n:] = Bc
        M_exp = expm(M * dt)
        Ad = M_exp[:n, :n]
        Bd = M_exp[:n, n:]
    elif method == 'euler':
        # First-order: Ad = I + Ac*dt, Bd = Bc*dt
        Ad = np.eye(n) + Ac * dt
        Bd = Bc * dt
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return Ad, Bd


def compute_lqr_gain(Ad: np.ndarray, Bd: np.ndarray,
                     Q: np.ndarray, R: np.ndarray) -> tuple:
    """Compute infinite-horizon LQR gain via DARE.
    
    Used for:
      1. Terminal cost in MPC (P_inf)
      2. TinyMPC's cached Riccati matrices (K_inf)
      3. Baseline controller for comparison
    
    Returns:
        K: optimal feedback gain [m x n]
        P: solution to DARE [n x n]
    """
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


# ---- CasADi Jacobians (for nonlinear MPC / verification) -----------------

def build_jacobians(p: QuadParams) -> tuple:
    """Build CasADi functions for df/dx and df/du.
    
    Used to:
      1. Verify analytical linearization against CasADi autodiff
      2. Online re-linearization for time-varying LTV-MPC
    
    Returns:
        jac_x: CasADi Function(x, u) -> df/dx [12 x 12]
        jac_u: CasADi Function(x, u) -> df/du [12 x 4]
    """
    f_cont = build_nonlinear_dynamics(p)
    
    x = ca.SX.sym('x', 12)
    u = ca.SX.sym('u', 4)
    
    x_dot = f_cont(x, u)
    
    A_sym = ca.jacobian(x_dot, x)
    B_sym = ca.jacobian(x_dot, u)
    
    jac_x = ca.Function('jac_x', [x, u], [A_sym], ['x', 'u'], ['A'])
    jac_u = ca.Function('jac_u', [x, u], [B_sym], ['x', 'u'], ['B'])
    
    return jac_x, jac_u


# ---- Verification / printing ---------------------------------------------

def verify_linearization(p: QuadParams):
    """Cross-check analytical linearization against CasADi autodiff.
    
    This is a critical verification step. If these don't match,
    the MPC will have wrong dynamics.
    """
    print("[verify] Linearization\n")
    
    # Analytical
    Ac_analytical, Bc_analytical = linearize_at_hover(p)
    
    # CasADi autodiff at hover
    jac_x, jac_u = build_jacobians(p)
    x_hover = np.zeros(12)
    u_hover = np.array([p.hover_thrust, 0, 0, 0])
    
    Ac_casadi = np.array(jac_x(x_hover, u_hover))
    Bc_casadi = np.array(jac_u(x_hover, u_hover))
    
    # Compare
    err_A = np.max(np.abs(Ac_analytical - Ac_casadi))
    err_B = np.max(np.abs(Bc_analytical - Bc_casadi))
    
    print(f"  Max |Ac_analytical - Ac_casadi| = {err_A:.2e}")
    print(f"  Max |Bc_analytical - Bc_casadi| = {err_B:.2e}")
    
    if err_A < 1e-10 and err_B < 1e-10:
        print("  PASS: Linearizations match perfectly\n")
    else:
        print("  FAIL: Linearizations differ!\n")
        print("  Ac analytical:\n", Ac_analytical)
        print("  Ac CasADi:\n", Ac_casadi)
    
    # Print key values
    print("  Key B matrix entries (control authority):")
    print(f"    B(vz, T)        = 1/m   = {1/p.mass:.2f}")
    print(f"    B(wx, tau_x)    = 1/Ixx = {1/p.Ixx:.0f}")
    print(f"    B(wy, tau_y)    = 1/Iyy = {1/p.Iyy:.0f}")
    print(f"    B(wz, tau_z)    = 1/Izz = {1/p.Izz:.0f}")
    print(f"\n  A(vx, theta) =  g = {p.g}")
    print(f"  A(vy, phi)   = -g = {-p.g}")
    
    return Ac_analytical, Bc_analytical


def print_discrete_system(Ad: np.ndarray, Bd: np.ndarray, dt: float):
    """Pretty-print the discrete system matrices."""
    print(f"\n[discrete system] dt = {dt*1000:.0f} ms\n")

    print("  Ad (non-identity entries):")
    n = Ad.shape[0]
    state_names = ['px','py','pz','vx','vy','vz','phi','theta','psi','wx','wy','wz']
    for i in range(n):
        for j in range(n):
            val = Ad[i, j]
            if i == j:
                if abs(val - 1.0) > 1e-10:
                    print(f"    Ad[{state_names[i]},{state_names[j]}] = {val:.6f}")
            else:
                if abs(val) > 1e-10:
                    print(f"    Ad[{state_names[i]},{state_names[j]}] = {val:.6f}")

    print("\n  Bd (nonzero entries):")
    ctrl_names = ['T', 'tau_x', 'tau_y', 'tau_z']
    for i in range(Bd.shape[0]):
        for j in range(Bd.shape[1]):
            if abs(Bd[i, j]) > 1e-10:
                print(f"    Bd[{state_names[i]},{ctrl_names[j]}] = {Bd[i,j]:.6f}")


if __name__ == '__main__':
    p = QuadParams()
    
    print("[quad_dynamics] CasADi + linearization\n")
    
    print(f"  Mass: {p.mass} kg")
    print(f"  Hover thrust: {p.hover_thrust:.5f} N")
    print(f"  Inertia: [{p.Ixx:.2e}, {p.Iyy:.2e}, {p.Izz:.2e}]")
    print(f"  Thrust range: [{p.thrust_min}, {p.thrust_max}] N")
    print(f"  Torque range: +/-{p.torque_max:.1e} N*m")
    print(f"  TWR: {p.thrust_max / p.hover_thrust:.2f}")
    print()
    
    # Verify linearization
    Ac, Bc = verify_linearization(p)
    
    # Discretize
    dt_mpc = 0.02  # 50 Hz
    Ad, Bd = discretize_dynamics(Ac, Bc, dt_mpc, method='expm')
    print_discrete_system(Ad, Bd, dt_mpc)
    
    # Compute LQR gain (useful as baseline / terminal cost)
    Q = np.diag([10, 10, 10, 1, 1, 1, 5, 5, 1, 0.1, 0.1, 0.1])
    R = np.diag([100, 1e4, 1e4, 1e4])
    K, P = compute_lqr_gain(Ad, Bd, Q, R)
    print(f"\n[LQR gain 12x4]")
    print(f"  ||K|| = {np.linalg.norm(K):.4f}")
    print(f"  K eigenvalue spread: {np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K))):.4f} (should be < 1)")

    # Verify RK4 integrator consistency
    print(f"\n[RK4 integrator check]")
    F_rk4 = build_rk4_integrator(p, dt_mpc)
    x0 = np.zeros(12); x0[2] = 1.0  # hover at 1m
    u_hover = np.array([p.hover_thrust, 0, 0, 0])
    x1 = np.array(F_rk4(x0, u_hover)).flatten()
    print(f"  Hover: x0 = [..., z={x0[2]:.3f}]")
    print(f"         x1 = [..., z={x1[2]:.6f}]")
    print(f"  Drift: {np.linalg.norm(x1 - x0):.2e} (should be ~0)")
