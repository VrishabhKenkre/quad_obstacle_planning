"""quad_env.py -- MuJoCo Crazyflie 2 simulation wrapper.

Wraps the Menagerie bitcraze_crazyflie_2 model with both a raw simulation
interface (for MPC) and helpers for building reference trajectories.
"""

import numpy as np
import mujoco
from pathlib import Path
from typing import Optional, Tuple, Dict
from scipy.spatial.transform import Rotation

# ---- Physical constants --------------------------------------------------

MASS = 0.027                    # kg
G = 9.81                        # m/s²
HOVER_THRUST = MASS * G         # 0.26487 N
IXX = IYY = 2.3951e-5           # kg·m²
IZZ = 3.2347e-5                 # kg·m²

# Actuator limits
THRUST_MIN = 0.0                # N
THRUST_MAX = 0.60               # N (TWR ~ 2.27, realistic for CF2)
TORQUE_RP_MAX = 0.0069          # N·m roll/pitch (arm × motor_thrust = 0.046 × 0.15)
TORQUE_YAW_MAX = 0.0036         # N·m yaw (reactive torque coefficient)

# Derived
THRUST_TO_WEIGHT = THRUST_MAX / HOVER_THRUST  # ~ 1.32


class CrazyflieEnv:
    """Raw MuJoCo simulation interface for Crazyflie 2.
    
    State representation for MPC (12D, Euler angles):
        x = [px, py, pz, vx, vy, vz, φ, θ, ψ, ωx, ωy, ωz]
    
    Control input (4D):
        u = [thrust, τx, τy, τz]
        thrust ∈ [0, 0.35] N
        τi ∈ [-1e-5, 1e-5] N·m
    
    The MuJoCo model uses quaternions internally. This class
    provides conversion to/from Euler angles for the MPC.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 dt_sim: float = 0.002,
                 dt_ctrl: float = 0.02):
        """
        Args:
            model_path: Path to scene.xml (auto-detected if None)
            dt_sim: MuJoCo simulation timestep [s]
            dt_ctrl: Control timestep [s] (must be multiple of dt_sim)
        """
        if model_path is None:
            # Auto-find the model relative to this file
            base = Path(__file__).parent.parent
            model_path = str(base / "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Override simulation timestep
        self.model.opt.timestep = dt_sim
        self.dt_sim = dt_sim
        self.dt_ctrl = dt_ctrl
        self.n_substeps = max(1, int(dt_ctrl / dt_sim))
        
        # State/control dimensions
        self.nx = 12   # [pos(3), vel(3), euler(3), omega(3)]
        self.nu = 4    # [thrust, τx, τy, τz]
        
        # Control limits
        self.u_min = np.array([THRUST_MIN, -TORQUE_RP_MAX, -TORQUE_RP_MAX, -TORQUE_YAW_MAX])
        self.u_max = np.array([THRUST_MAX, TORQUE_RP_MAX, TORQUE_RP_MAX, TORQUE_YAW_MAX])
        
        # Hover control
        self.u_hover = np.array([HOVER_THRUST, 0.0, 0.0, 0.0])
        
        # Reset to hover
        self.reset()
    
    def reset(self, pos: Optional[np.ndarray] = None,
              vel: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset to hover keyframe or specified state.
        
        Args:
            pos: [x, y, z] initial position (default: [0, 0, 1.0])
            vel: [vx, vy, vz] initial velocity (default: zeros)
        
        Returns:
            12D state vector
        """
        mujoco.mj_resetData(self.model, self.data)
        
        if pos is not None:
            self.data.qpos[0:3] = pos
        else:
            self.data.qpos[0:3] = [0.0, 0.0, 1.0]  # 1m hover height
        
        # Identity quaternion (upright)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        
        # Zero velocities (or specified)
        self.data.qvel[:] = 0.0
        if vel is not None:
            self.data.qvel[0:3] = vel
        
        # Apply hover thrust
        self.data.ctrl[:] = self._u_to_ctrl(self.u_hover)
        
        mujoco.mj_forward(self.model, self.data)
        
        return self.get_state()
    
    def step(self, u: np.ndarray) -> np.ndarray:
        """Apply control and simulate forward by dt_ctrl.
        
        Args:
            u: [thrust, τx, τy, τz] in physical units
               thrust ∈ [0, 0.35] N
               τi ∈ [-1e-5, 1e-5] N·m
        
        Returns:
            12D state vector after stepping
        """
        # Clip to actuator limits
        u_clipped = np.clip(u, self.u_min, self.u_max)
        
        # Convert to MuJoCo ctrl format
        self.data.ctrl[:] = self._u_to_ctrl(u_clipped)
        
        # Step simulation (multiple substeps for higher fidelity)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get 12D state: [pos(3), vel(3), euler_rpy(3), omega(3)].
        
        Note: MuJoCo stores quat as [w,x,y,z]. We convert to
        Euler angles [roll(φ), pitch(θ), yaw(ψ)] for the MPC.
        Angular velocity is in the BODY frame.
        """
        pos = self.data.qpos[0:3].copy()
        vel = self.data.qvel[0:3].copy()     # world frame linear velocity
        quat = self.data.qpos[3:7].copy()    # [w, x, y, z]
        omega = self.data.qvel[3:6].copy()   # body frame angular velocity
        
        # Quaternion -> Euler (ZYX convention: yaw-pitch-roll)
        euler = self._quat_to_euler(quat)
        
        return np.concatenate([pos, vel, euler, omega])
    
    def get_time(self) -> float:
        """Current simulation time [s]."""
        return self.data.time
    
    # ---- Internal helpers --------------------------------------------
    
    def _u_to_ctrl(self, u: np.ndarray) -> np.ndarray:
        """Convert physical units to MuJoCo ctrl.
        
        MuJoCo actuators: force = gear * ctrl
          body_thrust: gear=+1,    ctrl=[0, 0.35] -> force [0, 0.35] N
          x_moment:    gear=-1e-5, ctrl=[-1, 1]   -> torque [-1e-5, 1e-5] N*m
        
        So: ctrl[0] = thrust (direct, gear=+1)
            ctrl[1:4] = torque / gear = torque / (-1e-5) = -torque * 1e5
        
        CRITICAL: the negative gear was causing sign-inverted feedback!
        """
        ctrl = np.zeros(4)
        ctrl[0] = u[0]                        # thrust (gear = +1)
        ctrl[1] = -u[1] / 0.0069             # tau_x -> ctrl (gear = -0.0069)
        ctrl[2] = -u[2] / 0.0069             # tau_y -> ctrl (gear = -0.0069)
        ctrl[3] = -u[3] / 0.0036             # tau_z -> ctrl (gear = -0.0036)
        return ctrl
    
    @staticmethod
    def _quat_to_euler(q: np.ndarray) -> np.ndarray:
        """Quaternion [w,x,y,z] -> Euler [roll, pitch, yaw] (ZYX).

        Standard aerospace convention:
          roll  (phi)   = rotation about x
          pitch (theta) = rotation about y
          yaw   (psi)   = rotation about z
        """
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    @staticmethod
    def euler_to_quat(rpy: np.ndarray) -> np.ndarray:
        """Euler [roll, pitch, yaw] -> Quaternion [w,x,y,z] (ZYX)."""
        phi, theta, psi = rpy
        
        cy, sy = np.cos(psi/2), np.sin(psi/2)
        cp, sp = np.cos(theta/2), np.sin(theta/2)
        cr, sr = np.cos(phi/2), np.sin(phi/2)
        
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        
        return np.array([w, x, y, z])


# ---- Trajectory generators -----------------------------------------------

def generate_hover_reference(target: np.ndarray, duration: float, 
                              dt: float) -> np.ndarray:
    """Constant hover at target position.
    
    Args:
        target: [x, y, z] target position
        duration: total time [s]
        dt: timestep [s]
    
    Returns:
        ref_traj: [12 x N] reference trajectory
    """
    N = int(duration / dt)
    ref = np.zeros((12, N))
    ref[0:3, :] = target[:, np.newaxis]
    return ref


def generate_figure8_reference(center: np.ndarray, radius: float,
                                height: float, period: float,
                                duration: float, dt: float) -> np.ndarray:
    """Figure-8 trajectory in the XY plane at constant height.

    Parametric: x(t) = r*sin(omega*t), y(t) = r*sin(2*omega*t)/2
    
    Args:
        center: [x0, y0] center of figure-8
        radius: amplitude [m]
        height: z altitude [m]
        period: time for one complete figure-8 [s]
        duration: total time [s]
        dt: timestep [s]
    """
    N = int(duration / dt)
    t = np.arange(N) * dt
    omega = 2 * np.pi / period
    
    ref = np.zeros((12, N))
    # Position
    ref[0, :] = center[0] + radius * np.sin(omega * t)
    ref[1, :] = center[1] + radius * np.sin(2 * omega * t) / 2
    ref[2, :] = height
    
    # Velocity (analytical derivative)
    ref[3, :] = radius * omega * np.cos(omega * t)
    ref[4, :] = radius * omega * np.cos(2 * omega * t)
    ref[5, :] = 0.0
    
    return ref


def generate_helix_reference(center: np.ndarray, radius: float,
                              z_start: float, z_end: float,
                              period: float, duration: float,
                              dt: float) -> np.ndarray:
    """Helical trajectory -- circle in XY with linear Z climb.
    
    Args:
        center: [x0, y0] circle center
        radius: circle radius [m]
        z_start, z_end: altitude range [m]
        period: time per revolution [s]
        duration: total time [s]
        dt: timestep [s]
    """
    N = int(duration / dt)
    t = np.arange(N) * dt
    omega = 2 * np.pi / period
    
    ref = np.zeros((12, N))
    ref[0, :] = center[0] + radius * np.cos(omega * t)
    ref[1, :] = center[1] + radius * np.sin(omega * t)
    ref[2, :] = z_start + (z_end - z_start) * t / duration
    
    ref[3, :] = -radius * omega * np.sin(omega * t)
    ref[4, :] = radius * omega * np.cos(omega * t)
    ref[5, :] = (z_end - z_start) / duration
    
    return ref


def generate_step_response(start: np.ndarray, end: np.ndarray,
                            step_time: float, duration: float,
                            dt: float) -> np.ndarray:
    """Step response: jump from start to end position at step_time.
    
    Args:
        start: [x, y, z] initial position
        end: [x, y, z] target position after step
        step_time: when the step occurs [s]
        duration: total time [s]
        dt: timestep [s]
    """
    N = int(duration / dt)
    t = np.arange(N) * dt
    
    ref = np.zeros((12, N))
    for i in range(N):
        if t[i] < step_time:
            ref[0:3, i] = start
        else:
            ref[0:3, i] = end
    
    return ref
