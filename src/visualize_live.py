"""
visualize_live.py — Real-time Crazyflie Visualization with Error Display
=========================================================================
Shows reference trajectory, error metrics, and recovery behavior.

Usage:
    python3 src/visualize_live.py hover     # hover + push recovery
    python3 src/visualize_live.py fig8      # figure-8 tracking
    python3 src/visualize_live.py helix     # helical climb

What you'll see:
  - TERMINAL: live error (cm), tilt (deg), thrust, mode (LQR/RECOVERY)
  - 3D VIEWER: green sphere = reference, red line = error, blue trail = history

Perturbation (to test recovery):
  1. Double-click the drone body in the viewer
  2. Ctrl + right-click and drag to push it
  
Note: Propellers don't spin — the MJCF mesh is static. Normal for Menagerie.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))
from quad_dynamics import (QuadParams, linearize_at_hover, discretize_dynamics,
                           compute_lqr_gain)
from quad_env import (generate_figure8_reference, generate_hover_reference,
                      generate_helix_reference)


def get_state(data):
    """Extract 12D state from MuJoCo data."""
    pos = data.qpos[0:3].copy()
    vel = data.qvel[0:3].copy()
    quat = data.qpos[3:7].copy()
    omega = data.qvel[3:6].copy()
    
    w, x, y, z = quat
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    return np.concatenate([pos, vel, [roll, pitch, yaw], omega])


def try_add_geom(scn, geom_type, size, pos, mat, rgba):
    """Safely add a custom geometry to the scene."""
    try:
        idx = scn.ngeom
        if idx >= scn.maxgeom - 1:
            return False
        mujoco.mjv_initGeom(
            scn.geoms[idx],
            type=geom_type,
            size=np.array(size, dtype=np.float64),
            pos=np.array(pos, dtype=np.float64),
            mat=np.array(mat, dtype=np.float64).flatten(),
            rgba=np.array(rgba, dtype=np.float32)
        )
        scn.ngeom += 1
        return True
    except Exception:
        return False


def draw_overlays(viewer, drone_pos, ref_pos, trail, ref_traj, ref_idx):
    """Draw reference marker, error line, and trails in the 3D scene."""
    # Get the user scene handle (varies by MuJoCo version)
    scn = None
    for attr in ['user_scn', '_user_scn']:
        if hasattr(viewer, attr):
            scn = getattr(viewer, attr)
            break
    
    if scn is None:
        return  # can't draw overlays in this MuJoCo version
    
    scn.ngeom = 0  # clear previous custom geoms
    I3 = np.eye(3)
    
    # ── GREEN SPHERE at reference position ──
    try_add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                 [0.025, 0, 0], ref_pos, I3, [0, 1, 0, 0.8])
    
    # ── RED LINE from drone to reference (error vector) ──
    err_vec = ref_pos - drone_pos
    err_len = np.linalg.norm(err_vec)
    if err_len > 0.005:
        mid = (drone_pos + ref_pos) / 2
        z_axis = err_vec / err_len
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [1, 0, 0])
        x_norm = np.linalg.norm(x_axis)
        if x_norm > 1e-6:
            x_axis /= x_norm
            y_axis = np.cross(z_axis, x_axis)
            mat = np.column_stack([x_axis, y_axis, z_axis])
            try_add_geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE,
                         [0.004, err_len/2, 0], mid, mat, [1, 0.2, 0.2, 0.9])
    
    # ── BLUE TRAIL (drone history) ──
    for i, pt in enumerate(trail):
        if scn.ngeom >= scn.maxgeom - 100:
            break
        alpha = 0.1 + 0.4 * (i / max(len(trail), 1))
        try_add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                     [0.005, 0, 0], pt, I3, [0.3, 0.5, 1.0, alpha])
    
    # ── GREEN DOTS: future reference trajectory ──
    N_ref = ref_traj.shape[1]
    for k in range(ref_idx, min(N_ref, ref_idx + 100), 3):  # every 3rd point
        if scn.ngeom >= scn.maxgeom - 5:
            break
        alpha = 0.6 - 0.4 * ((k - ref_idx) / 100)
        try_add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                     [0.008, 0, 0], ref_traj[0:3, k], I3,
                     [0, 1, 0.3, max(0.05, alpha)])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fig8'
    
    p = QuadParams()
    dt_ctrl = 0.02
    dt_sim = 0.002
    n_substeps = int(dt_ctrl / dt_sim)
    
    # Build LQR
    Ac, Bc = linearize_at_hover(p)
    Ad, Bd = discretize_dynamics(Ac, Bc, dt_ctrl, method='expm')
    Q = np.diag([10, 10, 10, 1, 1, 1, 5, 5, 1, 0.1, 0.1, 0.1])
    R = np.diag([100, 1e4, 1e4, 1e4])
    K, _ = compute_lqr_gain(Ad, Bd, Q, R)
    
    # Load model
    model_path = str(Path(__file__).parent.parent / 
                     "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = dt_sim
    
    # Generate reference
    duration = 120.0
    if mode == 'hover':
        ref = generate_hover_reference(np.array([0, 0, 1.0]), duration, dt_ctrl)
        title = "HOVER — push the drone and watch it recover!"
    elif mode == 'helix':
        ref = generate_helix_reference(
            np.array([0.0, 0.0]), 0.4, 0.5, 1.5, 3.0, duration, dt_ctrl)
        title = "HELIX — watch the green reference dots"
    else:
        ref = generate_figure8_reference(
            np.array([0.0, 0.0]), 0.5, 1.0, 4.0, duration, dt_ctrl)
        title = "FIGURE-8 — red line = tracking error"
    
    N_ref = ref.shape[1]
    
    # Reset
    data.qpos[0:3] = ref[0:3, 0]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    data.ctrl[0] = p.hover_thrust
    mujoco.mj_forward(model, data)
    
    print(f"\n{'='*65}")
    print(f"  Crazyflie LQR Controller — {title}")
    print(f"{'='*65}")
    print(f"  Visual markers:")
    print(f"    GREEN sphere  = where drone SHOULD be (reference)")
    print(f"    RED line      = error vector (drone → target)")
    print(f"    BLUE dots     = where drone has BEEN (trail)")
    print(f"    GREEN dots    = upcoming reference trajectory")
    print(f"{'='*65}")
    print(f"  Push drone: double-click it → Ctrl+right-drag")
    print(f"{'='*65}\n")
    
    # Header for live telemetry
    print(f"  {'Time':>6s} │ {'Pos Err':>8s} │ {'Tilt':>6s} │ {'Thrust':>7s} │ {'Vz':>6s} │ Mode")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*10}")
    
    trail = deque(maxlen=300)
    sim_step = 0
    last_print = 0
    current_u = np.array([p.hover_thrust, 0, 0, 0])
    max_err = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = [0, 0, 0.8]
        
        while viewer.is_running():
            step_start = time.time()
            
            # ═══ CONTROL (at dt_ctrl rate) ═══
            if sim_step % n_substeps == 0:
                ctrl_idx = sim_step // n_substeps
                ref_idx = min(ctrl_idx, N_ref - 1)
                
                state = get_state(data)
                x_ref = ref[:, ref_idx]
                
                roll, pitch = state[6], state[7]
                z = state[2]
                tilt = np.sqrt(roll**2 + pitch**2)
                pos_err = np.linalg.norm(state[0:3] - x_ref[0:3])
                max_err = max(max_err, pos_err)
                
                trail.append(state[0:3].copy())
                
                # Controller
                if tilt > np.radians(60) or z < 0.08:
                    mode_str = "\033[91mRECOVERY\033[0m"  # red text
                    T = p.thrust_max * 0.95
                    kp, kd = 0.005, 0.0003
                    current_u = np.clip(np.array([
                        T,
                        -kp*roll - kd*state[9],
                        -kp*pitch - kd*state[10],
                        -kd*state[11]
                    ]), p.u_min, p.u_max)
                else:
                    mode_str = "\033[92mLQR\033[0m     "  # green text
                    x_err = state - x_ref
                    x_err[8] = ((x_err[8] + np.pi) % (2*np.pi)) - np.pi
                    delta_u = -K @ x_err
                    current_u = np.clip(np.array([
                        p.hover_thrust + delta_u[0],
                        delta_u[1], delta_u[2], delta_u[3]
                    ]), p.u_min, p.u_max)
                
                # ═══ LIVE TELEMETRY (every 0.25s) ═══
                t_now = data.time
                if t_now - last_print > 0.25:
                    err_color = "\033[91m" if pos_err > 0.10 else "\033[93m" if pos_err > 0.03 else "\033[92m"
                    print(f"  {t_now:6.1f}s │ {err_color}{pos_err*100:7.1f}cm\033[0m │ "
                          f"{np.degrees(tilt):5.1f}° │ {current_u[0]*1000:6.1f}mN │ "
                          f"{state[5]:+5.2f} │ {mode_str}")
                    last_print = t_now
            
            # Apply control
            data.ctrl[0] = current_u[0]
            data.ctrl[1] = -current_u[1] / 0.0069
            data.ctrl[2] = -current_u[2] / 0.0069
            data.ctrl[3] = -current_u[3] / 0.0036
            
            # Step physics
            mujoco.mj_step(model, data)
            sim_step += 1
            
            # ═══ RENDER + OVERLAYS (at ~30fps) ═══
            if sim_step % n_substeps == 0:
                ctrl_idx = sim_step // n_substeps
                ref_idx = min(ctrl_idx, N_ref - 1)
                
                draw_overlays(viewer, data.qpos[0:3].copy(), 
                              ref[0:3, ref_idx], trail, ref, ref_idx)
                viewer.sync()
            
            # Real-time pacing
            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    print(f"\n  {'─'*55}")
    print(f"  Session summary:")
    print(f"    Max position error: {max_err*100:.1f} cm")
    print(f"    Total time: {data.time:.1f} s")
    print(f"  Viewer closed.\n")


if __name__ == '__main__':
    main()
