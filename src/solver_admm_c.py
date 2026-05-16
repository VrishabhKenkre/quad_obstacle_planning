"""
solver_admm_c.py -- Python wrapper for the C ADMM solver.

Loads the compiled libadmm.so via ctypes and exposes the same interface
as the pure-Python ADMMSolver. The C code runs the hot loop; Python
handles dynamics setup, reference generation, and MuJoCo simulation.

Run from the repository root:
    python3 src/solver_admm_c.py [mode] [duration]
"""

import numpy as np
import ctypes
import os
import subprocess
from pathlib import Path
from scipy.linalg import solve_discrete_are
import time
from typing import Optional, Tuple


# ---- C struct definitions mirrored in Python -----------------------------

NX = 12
NU = 4
NHORIZON = 20


class ADMMCache(ctypes.Structure):
    """Mirror of the C ADMMCache struct."""
    _fields_ = [
        ("K_inf",       ctypes.c_double * (NU * NX)),
        ("P_inf",       ctypes.c_double * (NX * NX)),
        ("C1",          ctypes.c_double * (NU * NU)),
        ("C2",          ctypes.c_double * (NX * NX)),
        ("Ad",          ctypes.c_double * (NX * NX)),
        ("Bd",          ctypes.c_double * (NX * NU)),
        ("Q",           ctypes.c_double * NX),
        ("R",           ctypes.c_double * NU),
        ("P_terminal",  ctypes.c_double * NX),
        ("d",           ctypes.c_double * NX),
        ("u_hover",     ctypes.c_double * NU),
        ("x_min",       ctypes.c_double * NX),
        ("x_max",       ctypes.c_double * NX),
        ("u_min",       ctypes.c_double * NU),
        ("u_max",       ctypes.c_double * NU),
        ("rho",         ctypes.c_double),
    ]


class ADMMVars(ctypes.Structure):
    """Mirror of the C ADMMVars struct."""
    _fields_ = [
        ("x",       ctypes.c_double * (NX * (NHORIZON + 1))),
        ("u",       ctypes.c_double * (NU * NHORIZON)),
        ("zx",      ctypes.c_double * (NX * (NHORIZON + 1))),
        ("zu",      ctypes.c_double * (NU * NHORIZON)),
        ("yx",      ctypes.c_double * (NX * (NHORIZON + 1))),
        ("yu",      ctypes.c_double * (NU * NHORIZON)),
        ("p",       ctypes.c_double * (NX * (NHORIZON + 1))),
        ("d_ctrl",  ctypes.c_double * (NU * NHORIZON)),
    ]


class ADMMStats(ctypes.Structure):
    """Mirror of the C ADMMStats struct."""
    _fields_ = [
        ("iterations",      ctypes.c_int),
        ("primal_residual", ctypes.c_double),
        ("dual_residual",   ctypes.c_double),
        ("solve_time_us",   ctypes.c_double),
    ]


def _compile_if_needed(src_dir: Path) -> Path:
    """Compile admm_core.c -> libadmm.so if not already compiled."""
    so_path = src_dir / "libadmm.so"
    c_path = src_dir / "admm_core.c"
    h_path = src_dir / "admm_core.h"
    
    # Recompile if .so doesn't exist or is older than source
    needs_compile = not so_path.exists()
    if not needs_compile:
        so_mtime = so_path.stat().st_mtime
        if c_path.stat().st_mtime > so_mtime or h_path.stat().st_mtime > so_mtime:
            needs_compile = True
    
    if needs_compile:
        print("  Compiling admm_core.c -> libadmm.so ...")
        result = subprocess.run(
            ["gcc", "-O3", "-shared", "-fPIC", "-o", str(so_path),
             str(c_path), "-lm"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")
        print(f"  Compiled ({so_path.stat().st_size} bytes)")
    
    return so_path


class CADMMSolver:
    """Python interface to the C ADMM solver.
    
    Same API as the pure Python ADMMSolver, but the hot loop
    runs in compiled C. Expected 10-50x speedup over Python.
    """
    
    def __init__(self, Ad: np.ndarray, Bd: np.ndarray,
                 Q_diag: np.ndarray, R_diag: np.ndarray,
                 N: int,
                 u_min: np.ndarray, u_max: np.ndarray,
                 x_min: np.ndarray, x_max: np.ndarray,
                 u_hover: np.ndarray,
                 gravity_offset: np.ndarray,
                 rho: float = 1.0,
                 max_iter: int = 50,
                 eps_abs: float = 1e-4):
        
        assert Ad.shape == (NX, NX)
        assert Bd.shape == (NX, NU)
        assert len(Q_diag) == NX
        assert len(R_diag) == NU
        
        self.max_iter = max_iter
        self.eps_abs = eps_abs
        self.u_hover = u_hover
        self.rho = rho
        
        # Compile and load the C library
        src_dir = Path(__file__).parent
        so_path = _compile_if_needed(src_dir)
        self.lib = ctypes.CDLL(str(so_path))
        
        # Set function signatures
        self.lib.admm_init.argtypes = [ctypes.POINTER(ADMMVars)]
        self.lib.admm_init.restype = None
        
        self.lib.admm_solve.argtypes = [
            ctypes.POINTER(ADMMCache),      # cache
            ctypes.POINTER(ADMMVars),       # vars
            ctypes.POINTER(ctypes.c_double),  # x0
            ctypes.POINTER(ctypes.c_double),  # x_ref
            ctypes.POINTER(ctypes.c_double),  # u_ref
            ctypes.c_int,                     # max_iter
            ctypes.c_double,                  # eps_abs
            ctypes.POINTER(ctypes.c_double),  # u_out
            ctypes.POINTER(ADMMStats),        # stats
        ]
        self.lib.admm_solve.restype = None
        
        self.lib.admm_warm_shift.argtypes = [ctypes.POINTER(ADMMVars)]
        self.lib.admm_warm_shift.restype = None
        
        # Compute Riccati matrices in Python (this is the offline step)
        Q_aug = np.diag(Q_diag) + rho * np.eye(NX)
        R_aug = np.diag(R_diag) + rho * np.eye(NU)
        
        P_inf = solve_discrete_are(Ad, Bd, Q_aug, R_aug)
        BtPB = Bd.T @ P_inf @ Bd
        BtPA = Bd.T @ P_inf @ Ad
        K_coeff = R_aug + BtPB
        K_inf = np.linalg.solve(K_coeff, BtPA)
        C1 = np.linalg.inv(K_coeff)
        C2 = (Ad - Bd @ K_inf).T
        
        # Terminal cost (DARE without rho augmentation)
        P_terminal = solve_discrete_are(Ad, Bd, np.diag(Q_diag), np.diag(R_diag))
        P_terminal_diag = np.diag(P_terminal)  # simplified: use diagonal only
        
        # Fill the C cache struct
        self.cache = ADMMCache()
        self._fill_array(self.cache.K_inf, K_inf.flatten())
        self._fill_array(self.cache.P_inf, P_inf.flatten())
        self._fill_array(self.cache.C1, C1.flatten())
        self._fill_array(self.cache.C2, C2.flatten())
        self._fill_array(self.cache.Ad, Ad.flatten())
        self._fill_array(self.cache.Bd, Bd.flatten())
        self._fill_array(self.cache.Q, Q_diag)
        self._fill_array(self.cache.R, R_diag)
        self._fill_array(self.cache.P_terminal, P_terminal_diag)
        self._fill_array(self.cache.d, gravity_offset)
        self._fill_array(self.cache.u_hover, u_hover)
        self._fill_array(self.cache.x_min, x_min)
        self._fill_array(self.cache.x_max, x_max)
        self._fill_array(self.cache.u_min, u_min)
        self._fill_array(self.cache.u_max, u_max)
        self.cache.rho = rho
        
        # Initialize ADMM variables
        self.vars = ADMMVars()
        self.lib.admm_init(ctypes.byref(self.vars))
        
        # Statistics
        self.solve_times = []
        self.iterations_log = []
    
    @staticmethod
    def _fill_array(c_array, np_array):
        """Copy numpy array into a ctypes fixed-size array."""
        for i, val in enumerate(np_array.flat):
            c_array[i] = float(val)
    
    def solve(self, x0: np.ndarray, x_ref: np.ndarray,
              u_ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """Solve the MPC QP using the C ADMM solver.
        
        Args:
            x0: current state [NX]
            x_ref: reference trajectory [NX x (N+1)], column-major
            u_ref: reference controls [NU x N], column-major (default: u_hover)

        Returns:
            u_opt: optimal first control [NU]
            info: solve statistics
        """
        if u_ref is None:
            u_ref = np.tile(self.u_hover, (NHORIZON, 1)).T  # NU × N
        
        # Prepare C-compatible arrays (column-major = Fortran order for trajectories)
        # Our convention: x_ref[:, k] = x_ref[k*NX : (k+1)*NX]
        # numpy column k of x_ref (shape NX x N+1) -> contiguous in Fortran order
        x_ref_flat = np.ascontiguousarray(x_ref.T.flatten())  # (N+1)*NX
        u_ref_flat = np.ascontiguousarray(u_ref.T.flatten())   # N*NU
        x0_flat = np.ascontiguousarray(x0.flatten())
        
        u_out = np.zeros(NU)
        stats = ADMMStats()
        
        # Call C solver
        self.lib.admm_solve(
            ctypes.byref(self.cache),
            ctypes.byref(self.vars),
            x0_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x_ref_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            u_ref_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.max_iter),
            ctypes.c_double(self.eps_abs),
            u_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(stats),
        )
        
        self.solve_times.append(stats.solve_time_us / 1e6)  # convert to seconds
        self.iterations_log.append(stats.iterations)
        
        # Extract predicted trajectory from C vars
        x_pred = np.array(self.vars.x[:]).reshape(NHORIZON + 1, NX).T
        u_seq = np.array(self.vars.u[:]).reshape(NHORIZON, NU).T
        
        info = {
            'status': 'solved' if stats.iterations < self.max_iter else 'max_iter',
            'solve_time': stats.solve_time_us / 1e6,
            'solve_time_us': stats.solve_time_us,
            'iterations': stats.iterations,
            'primal_residual': stats.primal_residual,
            'dual_residual': stats.dual_residual,
            'x_pred': x_pred,
            'u_seq': u_seq,
        }
        
        return u_out, info
    
    def warm_shift(self):
        """Shift ADMM variables by one step for warm starting."""
        self.lib.admm_warm_shift(ctypes.byref(self.vars))


def run_c_admm_sim(mode='fig8', duration=10.0):
    """Run the C ADMM solver in closed loop with MuJoCo."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from quad_env import CrazyflieEnv, generate_figure8_reference, \
                         generate_hover_reference, generate_helix_reference, \
                         generate_step_response
    from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
    
    p = QuadParams()
    dt = 0.02
    N = NHORIZON
    
    Ac, Bc = linearize_at_hover(p)
    Ad, Bd = discretize_dynamics(Ac, Bc, dt, method='expm')
    
    Q_diag = np.array([10, 10, 10, 1, 1, 1, 5, 5, 1, 0.1, 0.1, 0.1])
    R_diag = np.array([100, 1e4, 1e4, 1e4])
    
    x_hover = np.zeros(12)
    u_hover = np.array([p.hover_thrust, 0, 0, 0])
    d = (np.eye(12) - Ad) @ x_hover - Bd @ u_hover
    
    INF = 1e10
    x_min = np.array([-INF]*3 + [-INF]*3 + [-np.radians(30), -np.radians(30), -INF] + [-INF]*3)
    x_max = np.array([INF]*3 + [INF]*3 + [np.radians(30), np.radians(30), INF] + [INF]*3)
    
    model_path = str(Path(__file__).parent.parent /
                     "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml")
    env = CrazyflieEnv(model_path=model_path, dt_sim=0.002, dt_ctrl=dt)
    
    total_steps = int(duration / dt)
    ref_duration = duration + N * dt + 1.0
    
    if mode == 'hover':
        ref = generate_hover_reference(np.array([0, 0, 1.0]), ref_duration, dt)
        title = "Hover"
    elif mode == 'fig8':
        ref = generate_figure8_reference(
            np.array([0.0, 0.0]), 0.5, 1.0, 4.0, ref_duration, dt)
        title = "Figure-8"
    elif mode == 'helix':
        ref = generate_helix_reference(
            np.array([0.0, 0.0]), 0.4, 0.5, 1.5, 3.0, ref_duration, dt)
        title = "Helix"
    else:
        ref = generate_step_response(
            np.array([0, 0, 1.0]), np.array([0.5, 0.3, 1.2]),
            1.0, ref_duration, dt)
        title = "Step"
    
    N_ref = ref.shape[1]
    
    print(f"\n[C ADMM] {title} -- horizon {N}, dt={dt*1000:.0f}ms, gcc -O3\n")
    
    solver = CADMMSolver(Ad, Bd, Q_diag, R_diag, N,
                         p.u_min, p.u_max, x_min, x_max,
                         u_hover, d, rho=1.0, max_iter=50, eps_abs=1e-4)
    
    x = env.reset(pos=ref[0:3, 0])
    x_log = np.zeros((12, total_steps + 1))
    u_log = np.zeros((4, total_steps))
    x_log[:, 0] = x
    
    print(f"  {'Time':>6s} | {'PosErr':>8s} | {'Thrust':>7s} | {'Solve':>9s} | {'Iters':>5s} | Status")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}-+-{'-'*5}-+-{'-'*10}")
    
    for i in range(total_steps):
        ref_window = np.zeros((12, N + 1))
        for k in range(N + 1):
            ref_window[:, k] = ref[:, min(i + k, N_ref - 1)]
        
        u_opt, info = solver.solve(x_log[:, i], ref_window)
        x_log[:, i + 1] = env.step(u_opt)
        u_log[:, i] = u_opt
        solver.warm_shift()
        
        if (i + 1) % int(1.0 / dt) == 0:
            t_now = (i + 1) * dt
            pos_err = np.linalg.norm(x_log[0:3, i+1] - ref[0:3, min(i+1, N_ref-1)])
            solve_us = info['solve_time_us']
            print(f"  {t_now:5.1f}s | {pos_err*100:7.2f}cm | "
                  f"{u_opt[0]*1000:6.1f}mN | "
                  f"{solve_us:7.1f} us | "
                  f"{info['iterations']:5d} | {info['status']}")
    
    # Results
    tracking_err = np.linalg.norm(
        x_log[0:3, :total_steps] - ref[0:3, :total_steps], axis=0)
    rmse = np.sqrt(np.mean(tracking_err**2))
    solve_times_us = np.array(solver.solve_times) * 1e6
    iters = np.array(solver.iterations_log)
    
    print(f"\n  Results (C ADMM with gcc -O3):")
    print(f"    RMSE tracking error:  {rmse*100:.2f} cm")
    print(f"    Avg solve time:       {np.mean(solve_times_us):.1f} us")
    print(f"    Median solve time:    {np.median(solve_times_us):.1f} us")
    print(f"    Max solve time:       {np.max(solve_times_us):.1f} us")
    print(f"    Avg iterations:       {np.mean(iters):.1f}\n")
    
    return x_log, u_log, ref, solver, tracking_err


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fig8'
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    run_c_admm_sim(mode=mode, duration=duration)
