# Quadrotor Obstacle Planning & Imitation Learning

Hierarchical A* + minimum-snap + NMPC planner that beats reactive learned policies on quadrotor obstacle avoidance. Plus DAgger+DART imitation learning of MPC controllers for distilled fast inference.

## Headline result

| Controller                          | Goal err        | Max field         | Speed   |
|-------------------------------------|-----------------|-------------------|---------|
| **Hierarchical Planner (ours)**     | **16 ± 14 mm**  | **0.046 ± 0.025** | 23 ms   |
| NMPC teacher                        | 46 ± 10 mm      | 0.106 ± 0.040     | 59 ms   |
| DAgger+DART student (SDF reactive)  | 420 ± 204 mm    | 0.475 ± 0.273     | 91 µs   |
| Blind linear MPC                    | 49 mm           | 0.823 ± 0.384     | 77 µs   |

Hierarchical planner: 26× lower goal error than the reactive student, 3× lower than the NMPC teacher. 10-seed evaluation, MuJoCo simulation, Bitcraze Crazyflie 2 platform.

🎥 [Demo video](videos/planner_demo.mp4)

## Quickstart

```bash
conda activate quad_mpc
python3 scripts/verify_planner_16mm.py     # confirms the 16mm result
python3 planning/eval_planner.py            # runs the full 10-seed eval
```

## What's in this repo
planning/                  Hierarchical A* + min-snap + NMPC stack
voxelize.py                  occupancy grid + ESDF
astar.py                     3D A* on the voxel grid
min_snap.py                  minimum-snap polynomial smoothing
hierarchical_ctrl.py         planner-NMPC integration
eval_planner.py              10-seed eval harness
record_demo.py               video recording
tests/                       pytest suite (17/17 passing)
src/                       Crazyflie dynamics, MuJoCo env, NMPC, DAgger
quad_dynamics.py             Crazyflie 12-state linearization
quad_env.py                  MuJoCo Crazyflie env wrapper
nonlinear_mpc.py             SE(3) NMPC with CasADi
obstacle_course.py           Gaussian obstacle field generator
dagger.py                    DAgger+DART distillation pipeline
policy_inference.{c,h}       distilled-policy C inference
mujoco_menagerie/          Bitcraze Crazyflie 2 model (with actuator
corrections; see Known Issues)
results/                   10-seed JSONs + verification plots
videos/                    Demo videos
scripts/                   Numerical verification scripts

## Reproducing the result

```bash
# build the C solver (used by the planner for fast NMPC tracking)
gcc -O3 -ffast-math -march=native -shared -fPIC \
    src/admm_core.c -o src/libadmm.so -lm

# 10-seed eval (runs in ~2 minutes)
python3 planning/eval_planner.py

# verify the headline numbers
python3 scripts/verify_planner_16mm.py
python3 scripts/verify_planner_clearance.py
```

## Hardware notes

All numbers measured on a single core of an Intel i9-14900HX laptop CPU.
The NMPC tracker uses a CasADi/IPOPT solver; expect ~1.5× slower on
older laptop CPUs. No GPU required for the planner; the IL pipeline in
`src/dagger.py` uses PyTorch with CUDA if available.

## Citation

```bibtex
@misc{kenkre2026quadplan,
  title = {Hierarchical Planning and Reactive Distillation for
           Quadrotor Obstacle Avoidance},
  author = {Kenkre, Vrishabh},
  year = {2026},
  note = {Manuscript in preparation.}
}
```

## License

MIT for code in `planning/`, `src/`, `scripts/`. The Bitcraze Crazyflie
MuJoCo model in `mujoco_menagerie/` is from MuJoCo Menagerie under
Apache 2.0; the actuator corrections (see Known Issues) are derived
from manufacturer-rated values and remain under Apache 2.0.

CMU MS MechE Robotics, 2026.
