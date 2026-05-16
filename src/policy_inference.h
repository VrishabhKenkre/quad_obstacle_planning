/*
 * policy_inference.h -- plain-C forward pass for the DAgger-distilled
 * quadrotor MPC student.
 *
 * Architecture (from src/dagger.py PolicyNet):
 *
 *     obs (20) -> Linear(20,64) -> ReLU
 *              -> Linear(64,64) -> ReLU
 *              -> Linear(64,4)  -> tanh -> action (4, in [-1,1])
 *
 * Total parameters: 5,764
 * No LayerNorm. No batch norm. No dropout (eval mode).
 *
 * Observation layout (20-D):
 *   [0:12]  : full state  [px,py,pz, vx,vy,vz, roll,pitch,yaw, wx,wy,wz]
 *   [12:15] : tracking error  [ex,ey,ez] = state[0:3] - ref[0:3]
 *   [15:18] : reference velocity [vrx,vry,vrz]
 *   [18:20] : reference attitude [roll_ref, pitch_ref]
 *
 * Action output (4-D, in [-1,1]):
 *   Apply env action_to_control mapping:
 *     u_phys = u_mid + u_half * action
 *   where u_mid = (u_max + u_min)/2 and u_half = (u_max - u_min)/2.
 */

#ifndef POLICY_INFERENCE_H
#define POLICY_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#define POLICY_OBS_DIM 20
#define POLICY_ACT_DIM 4
#define POLICY_HIDDEN  64

/* Forward pass. Thread-safe. */
void policy_forward(const float *obs, float *action);

#ifdef __cplusplus
}
#endif

#endif /* POLICY_INFERENCE_H */
