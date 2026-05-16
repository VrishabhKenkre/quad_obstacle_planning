/*
 * policy_inference.c -- DAgger-distilled MLP forward pass.
 *
 * Architecture: 20 -> 64 -> 64 -> 4, ReLU x2, tanh out, no LayerNorm.
 *
 * Build: gcc -O3 -ffast-math -march=native -fPIC -shared
 *            policy_inference.c -o libpolicy_inference.so -lm
 */

#include "policy_inference.h"
#include "policy_weights.h"
#include <math.h>
#include <stddef.h>

/* Linear + ReLU. W is stored row-major (out_dim, in_dim). */
static inline void linear_relu(const float *x,
                               const float *W, const float *b,
                               float *y, int in_dim, int out_dim)
{
    for (int j = 0; j < out_dim; ++j) {
        float s = b[j];
        const float *Wj = W + (size_t)j * in_dim;
        for (int i = 0; i < in_dim; ++i) {
            s += Wj[i] * x[i];
        }
        y[j] = s > 0.0f ? s : 0.0f;
    }
}

/* Linear + tanh, final action layer. */
static inline void linear_tanh(const float *x,
                               const float *W, const float *b,
                               float *y, int in_dim, int out_dim)
{
    for (int j = 0; j < out_dim; ++j) {
        float s = b[j];
        const float *Wj = W + (size_t)j * in_dim;
        for (int i = 0; i < in_dim; ++i) {
            s += Wj[i] * x[i];
        }
        y[j] = tanhf(s);
    }
}

/* Public API. Stack: 64 + 64 = 128 floats = 512 bytes. */
void policy_forward(const float *obs, float *action)
{
    float h1[POLICY_HIDDEN];
    float h2[POLICY_HIDDEN];

    linear_relu(obs, W1, B1, h1, POLICY_OBS_DIM, POLICY_HIDDEN);
    linear_relu(h1,  W2, B2, h2, POLICY_HIDDEN,   POLICY_HIDDEN);
    linear_tanh(h2,  W3, B3, action, POLICY_HIDDEN, POLICY_ACT_DIM);
}
