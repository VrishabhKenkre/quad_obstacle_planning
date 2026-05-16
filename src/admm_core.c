/*
 * admm_core.c -- hand-rolled ADMM solver implementation.
 *
 * Matrices are row-major flat arrays (M[i*ncols + j]); trajectory arrays are
 * column-major by timestep (x[k*NX + i] is the i-th state at time k).
 *
 * Compile: gcc -O3 -shared -fPIC -o libadmm.so admm_core.c -lm
 */

#include "admm_core.h"
#include <math.h>
#include <string.h>
#include <time.h>

/* ---- Low-level linear algebra --------- */

/**
 * y = A * x,  where A is [rows × cols], x is [cols], y is [rows]
 * A is row-major: A[i*cols + j]
 */
static void matvec(const double *A, const double *x, double *y,
                   int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += A[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

/**
 * y = A^T * x,  where A is [rows × cols], x is [rows], y is [cols]
 * Transpose multiply without forming A^T explicitly.
 */
static void matvec_transpose(const double *A, const double *x, double *y,
                             int rows, int cols)
{
    for (int j = 0; j < cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            sum += A[i * cols + j] * x[i];
        }
        y[j] = sum;
    }
}

/**
 * y = a + b  (element-wise, length n)
 */
static void vec_add(const double *a, const double *b, double *y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = a[i] + b[i];
    }
}

/**
 * y = a - b  (element-wise, length n)
 */
static void vec_sub(const double *a, const double *b, double *y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = a[i] - b[i];
    }
}

/**
 * y = a + alpha * b  (element-wise, length n)
 */
static void vec_add_scaled(const double *a, double alpha, const double *b,
                           double *y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = a[i] + alpha * b[i];
    }
}

/**
 * ||x||_2  (Euclidean norm, length n)
 */
static double vec_norm(const double *x, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

/**
 * y = clamp(x, lo, hi)  (element-wise, length n)
 */
static void vec_clamp(const double *x, const double *lo, const double *hi,
                      double *y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
        if (y[i] < lo[i]) y[i] = lo[i];
        if (y[i] > hi[i]) y[i] = hi[i];
    }
}

/**
 * Copy n doubles from src to dst
 */
static void vec_copy(const double *src, double *dst, int n)
{
    memcpy(dst, src, n * sizeof(double));
}

/**
 * Set n doubles to zero
 */
static void vec_zero(double *x, int n)
{
    memset(x, 0, n * sizeof(double));
}

/**
 * y[i] = diag[i] * x[i]  (diagonal matrix multiply, length n)
 * When Q, R are diagonal, this replaces full matvec.
 */
static void diag_vec_mul(const double *diag, const double *x, double *y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = diag[i] * x[i];
    }
}

/* ---- ADMM steps --------- */

/**
 * Backward pass: compute affine cost-to-go p[] and control corrections d_ctrl[].
 *
 * This is the "linear Riccati" backward sweep. The QUADRATIC part (matrices)
 * is cached in P_inf, K_inf, C1, C2. Only the LINEAR part (vectors p, d_ctrl)
 * needs to be recomputed because the reference trajectory and ADMM
 * slack/dual variables change each iteration.
 *
 * For k = N-1 down to 0:
 *   p_next_adj = p[k+1] + P_inf @ d          (gravity offset)
 *   d_ctrl[k]  = C1 @ (Bd^T @ p_next_adj + r_tilde[k])
 *   p[k]       = q_tilde[k] + C2 @ p_next_adj - K_inf^T @ r_tilde[k]
 *
 * where:
 *   q_tilde[k] = -Q @ xref[k] + rho * (yx[k] - zx[k])
 *   r_tilde[k] = -R @ uref[k] + rho * (yu[k] - zu[k])
 */
static void backward_pass(const ADMMCache *c, ADMMVars *v,
                          const double *x_ref, const double *u_ref)
{
    double q_tilde[NX], r_tilde[NU];
    double p_next_adj[NX];
    double Bt_p[NU];       /* Bd^T @ p_next_adj */
    double Bt_p_plus_r[NU];
    double C2_p[NX];
    double Kt_r[NX];
    double Pinf_d[NX];     /* P_inf @ gravity_offset */
    
    /* Precompute P_inf @ d (constant across all k) */
    matvec(c->P_inf, c->d, Pinf_d, NX, NX);
    
    /* Terminal: p[N] = q_tilde[N] */
    /* q_tilde[N] = -P_inf * xref[N] + rho * (yx[N] - zx[N]) */
    /* Using full P_inf matrix (not diagonal approximation) */
    int kN = NHORIZON;
    double Pinf_xref[NX];
    matvec(c->P_inf, &x_ref[kN * NX], Pinf_xref, NX, NX);
    for (int i = 0; i < NX; i++) {
        q_tilde[i] = -Pinf_xref[i]
                    + c->rho * (v->yx[kN * NX + i] - v->zx[kN * NX + i]);
    }
    vec_copy(q_tilde, &v->p[kN * NX], NX);
    
    /* Backward sweep */
    for (int k = NHORIZON - 1; k >= 0; k--) {
        /* Compute q_tilde[k] = -Q * xref[k] + rho * (yx[k] - zx[k]) */
        for (int i = 0; i < NX; i++) {
            q_tilde[i] = -c->Q[i] * x_ref[k * NX + i]
                        + c->rho * (v->yx[k * NX + i] - v->zx[k * NX + i]);
        }
        
        /* Compute r_tilde[k] = -R * uref[k] + rho * (yu[k] - zu[k]) */
        for (int i = 0; i < NU; i++) {
            r_tilde[i] = -c->R[i] * u_ref[k * NU + i]
                        + c->rho * (v->yu[k * NU + i] - v->zu[k * NU + i]);
        }
        
        /* p_next_adj = p[k+1] + P_inf @ d */
        vec_add(&v->p[(k + 1) * NX], Pinf_d, p_next_adj, NX);
        
        /* Bt_p = Bd^T @ p_next_adj */
        matvec_transpose(c->Bd, p_next_adj, Bt_p, NX, NU);
        
        /* d_ctrl[k] = C1 @ (Bt_p + r_tilde) */
        vec_add(Bt_p, r_tilde, Bt_p_plus_r, NU);
        matvec(c->C1, Bt_p_plus_r, &v->d_ctrl[k * NU], NU, NU);
        
        /* p[k] = q_tilde + C2 @ p_next_adj - K_inf^T @ r_tilde */
        matvec(c->C2, p_next_adj, C2_p, NX, NX);
        matvec_transpose(c->K_inf, r_tilde, Kt_r, NU, NX);
        
        for (int i = 0; i < NX; i++) {
            v->p[k * NX + i] = q_tilde[i] + C2_p[i] - Kt_r[i];
        }
    }
}

/**
 * Forward pass: roll out trajectory using cached K_inf and computed d_ctrl.
 *
 * x[0] = x0  (given)
 * For k = 0 to N-1:
 *   u[k] = -K_inf @ x[k] - d_ctrl[k]
 *   x[k+1] = Ad @ x[k] + Bd @ u[k] + d
 */
static void forward_pass(const ADMMCache *c, ADMMVars *v,
                         const double *x0)
{
    double Kx[NU];      /* K_inf @ x[k] */
    double Ax[NX];      /* Ad @ x[k] */
    double Bu[NX];      /* Bd @ u[k] */
    
    /* x[0] = x0 */
    vec_copy(x0, &v->x[0], NX);
    
    for (int k = 0; k < NHORIZON; k++) {
        /* u[k] = -K_inf @ x[k] - d_ctrl[k] */
        matvec(c->K_inf, &v->x[k * NX], Kx, NU, NX);
        for (int i = 0; i < NU; i++) {
            v->u[k * NU + i] = -Kx[i] - v->d_ctrl[k * NU + i];
        }
        
        /* x[k+1] = Ad @ x[k] + Bd @ u[k] + d */
        matvec(c->Ad, &v->x[k * NX], Ax, NX, NX);
        matvec(c->Bd, &v->u[k * NU], Bu, NX, NU);
        for (int i = 0; i < NX; i++) {
            v->x[(k + 1) * NX + i] = Ax[i] + Bu[i] + c->d[i];
        }
    }
}

/**
 * Slack update: project onto constraint set.
 *
 * z_x[k] = clamp(x[k] + y_x[k], x_min, x_max)
 * z_u[k] = clamp(u[k] + y_u[k], u_min, u_max)
 *
 * This is the ONLY place constraints are enforced.
 * For box constraints, projection = element-wise clamping.
 * No matrix operations. This is why ADMM is elegant for box constraints.
 */
static void slack_update(const ADMMCache *c, ADMMVars *v)
{
    double temp[NX > NU ? NX : NU];
    
    for (int k = 0; k <= NHORIZON; k++) {
        /* temp = x[k] + yx[k] */
        vec_add(&v->x[k * NX], &v->yx[k * NX], temp, NX);
        /* zx[k] = clamp(temp, x_min, x_max) */
        vec_clamp(temp, c->x_min, c->x_max, &v->zx[k * NX], NX);
    }
    
    for (int k = 0; k < NHORIZON; k++) {
        vec_add(&v->u[k * NU], &v->yu[k * NU], temp, NU);
        vec_clamp(temp, c->u_min, c->u_max, &v->zu[k * NU], NU);
    }
}

/**
 * Dual update: standard ADMM dual ascent.
 *
 * y_x[k] += x[k] - z_x[k]
 * y_u[k] += u[k] - z_u[k]
 *
 * The dual variable is the "price" of constraint violation.
 * If x > x_max, then z_x = x_max, so x - z_x > 0, so y increases.
 * Higher y pushes the primal solution down in the next iteration.
 * This is how ADMM enforces constraints without hard projection in the primal.
 */
static void dual_update(ADMMVars *v)
{
    for (int k = 0; k <= NHORIZON; k++) {
        for (int i = 0; i < NX; i++) {
            int idx = k * NX + i;
            v->yx[idx] += v->x[idx] - v->zx[idx];
        }
    }
    
    for (int k = 0; k < NHORIZON; k++) {
        for (int i = 0; i < NU; i++) {
            int idx = k * NU + i;
            v->yu[idx] += v->u[idx] - v->zu[idx];
        }
    }
}

/**
 * Compute primal and dual residuals for convergence check.
 *
 * Primal residual: ||x - z_x|| + ||u - z_u||
 *   (how much do primal and slack disagree?)
 *
 * Dual residual: rho * (||z_x_new - z_x_old|| + ||z_u_new - z_u_old||)
 *   (how much did the slack change?)
 *
 * Convergence: both below tolerance.
 */
static void compute_residuals(const ADMMCache *c, const ADMMVars *v,
                              const double *zx_old, const double *zu_old,
                              double *pri_res, double *dual_res)
{
    double diff;
    double pri = 0.0, dua = 0.0;
    
    /* Primal: ||x - zx|| */
    for (int k = 0; k <= NHORIZON; k++) {
        for (int i = 0; i < NX; i++) {
            int idx = k * NX + i;
            diff = v->x[idx] - v->zx[idx];
            pri += diff * diff;
        }
    }
    
    /* Primal: ||u - zu|| */
    for (int k = 0; k < NHORIZON; k++) {
        for (int i = 0; i < NU; i++) {
            int idx = k * NU + i;
            diff = v->u[idx] - v->zu[idx];
            pri += diff * diff;
        }
    }
    
    /* Dual: rho * ||z_new - z_old|| */
    for (int k = 0; k <= NHORIZON; k++) {
        for (int i = 0; i < NX; i++) {
            int idx = k * NX + i;
            diff = v->zx[idx] - zx_old[idx];
            dua += diff * diff;
        }
    }
    for (int k = 0; k < NHORIZON; k++) {
        for (int i = 0; i < NU; i++) {
            int idx = k * NU + i;
            diff = v->zu[idx] - zu_old[idx];
            dua += diff * diff;
        }
    }
    
    *pri_res = sqrt(pri);
    *dual_res = c->rho * sqrt(dua);
}

/* ---- Public functions --------- */

void admm_init(ADMMVars *vars)
{
    memset(vars, 0, sizeof(ADMMVars));
}

void admm_solve(const ADMMCache *cache, ADMMVars *vars,
                const double *x0, const double *x_ref, const double *u_ref,
                int max_iter, double eps_abs,
                double *u_out, ADMMStats *stats)
{
    /* Storage for old slack (needed for dual residual) */
    double zx_old[NX * (NHORIZON + 1)];
    double zu_old[NU * NHORIZON];
    
    /* Timer */
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    
    double pri_res = 0.0, dual_res = 0.0;
    int iter;
    
    for (iter = 0; iter < max_iter; iter++) {
        
        /* Step 1: Backward pass (compute affine terms) */
        backward_pass(cache, vars, x_ref, u_ref);
        
        /* Step 2: Forward pass (roll out trajectory) */
        forward_pass(cache, vars, x0);
        
        /* Step 3: Save old slack, then update slack (clamp) */
        vec_copy(vars->zx, zx_old, NX * (NHORIZON + 1));
        vec_copy(vars->zu, zu_old, NU * NHORIZON);
        slack_update(cache, vars);
        
        /* Step 4: Dual update */
        dual_update(vars);
        
        /* Step 5: Convergence check */
        compute_residuals(cache, vars, zx_old, zu_old, &pri_res, &dual_res);
        
        double eps = eps_abs * sqrt((double)(NX * (NHORIZON + 1) + NU * NHORIZON));
        
        if (pri_res < eps && dual_res < eps) {
            iter++;  /* count the final iteration */
            break;
        }
    }
    
    /* Timer end */
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed_us = (t_end.tv_sec - t_start.tv_sec) * 1e6 +
                        (t_end.tv_nsec - t_start.tv_nsec) / 1e3;
    
    /* Extract first control from SLACK (clamped), not primal (unclamped).
     * When converged: u = zu (identical).
     * When max_iter hit: u can be out of bounds, zu is always feasible. */
    vec_copy(&vars->zu[0], u_out, NU);
    
    /* Fill stats */
    stats->iterations = iter;
    stats->primal_residual = pri_res;
    stats->dual_residual = dual_res;
    stats->solve_time_us = elapsed_us;
}

void admm_warm_shift(ADMMVars *vars)
{
    /* Shift states left: x[:,1:] -> x[:,:-1], duplicate last */
    memmove(&vars->x[0], &vars->x[NX], NX * NHORIZON * sizeof(double));
    /* x[:,N] stays (already in place from memmove leaving last element) */

    memmove(&vars->zx[0], &vars->zx[NX], NX * NHORIZON * sizeof(double));
    memmove(&vars->yx[0], &vars->yx[NX], NX * NHORIZON * sizeof(double));

    /* Shift controls left: u[:,1:] -> u[:,:-1], duplicate last */
    if (NHORIZON > 1) {
        memmove(&vars->u[0], &vars->u[NU], NU * (NHORIZON - 1) * sizeof(double));
        memmove(&vars->zu[0], &vars->zu[NU], NU * (NHORIZON - 1) * sizeof(double));
        memmove(&vars->yu[0], &vars->yu[NU], NU * (NHORIZON - 1) * sizeof(double));
    }
}
