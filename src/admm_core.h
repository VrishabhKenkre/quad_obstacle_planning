/*
 * admm_core.h -- hand-rolled ADMM solver for quadrotor MPC.
 *
 * Plain C, no dependencies. Statically sized (no malloc). Callable from
 * Python via ctypes, or droppable onto a bare-metal microcontroller.
 *
 * State NX=12, control NU=4, horizon NHORIZON=20.
 */

#ifndef ADMM_CORE_H
#define ADMM_CORE_H

/* ---- Problem dimensions (compile-time constants) --------- */
#define NX 12       /* state dimension */
#define NU 4        /* control dimension */
#define NHORIZON 20 /* prediction horizon */

/* ---- Cached Riccati matrices (precomputed offline in Python) ---------
 *
 * From the DARE with augmented costs Q_tilde = Q + rho*I, R_tilde = R + rho*I.
 *
 *   K_inf  [NU x NX]:  optimal feedback gain
 *   P_inf  [NX x NX]:  infinite-horizon cost-to-go
 *   C1     [NU x NU]:  (R_tilde + Bd'P_inf Bd)^{-1}, used in backward pass
 *   C2     [NX x NX]:  (Ad - Bd K_inf)', closed-loop A transposed
 */
typedef struct {
    double K_inf[NU * NX];       /* row-major: K_inf[i*NX + j] */
    double P_inf[NX * NX];
    double C1[NU * NU];
    double C2[NX * NX];
    double Ad[NX * NX];          /* discrete dynamics A */
    double Bd[NX * NU];          /* discrete dynamics B */
    double Q[NX];                /* diagonal of state cost (stored as vector) */
    double R[NU];                /* diagonal of control cost */
    double P_terminal[NX];       /* diagonal of terminal cost (simplified) */
    double d[NX];                /* gravity offset */
    double u_hover[NU];          /* hover control */
    double x_min[NX];            /* state lower bounds */
    double x_max[NX];            /* state upper bounds */
    double u_min[NU];            /* control lower bounds */
    double u_max[NU];            /* control upper bounds */
    double rho;                  /* ADMM penalty parameter */
} ADMMCache;

/* ---- ADMM variables (persist between solves for warm starting) --------- */
typedef struct {
    /* Primal variables */
    double x[NX * (NHORIZON + 1)];   /* states:   x[:, k] = x[k*NX .. (k+1)*NX-1] */
    double u[NU * NHORIZON];          /* controls: u[:, k] = u[k*NU .. (k+1)*NU-1] */
    
    /* Slack variables (constraint-satisfying copies) */
    double zx[NX * (NHORIZON + 1)];
    double zu[NU * NHORIZON];
    
    /* Dual variables (Lagrange multipliers) */
    double yx[NX * (NHORIZON + 1)];
    double yu[NU * NHORIZON];
    
    /* Temporaries for backward pass */
    double p[NX * (NHORIZON + 1)];    /* affine cost-to-go */
    double d_ctrl[NU * NHORIZON];     /* affine control corrections */
} ADMMVars;

/* ---- Solve statistics --------- */
typedef struct {
    int iterations;
    double primal_residual;
    double dual_residual;
    double solve_time_us;  /* microseconds */
} ADMMStats;

/* ---- Function declarations --------- */

/**
 * Initialize all ADMM variables to zero.
 * Call once at startup.
 */
void admm_init(ADMMVars *vars);

/**
 * Run the full ADMM solve.
 *
 * @param cache     Precomputed Riccati matrices + problem data
 * @param vars      ADMM variables (warm-started from previous solve)
 * @param x0        Current measured state [NX]
 * @param x_ref     Reference trajectory [NX x (NHORIZON+1)], column-major
 * @param u_ref     Reference controls [NU x NHORIZON], column-major
 * @param max_iter  Maximum ADMM iterations
 * @param eps_abs   Absolute convergence tolerance
 * @param u_out     Output: optimal first control [NU]
 * @param stats     Output: solve statistics
 */
void admm_solve(const ADMMCache *cache, ADMMVars *vars,
                const double *x0, const double *x_ref, const double *u_ref,
                int max_iter, double eps_abs,
                double *u_out, ADMMStats *stats);

/**
 * Shift ADMM variables by one step for warm starting.
 * Call after applying u[0] and before the next solve.
 */
void admm_warm_shift(ADMMVars *vars);

#endif /* ADMM_CORE_H */
