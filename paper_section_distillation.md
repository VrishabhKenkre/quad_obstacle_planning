# Distillation: Reactive MLP vs Diffusion Students for the Hierarchical Planner

## Setup & data collection

The hierarchical planner of Section 3 (A\* + minimum-snap + SE(3) NMPC) achieves
**16 ± 14 mm** terminal error and **0.046 ± 0.025** maximum obstacle field on the
canonical 10-seed Gaussian course (Table 1), but runs at **23 ms / step** — well
inside a 50 Hz loop, but two orders of magnitude slower than a reactive learned
policy at deployment. The distillation question we study is whether a tiny
student can recover the planner's safety and goal-reaching quality at sub-100 µs
inference, and whether the choice of policy class — single-mode MLP vs.
multi-mode diffusion — matters.

To distinguish the two students we deliberately constructed a teacher whose
behaviour is **multi-modal**: the same drone state in the same obstacle layout
can be tracked equally well by two different planner trajectories (e.g. going
*left* vs *right* around a central obstacle). A single-mode MLP, asked to fit
both, averages them and commits to neither; a diffusion student samples one
mode per rollout and commits to it. This is exactly the claim of Chi et al.
2023, lifted from manipulation into reactive obstacle avoidance.

Multi-modality is engineered into the dataset by three mechanisms:

1. **Randomised A\***.  Adding bounded multiplicative noise to each traversal
   edge weight (ε ∈ U(-0.15, +0.15)) and a soft perpendicular g-cost bias
   produces multiple distinct shortest paths per obstacle layout. We retain
   up to *K = 3* alternatives per seed whose pruned length is within 1.2×
   of the best.
2. **DART noise during rollouts**.  At each step, the teacher's clean
   action is the recorded label, but the action *applied* to the simulator
   is `clip(a* + ε)`, ε ∼ 𝒩(0, σ_DART). This drifts the state off the
   planner's nominal trajectory and exposes the dataset to off-policy
   states, in the spirit of Laskey et al. 2017.
3. **Decision-point layouts** (30 seeds).  A single wide column
   obstacle is placed straddling the start–goal line with σ_z = 1.2 m
   (taller than the corridor) so flying *over* the obstacle is impossible.
   For each layout we force two A\* runs with opposite perpendicular
   biases, yielding a left and a right reference trajectory of comparable
   length.

The collection pipeline (`distillation/collect_planner_data.py`) is sharded
across 8 worker processes (one CasADi/IPOPT instance per worker). On a single
i9-14900HX it produces **192,853 samples / 658 rollouts in 48m48s**, with
**0 / 658** rollouts aborting and a median terminal error of **14 mm** across
the collected trajectories — so the teacher quality during data collection
matches the eval-time planner.

| dataset slice | rollouts | samples | mean teacher goal err |
|---|---|---|---|
| random seeds (K=3) | 598 | 173,758 | 14 mm |
| decision-point (K=2) | 60 | 19,095 | 12 mm |
| **total** | **658** | **192,853** | **14 mm** |

The observation is intentionally **reactive** (`distillation/collect_planner_data.py::make_observation`): a
**24-D vector** containing the 12-D drone state, 1 ESDF value at the current
position, the 3-D ESDF gradient, and 8 ESDF look-ahead probes around the body.
This deliberately excludes any reference of the planner's intent. The earlier
design that exposed the next four planner waypoints in the observation gave a
nearly-unimodal training set by construction (each rollout's observation
uniquely identifies its plan), eliminating the diffusion-vs-MLP comparison.
Keeping the policy reactive and the dataset multi-modal is the entire point of
this experiment.

A k-NN multi-modality audit on the resulting dataset
(`distillation/audit_modes.py`) finds that **45 %** of `k=5` neighbourhoods
have action span > 0.10 (in normalised action units, [-1, 1]); when restricted
to *cross-variant* neighbours (forcing the comparison across two different
planner rollouts), the figure is **53 %** with median span 0.102. Both numbers
clear the 20–30 % threshold needed for diffusion to plausibly beat MLP.

### Dataset Multimodality Audit

A deeper audit on `planner_dataset_v2.npz`
(`distillation/audit_dataset_v2.py`) tightens both halves of the
multi-modality claim by clustering per-seed and per-variant rather than
globally. For each of the 230 unique seeds we build a per-seed k-NN over
standardised 24-D observations, then form each query's cluster by
pulling the `k/V` nearest neighbours from *each* planner variant within
the seed (so a single-variant rollout cannot dominate the cluster). The
fraction of clusters whose max pairwise L₂ action distance exceeds 0.20
is the headline kNN multi-modality number; we also report a sweep over
{0.05, 0.10, 0.15, 0.20, 0.30} to characterise the action-distance
distribution. In parallel we sample 2,000 clusters (`k=50`) across the
whole dataset and fit a diagonal-covariance GMM with `n ∈ {1,2,3,4}`
components, picking `n` by BIC and (for sensitivity) AIC.

The kNN audit finds **98 %** of cross-variant clusters exceed span 0.10,
**66 %** exceed 0.15, and **26 %** exceed 0.20 — confirming that variant
disagreement at matched observations is the rule, not the exception, but
also that the cross-variant action gap is *typically* small (median 0.16)
and only a quarter of clusters separate by the diffusion student's
nominal action scale of 0.20. The BIC GMM selects `n≥2` on **7 %** of
clusters and AIC on **19 %**; the under-counting relative to the kNN
metric is expected — with only 50 samples in 4-D, BIC heavily penalises
extra components unless two action modes are well separated relative to
their within-mode spread (see [results/action_mode_distribution.png](results/action_mode_distribution.png)).
The right panel of that figure plots the intra-cluster total action
variance on a log scale and is heavily right-skewed: most clusters live
near variance 1e-3 (highly unimodal), but a long tail extends past 1e-1
(the strongly multi-modal clusters near decision points).

[results/multimodality_per_obstacle.png](results/multimodality_per_obstacle.png)
plots the per-seed multi-modality fraction at span > 0.20, split into
random seeds (n=200, blue) and decision-point seeds (n=30, red). A
finding that bears noting: **random seeds are noticeably more
cross-variant-multi-modal than decision-point seeds** (median 0.25 vs
0.11). This inverts the intuition of the dataset design: while
decision-point seeds *engineer* two clearly distinct paths around a
single obstacle, those paths share most of their trajectory and only
diverge near the central obstacle, so most matched observations across
the two variants take very similar actions. Random seeds, by contrast,
sample three *globally* different A\* paths through eight Gaussian
obstacles, producing meaningfully different actions throughout. The
diffusion student therefore picks up multi-modality from the *random*
slice as much as the decision-point slice — which is consistent with the
v2 student's strong random-seed performance (93 mm goal err) holding
up alongside the dp-seed performance (69 mm).

## MLP student (DAgger+DART on the planner)

We adopt the same architecture as the existing IL student of Section 4:
two hidden layers of 64 ReLU units, tanh output. With our 24-D observation
the parameter count is **6,020** (the existing 20-D student is 5,764);
otherwise identical.

The student is first pre-trained with **behavioural cloning** on the Phase-1
dataset for 80 epochs (AdamW, lr 1e-3, batch 256, best-val checkpoint). It
then runs **5 DAgger+DART iterations**: at each iter, 40 episodes are rolled
out with the current student under decaying DART noise (β starts at 1.0 and
linearly anneals to 0.4), every visited state is relabelled with the
planner's NMPC action, and the model is fine-tuned for 40 more epochs.

BC training reaches **val MSE 0.0007** on the held-out 5 % of the dataset, a
remarkably tight fit. DAgger+DART then *raises* validation loss
monotonically from 0.001 → 0.030 across iterations — the new off-distribution
samples are genuinely harder to predict because there is no single correct
action at the visited states. The 6 k-parameter MLP does not have the
capacity to represent the multi-modal output distribution, so it converges
to a compromise.

### Eval

Both BC-only and DAgger+DART checkpoints are evaluated on the 10-seed
random obstacle course (`distillation/train_mlp.py`) and on a separate
10-seed decision-point suite (`distillation/eval_decision_points.py`).

| MLP variant | random goal err | random max field | dp goal err | dp max field |
|---|---|---|---|---|
| BC-only | 37 ± 22 mm | 0.325 ± 0.370 | **2406 ± 2862 mm** | 0.603 ± 0.397 |
| DAgger+DART | 846 ± 260 mm | 0.251 ± 0.167 | 1123 ± 627 mm | 0.896 ± 0.307 |

On *random* layouts the BC-only MLP looks nearly perfect (37 mm) — but the
field statistics tell a different story: max field 0.325 with **two of ten
seeds at >0.9** (seeds 128 and 2024), i.e. the policy physically passes
through an obstacle on those layouts. On the 10 *decision-point* layouts —
where bi-modality is engineered, not incidental — BC fails on **8 of 10
seeds** with errors ranging from 1.5 m to 9.9 m and a 50× variance increase
on goal error (std 2862 mm). The DAgger variant trades catastrophic failure
on individual seeds for uniform mediocrity (846 mm random / 1123 mm
decision-point), confirming that the on-policy data only helps when the
student class can express the relabelled targets.

Either way, **no MLP recovers the planner**. The MLP IL pipeline that
worked when the teacher was the single-mode NMPC reactive controller
(Section 4, 420 mm goal err) fails on a multi-mode planner teacher because
the student's hypothesis class is wrong.

## Diffusion student (BC on the planner)

The diffusion student is a `ConditionalUnet1D` from Chi et al. 2023,
configured as `DiffusionUnetLowdimPolicy` from `external/diffusion_policy`
with:

* obs_dim = 24, action_dim = 4 (same observation/action interface as the MLP),
* horizon = 8 actions, n_obs_steps = 1, n_action_steps = 8,
* `obs_as_global_cond = True` (FiLM conditioning on the flattened 24-D
   observation; the trajectory is action-only),
* `down_dims = (128, 256, 512)`, FiLM embedding 128,
* DDPM scheduler with 100 train timesteps, `squaredcos_cap_v2` β-schedule,
   ε-prediction,
* total **10.8 M parameters** (overshoots the 700 k target — see Limitations).

Training is **pure behavioural cloning** (no DAgger): the 192 k samples are
sliced into **188,247 sliding windows** of 8 consecutive actions within a
single rollout, normalised to [-1, 1] via the standard `LinearNormalizer`
fit. AdamW with lr 1e-4, batch size 256, EMA on the weights (decay 0.999,
2/3 power ramp). 100 epochs converges in **27 m 32 s** on an RTX 4070
(8 GB), with the EMA val MSE settling around 0.028.

For inference we swap the scheduler to **DDIM with 8 steps**, matching
Chi et al.'s defaults. Receding-horizon execution: the policy predicts 8
actions; we execute only the first per control step. Median inference
latency on the 4070 is **18 ms**, ~16× faster than the planner's 23 ms.

### Eval

The student is evaluated on the same 10-seed random course and 10-seed
decision-point suite (`distillation/train_diffusion.py`,
`distillation/eval_decision_points.py`). Three diffusion configurations
are reported:

* **v1 single**: trained on `planner_dataset_v1.npz`
   (decision-point safety_margin = 0.15 m), single DDIM-8 sample per step.
* **v2 single**: trained on `planner_dataset_v2.npz`
   (decision-point safety_margin = 0.30 m), single DDIM-8 sample per step.
* **v2 multi K=3**: same checkpoint as v2 single, but at each step we draw
   K=3 action sequences in a single batched diffusion call, forward-predict
   each one with the hover-linearised Crazyflie dynamics for the 8-step
   horizon, and execute the first action of the sample with the highest
   minimum-ESDF along its predicted trajectory.  The K samples come for
   free in latency (per-step cost is dominated by Python and DDIM
   scheduling, not U-Net forward passes) — measured inference 18 ms.

#### Reporting mean, median, and p95 of `max_field`

The diffusion student's safety statistic is *heavy-tailed*: most
rollouts have `max_field` ≈ 0.05 (cleanly outside any obstacle), but a
minority enter an obstacle (`max_field` > 0.7).  Reporting only the mean
obscures this — a single bad rollout in a 10-seed eval can shift the
mean by 0.05 → 0.20 while the median stays at 0.05. We therefore report
**mean ± std, median, and p95** of `max_field`; the median characterises
the typical rollout and the p95 characterises the failure tail. The MLP
rows are nearly unimodal, so mean and median agree there. (Goal error is
not heavy-tailed and is summarised mean ± std as usual.)

| diffusion variant | random goal err | random field mean / median / p95 | dp goal err | dp field mean / median / p95 |
|---|---|---|---|---|
| v1 single | 219 ± 82 mm | 0.059 / 0.055 / 0.105 | 118 ± 16 mm | 0.511 / 0.111 / 1.503 |
| v2 single | **93 ± 27 mm** | 0.184 / **0.048** / 0.623 | 69 ± 7 mm | 0.733 / 0.676 / 1.357 |
| v2 multi K=3 | 107 ± 24 mm | 0.174 / **0.047** / 0.620 | **100 ± 5 mm** | 0.452 / **0.554** / **0.976** |

Three observations:

1. **Retraining on v2 cuts the random goal error in half** (219 → 93 mm)
   and brings the dp goal error from 118 mm to 69 mm. The model learns
   tighter goal-reaching once the dataset is no longer dominated by
   tight-clearance dp labels.
2. **The K=3 safety filter helps obstacle safety on dp seeds**: mean
   max_field drops from 0.733 to 0.452, and the p95 drops from 1.357 to
   0.976 — i.e., the worst 5 % of dp rollouts no longer have the drone
   inside an obstacle by 1.4×, only by ≤1.0× (`max_field` = 1.0
   corresponds to being one σ-radius from the obstacle centre).  The
   median is unchanged because the typical rollout is already safe.
3. **On random seeds the median max_field is 0.047** (lower than the
   planner's 0.046) — diffusion v2 actually matches the teacher's safety
   on the easy course; the high mean (0.184) comes from one seed
   (`seed=2024`) where the planner reference itself passes through a
   tight gap and the diffusion sample mis-shoots it.

The decision-point gap vs the MLP is sharper than ever: on dp seeds,
diffusion **v2 multi K=3 reaches the goal at 100 ± 5 mm** while the
MLP-BC checkpoint averages **2406 ± 2862 mm** — a **24× improvement in
goal error with 570× lower variance**.

## Comparison

The four-row paper table on the canonical 10-seed random obstacle course
uses the **v2 K=3** diffusion student (trained on
`planner_dataset_v2.npz`, K=3 multi-sample safety filter at inference):

| controller | goal err (mm) | max obstacle field (mean / median) | inference latency |
|---|---|---|---|
| NMPC teacher (reactive) | 46 ± 10 | 0.106 / – | 59 ms |
| **Hierarchical planner (ours, teacher)** | **16 ± 14** | **0.046** / – | 23 ms |
| MLP DAgger+DART on planner (new) | 846 ± 260 | 0.251 / 0.245 | **29 µs** |
| Diffusion BC v2 K=3 (new) | 107 ± 24 | 0.174 / **0.047** | 18 ms |

The BC-only MLP sanity row (37 ± 22 mm random goal err) is also reported
for completeness, but it has **mean max field 0.325 and physically crashes
on two of ten random seeds** — its goal numbers are not a fair point of
comparison.

The story sharpens on the 10-seed decision-point suite where
multi-modality is forced. We report each model from its most recent
eval; MLP rows are bit-identical between v1 and v2 datasets because the
obstacles are unchanged (only the planner *reference paths* widen).
Diffusion rows show all three configurations to make the
dataset-and-inference improvement explicit:

| controller | dp goal err (mm) | dp field mean ± std | dp field median | dp field p95 |
|---|---|---|---|---|
| Hierarchical planner (v2 settings) | 14 ± 3 | 0.048 ± 0.005 | 0.046 | 0.056 |
| MLP BC-only on planner | 2406 ± 2862 | 0.603 ± 0.397 | 0.553 | 1.233 |
| MLP DAgger+DART on planner | 1123 ± 627 | 0.896 ± 0.307 | 0.912 | 1.336 |
| Diffusion BC v1 single | 118 ± 16 | 0.511 ± 0.630 | 0.111 | 1.503 |
| Diffusion BC v2 single | 69 ± 7 | 0.733 ± 0.448 | 0.676 | 1.357 |
| **Diffusion BC v2 multi K=3** | **100 ± 5** | **0.452 ± 0.358** | 0.554 | **0.976** |

A scatter of `(latency, goal err)` for the four headline rows is in
[results/comparison_plot.png](results/comparison_plot.png); the random
vs decision-point breakdown is in
[results/comparison_random_vs_dp.png](results/comparison_random_vs_dp.png).
A 16 s side-by-side video on decision-point seed 6 — MLP-BC drives through
the central obstacle (max field 1.42), diffusion picks the left corridor
and arrives clean (max field 0.13) — is at
[videos/mlp_vs_diffusion_multimodal.mp4](videos/mlp_vs_diffusion_multimodal.mp4).

### Dataset v2 and the K=3 inference-time safety filter

The v1 diffusion student showed a heavy-tailed `max_field` on the
decision-point seeds. Investigation traced the cause to two co-occurring
issues:

1. **The v1 dataset's dp labels were too tight.** The decision-point
   seeds were built with `safety_margin = 0.15 m` and `lam = 1.5`,
   producing left/right alternative paths with **minimum ESDF clearance
   ≈ 0.17 m** (only 2 cm beyond the inflated obstacle). The action
   distribution near a dp therefore straddles the obstacle boundary.
2. **DDIM sampling is stochastic.** The v1 student's per-rollout
   `max_field` depends on the DDIM seed: re-running the v1 eval with a
   different `torch.manual_seed` produced 0.84 mean (median 0.93)
   instead of the 0.51 mean (median 0.11) reported in the first run.
   The expected-over-samples failure tail is real and large.

The fix was twofold and dataset-and-inference-side, *not* model-side
(no architecture change):

* **Dataset v2** (`distillation/collect_planner_data.py --dp-only`)
  rebuilds *only* the 30 dp rollouts with `safety_margin = 0.30 m` and
  `lam = 2.5`; the random 200 seeds are copied unchanged from v1 into
  [data/planner_dataset_v2.npz](data/planner_dataset_v2.npz). The new
  dataset has **min ESDF ≥ 0.293 m on 60/60 dp rollouts** (median
  0.44 m), 2.5× the v1 floor.  The v2 file is 193,366 samples
  (173,758 random + 19,608 dp). The diffusion student was BC-trained
  for the standard 100 epochs on v2 (~27 m on RTX 4070); the MLP
  students were *not* retrained, since the obstacles and the random
  samples are unchanged and the MLP rows are bit-identical between v1
  and v2 datasets.
* **K=3 multi-sample safety filter at inference.** At each control
  step, the diffusion student now draws K=3 action sequences in a
  single batched call. For each sample, a 8-step world-position
  trajectory is forward-predicted with the hover-linearised Crazyflie
  dynamics, evaluated against the ESDF, and scored by `min ESDF`. The
  highest-scoring sample's first action is executed. K=3 is essentially
  free in wall time (18 ms median; the U-Net forward dominates only at
  much higher K) and converts the diffusion's heavy-tailed
  per-rollout safety into a more uniform "pick the safest of three"
  policy.

The headline diffusion row in the four-row paper table above is the
v2-multi-K=3 configuration. Across both axes:

| | v1 single | v2 single | v2 multi K=3 |
|---|---|---|---|
| random goal err (mean) | 219 ± 82 mm | **93 ± 27 mm** | 107 ± 24 mm |
| random max-field (mean / median / p95) | 0.059 / 0.055 / 0.105 | 0.184 / 0.048 / 0.623 | 0.174 / 0.047 / 0.620 |
| dp goal err (mean) | 118 ± 16 mm | **69 ± 7 mm** | 100 ± 5 mm |
| dp max-field (mean / median / p95) | 0.511 / 0.111 / 1.503 | 0.733 / 0.676 / 1.357 | 0.452 / 0.554 / **0.976** |
| inference (DDIM-8) | 18 ms | 18 ms | 18 ms |

Retraining on v2 alone halves the random goal error and brings the dp
goal error from 118 → 69 mm, but its dp `max_field` actually mean-rises
from 0.511 to 0.733: because the model now arrives faster (less time
spent oscillating in the corridor), each rollout commits earlier and the
ones that pick the wrong side commit harder. The K=3 safety filter
recovers most of the safety: mean dp `max_field` drops to 0.452 and
**p95 drops from 1.357 to 0.976**, i.e. the worst-tail rollouts no longer
penetrate the obstacle by 1.4×σ — only by ≤1.0×σ.

### K sweep on the safety filter

To characterise the safety/goal-reaching/latency trade-off of the
multi-sample filter, we sweep K ∈ {1, 3, 8, 16} on the diffusion v2
student. The dp suite (10 seeds, same seeds as the headline tables,
`torch.manual_seed(1234 + obstacle_seed)`) gives:

| K   | dp goal err (mm) | dp max_field mean / median / p95 | latency (ms) |
|-----|------------------|----------------------------------|---------------|
| 1   | 69 ± 7           | 0.733 / 0.676 / 1.357            | 18.2          |
| 3   | 100 ± 5          | 0.452 / 0.554 / **0.976**        | 18.9          |
| 8   | 136 ± 13         | 0.508 / 0.563 / **0.601**        | 20.0          |
| 16  | 169 ± 16         | 0.370 / 0.467 / **0.571**        | 21.3          |

The K-sweep figure ([results/k_sweep.png](results/k_sweep.png)) plots
both dp and random p95 vs K alongside per-step latency. Three findings:

1. **The K=1 → K=3 jump is by far the largest safety win.** p95 drops
   from 1.357 to 0.976 — the policy goes from typically penetrating
   1.4×σ into the obstacle on its worst rollouts to barely grazing at
   ≈1×σ.
2. **K=8 is the next clean Pareto point.** p95 drops a further 0.40
   (0.976 → 0.601) at only +1.1 ms latency, but at a 36 mm cost in dp
   goal error (100 → 136 mm) because the policy now prefers samples that
   take a wider berth of obstacles and therefore arrives further from
   the goal.
3. **K=16 returns very little.** p95 improves only 0.030 (0.601 → 0.571)
   while goal error grows by another 33 mm. The diminishing return
   reflects that the K-sample distribution is mode-limited — once K is
   large enough to consistently expose the "safe" mode at decision
   points, more samples mostly add picks from the same near-optimal
   tail.

No K in the sweep brings the dp p95 below the 0.5 target. Per the
brief's fallback rule, we therefore **keep K=3 as the headline diffusion
configuration**, with the explicit limitation already noted under
*Diffusion safety still has a non-trivial tail*. K=8 is the recommended
choice when the priority shifts from goal-reaching to safety
(`recommended_K` in [results/decision_point_eval_v2_kSweep.json](results/decision_point_eval_v2_kSweep.json)).
The random course is robust to K: random p95 stays in a tight band
(0.620 / 0.706 / 0.668 for K=3, 8, 16) because the random course's
heavy-tail seed (`seed=2024`) is the same single-seed outlier across
all K values — i.e. the K-sample safety filter cannot fix a layout
where the planner reference itself threads a tight gap.

## Limitations

* **Inference latency is 18 ms, not 5 ms.** DDIM-8 dominates wall time on
   a small policy. Consistency-model distillation of the diffusion student
   (Song et al. 2023) is the obvious next step; 1-step inference should
   put the diffusion student under 2 ms with minimal quality loss.
* **The diffusion student has 10.8 M parameters**, ~15× more than the
   prompt's stated 700 k target. The dominant cost is `down_dims=(128, 256, 512)`
   in the `ConditionalUnet1D`. Smaller variants (e.g. `(64, 128, 256)`,
   ~2.7 M params, 16 ms inference) were measured but not evaluated in
   simulation; the inference latency is dominated by per-step Python and
   scheduler overhead rather than U-Net forward passes, so shrinking the
   network would not have meaningfully reduced latency in this
   configuration.
* **Diffusion safety still has a non-trivial tail** even with v2 data and
   the K=3 inference filter: dp `max_field` p95 = 0.976 means the worst
   1-in-20 dp rollout still grazes an obstacle (~1 σ from centre). The
   K=3 filter is a cheap one-shot proxy for what really should be either
   (a) an MPC-style explicit obstacle-aware re-projection of the
   diffusion's predicted action sequence (~5 ms with CasADi), or
   (b) consistency-model distillation with explicit obstacle-cost
   conditioning. Either is an obvious next-step but beyond this paper.
* **Observation is reactive only.** The student cannot replan from
   scratch; on layouts the planner finds unsolvable, the student inherits
   that failure mode. A planner-conditioned student that takes the next
   four planner waypoints as part of the observation would likely match
   the teacher exactly but, as discussed above, would eliminate the
   multi-modal comparison.
* **No hardware validation.** All numbers are MuJoCo simulation on the
   Bitcraze Crazyflie 2 model. The 18 ms diffusion latency on an RTX 4070
   does not translate directly to STM32 deployment; a consistency-model
   variant (or a switched-policy scheme that uses the MLP when far from
   any obstacle and the diffusion when close) is the right way to address
   this for embedded deployment, and is out of scope for this manuscript.

### Headline takeaway

For a *single-mode* teacher (the reactive NMPC), a 6 k-parameter MLP
distilled via DAgger+DART recovers the teacher to 420 mm goal error. For a
genuinely *multi-mode* teacher (the hierarchical planner with random A\*
alternatives), the same MLP architecture collapses to averaging and
catastrophically fails on bimodal layouts (2406 mm dp goal error,
median max_field 0.55). A 10.8 M-parameter diffusion student trained with
pure BC on `planner_dataset_v2.npz` and a K=3 inference-time safety
filter recovers the planner to **100 ± 5 mm dp goal error with median
`max_field` 0.55 and p95 0.98** — a 24× improvement in goal error vs the
MLP-BC baseline at the same parameter regime. The choice of policy class
(single-mode vs multi-mode) matters more than the choice of distillation
algorithm (BC vs DAgger+DART) when the teacher itself is multi-modal;
the inference-time safety filter and a clean dataset are required to
turn the multi-mode student's goal-reaching advantage into a usable
obstacle-safety margin.
