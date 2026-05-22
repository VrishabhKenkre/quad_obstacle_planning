"""
ppo_finetune/train_ppo.py -- phase-2 training loop.

Changes from phase 1 (see brief):
  - Reward redesigned: per-step is safety + action-magnitude only;
    goal-reaching is a TERMINAL cost (see ppo_finetune.reward).
  - Relative drift cap (||theta-theta_init|| / ||theta_init||) replaces
    the absolute L2 cap that was too tight for 10.8M params.
  - Learning rate bumped 4x to 1e-6.
  - Value net is WARMSTARTED across iterations (10 epochs/iter vs the
    phase-1 30-epoch from-scratch refit).
  - 30 iterations, 100 rollouts/iter, 500-step horizon, 100 AWR steps,
    230-seed pool.
  - Checkpoint every 5 iterations: iter5, iter10, iter15, iter20,
    iter25, iter30.
  - Abort conditions per the brief:
      relative drift > 0.05 cumulative           -> abort
      mean_return regression > 30% from peak     -> abort
      iter mean random goal err > 150 mm         -> abort
      any random seed max_field > 1.5 this iter  -> abort

Usage:
  python -m ppo_finetune.train_ppo \\
      --init-checkpoint data/diffusion_student_v2_ema.pt \\
      --n-iters 30 --n-rollouts 100 --horizon 500 \\
      --awr-steps 100 --awr-lr 1e-6 --awr-temperature 0.1
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'distillation'))
sys.path.insert(0, str(_ROOT / 'external' / 'diffusion_policy'))

from collect_planner_data import OBS_DIM, ACT_DIM
from diffusion_student import (build_diffusion_policy, make_inference_scheduler,
                                build_normalizer_from_arrays, DEFAULT_HORIZON)

from ppo_finetune.rollout import collect_rollouts, save_rollouts_npz, ROLLOUT_SEED_POOL
from ppo_finetune.advantage import estimate_advantages
from ppo_finetune.awr_update import awr_step, _param_l2, param_drift_relative
from ppo_finetune.reward import compute_episode_undiscounted_return


CHECKPOINT_EVERY = 5     # default: save every 5 iters: iter5, 10, 15, ...


def load_diffusion_policy(checkpoint_path: str, dataset_path: str,
                          device: torch.device):
    data = np.load(_ROOT / dataset_path)
    norm = build_normalizer_from_arrays(data['observations'], data['actions'])
    policy = build_diffusion_policy(
        obs_dim=OBS_DIM, action_dim=ACT_DIM,
        horizon=DEFAULT_HORIZON, n_obs_steps=1,
        n_action_steps=DEFAULT_HORIZON,
        down_dims=(128, 256, 512),
        diffusion_step_embed_dim=128,
        num_inference_steps=8,
    )
    policy.set_normalizer(norm)
    policy.load_state_dict(torch.load(_ROOT / checkpoint_path,
                                      map_location='cpu'))
    policy = policy.to(device).eval()
    policy.noise_scheduler = make_inference_scheduler(num_inference_steps=8)
    policy.num_inference_steps = 8
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-checkpoint', type=str,
                    default='data/diffusion_student_v2_ema.pt')
    ap.add_argument('--snapshot-checkpoint', type=str, default=None,
                    help='if set, drift / KL are measured against THIS '
                         'checkpoint instead of --init-checkpoint. Use '
                         'this when resuming training from a fine-tuned '
                         'policy but you still want drift measured from '
                         'the original BC weights.')
    ap.add_argument('--start-iter', type=int, default=1,
                    help='iteration index to label the first iter with '
                         '(use when resuming from a saved checkpoint).')
    ap.add_argument('--data', type=str,
                    default='data/planner_dataset_v2.npz')

    # Phase-2 scale-up defaults
    ap.add_argument('--n-iters', type=int, default=30)
    ap.add_argument('--n-rollouts', type=int, default=100)
    ap.add_argument('--horizon', type=int, default=500)
    ap.add_argument('--awr-steps', type=int, default=100)
    ap.add_argument('--awr-lr', type=float, default=1e-6,
                    help='Phase-2 bumped 4x from phase-1 (5e-7 -> 1e-6) '
                         'because the relative drift cap is more permissive.')
    ap.add_argument('--awr-batch', type=int, default=64)
    ap.add_argument('--awr-temperature', type=float, default=0.1)
    ap.add_argument('--awr-weight-clip', type=float, default=5.0)
    ap.add_argument('--reward-version', type=str, default='phase3',
                    choices=['phase2', 'phase3'],
                    help='which reward shaping to use during rollouts. '
                         'phase2 = terminal goal cost (legacy). '
                         'phase3 = per-step progress + path penalty + '
                         'safety + small control (default).')
    # Phase-3 reward weight overrides (None = use reward.py defaults).
    ap.add_argument('--reward-lambda', type=float, default=None)
    ap.add_argument('--reward-eta', type=float, default=None)
    ap.add_argument('--reward-beta', type=float, default=None)
    ap.add_argument('--reward-safe-margin', type=float, default=None)
    ap.add_argument('--reward-kappa', type=float, default=None)
    ap.add_argument('--gae-gamma', type=float, default=0.99)
    ap.add_argument('--gae-lambda', type=float, default=0.95)
    ap.add_argument('--value-epochs', type=int, default=10,
                    help='per-iter value refit epochs; warmstarted, so '
                         'fewer epochs per iter than phase 1.')

    # Phase-3 abort: stalled-policy guard (no movement + bad goal)
    ap.add_argument('--stall-step-length-cm', type=float, default=1.0,
                    help='if mean per-step path length < this AND '
                         'random goal err > stall_goal_limit by iter '
                         'stall_after_iter, abort (the policy isn''t '
                         'moving but isn''t converging either).')
    ap.add_argument('--stall-goal-limit-mm', type=float, default=120.0)
    ap.add_argument('--stall-after-iter', type=int, default=5)

    # Phase-2 abort thresholds (relative drift + behavioural regression)
    ap.add_argument('--drift-rel-iter-limit', type=float, default=0.02,
                    help='abort if per-iter relative drift exceeds 2%')
    ap.add_argument('--drift-rel-cum-limit', type=float, default=0.05,
                    help='abort if cumulative relative drift exceeds 5%')
    ap.add_argument('--return-regression-limit', type=float, default=0.30,
                    help='abort if mean_return drops by more than 30% '
                         'from the running peak (after warmup)')
    ap.add_argument('--regression-warmup-iters', type=int, default=5,
                    help='skip the return-regression check for the first '
                         'N iterations (the peak needs to stabilise '
                         'before regression-from-peak is meaningful)')
    ap.add_argument('--goal-err-limit-mm', type=float, default=150.0,
                    help='abort if iter mean random-seed goal err > 150 mm')
    ap.add_argument('--max-field-limit', type=float, default=1.5,
                    help='abort if any random-seed max_field exceeds 1.5')

    ap.add_argument('--save-iters', type=int, nargs='+', default=None,
                    help='explicit list of iterations to checkpoint. If '
                         'omitted, defaults to every CHECKPOINT_EVERY=5 '
                         'iters as in phase 2.')
    ap.add_argument('--checkpoint-dir', type=str, default='data')
    ap.add_argument('--checkpoint-prefix', type=str, default='diffusion_v2_ppo_phase2')
    ap.add_argument('--rollouts-dir', type=str, default='data/ppo_rollouts')
    ap.add_argument('--log-out', type=str,
                    default='results/ppo_phase2_training_log.json')
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[ppo2-train] device={device}, seed pool size={len(ROLLOUT_SEED_POOL)}")

    # ---- Load policy + drift snapshot ----
    policy = load_diffusion_policy(args.init_checkpoint, args.data, device)
    print(f"[ppo2-train] loaded init checkpoint: {args.init_checkpoint}")
    if args.snapshot_checkpoint is not None:
        init_snapshot = load_diffusion_policy(args.snapshot_checkpoint,
                                              args.data, device)
        print(f"[ppo2-train] drift measured vs: {args.snapshot_checkpoint}")
    else:
        init_snapshot = copy.deepcopy(policy).to(device).eval()
    for p in init_snapshot.parameters():
        p.requires_grad_(False)

    # Value net is warmstarted across iterations.
    value_net = None

    log = dict(
        config=vars(args),
        iterations=[],
        aborted=False,
        abort_reason=None,
        peak_return=None,
    )

    t_train_start = time.time()
    peak_return = -float('inf')
    for raw_it in range(1, args.n_iters + 1):
        it = raw_it + args.start_iter - 1   # logical iter index
        print(f"\n[ppo2-train] === iteration {it}/"
              f"{args.start_iter + args.n_iters - 1} ===")
        t_iter = time.time()
        # 1. Collect rollouts under current policy
        print(f"[ppo2-train] iter {it}: collecting {args.n_rollouts} rollouts "
              f"(horizon={args.horizon})")
        t0 = time.time()
        reward_kwargs = dict(version=args.reward_version)
        for k, v in (('lam', args.reward_lambda),
                     ('eta', args.reward_eta),
                     ('beta', args.reward_beta),
                     ('safe_margin', args.reward_safe_margin),
                     ('kappa', args.reward_kappa)):
            if v is not None:
                reward_kwargs[k] = float(v)
        episodes = collect_rollouts(
            policy, n_episodes=args.n_rollouts, horizon=args.horizon,
            device=device, rng_seed=it,
            reward_kwargs=reward_kwargs)
        t_rollouts = time.time() - t0
        rollouts_npz = Path(_ROOT / args.rollouts_dir
                            / f'{args.reward_version}_round_{it}.npz')
        save_rollouts_npz(episodes, rollouts_npz)

        # 2. Estimate advantages (value net warmstarted)
        print(f"[ppo2-train] iter {it}: GAE (value warmstart, "
              f"{args.value_epochs} epochs)")
        adv = estimate_advantages(
            episodes, device, gamma=args.gae_gamma, lam=args.gae_lambda,
            value_epochs=args.value_epochs,
            value_net=value_net)
        value_net = adv['value_net']   # carry forward

        # 3. AWR update
        print(f"[ppo2-train] iter {it}: AWR ({args.awr_steps} grad steps, "
              f"lr={args.awr_lr}, T={args.awr_temperature})")
        actions_all = np.concatenate([e.actions for e in episodes], axis=0)
        upd = awr_step(
            policy, adv['obs'], actions_all, adv['advantages_norm'],
            adv['episode_ends'], device,
            n_grad_steps=args.awr_steps, batch_size=args.awr_batch,
            lr=args.awr_lr, temperature=args.awr_temperature,
            weight_clip=args.awr_weight_clip,
            window=DEFAULT_HORIZON,
            init_policy=init_snapshot)
        cum_drift_abs = _param_l2(policy, init_snapshot)
        cum_drift_rel = param_drift_relative(policy, init_snapshot)

        # 4. Aggregate metrics
        ep_returns = [compute_episode_undiscounted_return(e.rewards)
                      for e in episodes]
        ep_goal_err = [e.goal_err_mm for e in episodes]
        ep_goal_err_random = [e.goal_err_mm for e in episodes
                              if e.layout == 'random']
        ep_max_field = [e.max_field for e in episodes]
        ep_max_field_random = [e.max_field for e in episodes
                                if e.layout == 'random']
        peak_return = max(peak_return, float(np.mean(ep_returns)))

        # Per-iter reward-component means and path-length stats
        comp_keys = ('r_progress', 'r_safety', 'r_path',
                     'r_control', 'r_terminal')
        component_means = {}
        for k in comp_keys:
            vals = [(e.reward_components or {}).get(k, 0.0)
                    for e in episodes]
            component_means[k] = float(np.mean(vals)) if vals else 0.0
        path_lengths_m = [float(getattr(e, 'path_length_m', 0.0))
                          for e in episodes]
        mean_step_length_cm = (100.0 * float(np.mean(path_lengths_m))
                               / max(float(np.mean(
                                   [e.n_steps for e in episodes])), 1.0))
        iter_log = dict(
            iteration=int(it),
            wallclock_seconds=float(time.time() - t_iter),
            rollout_seconds=float(t_rollouts),
            n_episodes=len(episodes),
            n_transitions=int(sum(e.n_steps for e in episodes)),
            mean_return=float(np.mean(ep_returns)),
            mean_goal_err_mm=float(np.mean(ep_goal_err)),
            mean_goal_err_mm_random=float(np.mean(ep_goal_err_random)
                                           if ep_goal_err_random else 0.0),
            mean_max_field=float(np.mean(ep_max_field)),
            median_max_field=float(np.median(ep_max_field)),
            p95_max_field=float(np.percentile(ep_max_field, 95)),
            max_field_max_random=float(np.max(ep_max_field_random)
                                       if ep_max_field_random else 0.0),
            param_drift_l2=float(cum_drift_abs),
            param_drift_l2_iter=float(upd['param_drift_l2']),
            param_drift_relative=float(cum_drift_rel),
            kl_to_init=float(upd['kl_to_init']),
            awr_loss_first=float(upd['loss_curve'][0]),
            awr_loss_last=float(upd['loss_curve'][-1]),
            awr_weighted_loss_first=float(upd['weighted_loss_curve'][0]),
            awr_weighted_loss_last=float(upd['weighted_loss_curve'][-1]),
            adv_mean_raw=float(adv['adv_mean']),
            adv_std_raw=float(adv['adv_std']),
            peak_return_so_far=float(peak_return),
            # Phase-3 additions
            reward_version=args.reward_version,
            reward_components=component_means,
            mean_path_length_m=float(np.mean(path_lengths_m)),
            mean_step_length_cm=float(mean_step_length_cm),
        )
        log['iterations'].append(iter_log)
        log['peak_return'] = float(peak_return)
        print(f"[ppo2-train] iter {it}: "
              f"return={iter_log['mean_return']:.2f} (peak={peak_return:.2f})  "
              f"goal={iter_log['mean_goal_err_mm']:.0f}mm  "
              f"max_field mean/p95={iter_log['mean_max_field']:.3f}/"
              f"{iter_log['p95_max_field']:.3f}  "
              f"drift_rel={iter_log['param_drift_relative']:.4f}  "
              f"step={iter_log['mean_step_length_cm']:.2f}cm")
        cm = iter_log['reward_components']
        print(f"           [reward] r_progress={cm['r_progress']:.2f}  "
              f"r_safety={cm['r_safety']:.2f}  "
              f"r_path={cm['r_path']:.2f}  "
              f"r_control={cm['r_control']:.4f}  "
              f"r_terminal={cm['r_terminal']:.2f}")

        # 5. Periodic checkpoint
        should_save = ((args.save_iters is None and it % CHECKPOINT_EVERY == 0)
                       or (args.save_iters is not None and it in args.save_iters))
        if should_save:
            ckpt_path = Path(_ROOT / args.checkpoint_dir
                             / f'{args.checkpoint_prefix}_iter{it}.pt')
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[ppo2-train] iter {it}: saved checkpoint {ckpt_path}")

        # 6. Persist log after every iter (so partial logs survive aborts)
        log_path = Path(_ROOT / args.log_out)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)

        # 7. Abort guards (phase 2)
        abort_reason = None
        if iter_log['param_drift_relative'] > args.drift_rel_cum_limit:
            abort_reason = (
                f"cumulative relative drift {iter_log['param_drift_relative']:.4f}"
                f" exceeded the {args.drift_rel_cum_limit:.2%} cap")
        # Use a 3-iter rolling average for both the peak and the current
        # to filter out single-iter noise from the random/dp seed-mix.
        elif it > args.regression_warmup_iters:
            recent3 = [r['mean_return'] for r in log['iterations'][-3:]]
            if len(recent3) >= 3 and len(log['iterations']) >= 6:
                cur_avg = float(np.mean(recent3))
                past_avgs = [
                    float(np.mean([r['mean_return']
                                   for r in log['iterations'][i:i+3]]))
                    for i in range(len(log['iterations']) - 3)
                ]
                smooth_peak = max(past_avgs) if past_avgs else cur_avg
                if (smooth_peak - cur_avg
                        > args.return_regression_limit * abs(smooth_peak)):
                    abort_reason = (
                        f"3-iter avg return {cur_avg:.1f} regressed by more "
                        f"than {args.return_regression_limit:.0%} from "
                        f"smoothed peak {smooth_peak:.1f}")
        elif iter_log['mean_goal_err_mm_random'] > args.goal_err_limit_mm:
            abort_reason = (
                f"random-seed mean goal_err "
                f"{iter_log['mean_goal_err_mm_random']:.0f}mm exceeded "
                f"the {args.goal_err_limit_mm:.0f}mm limit")
        elif iter_log['max_field_max_random'] > args.max_field_limit:
            abort_reason = (
                f"random-seed max_field {iter_log['max_field_max_random']:.3f}"
                f" exceeded the {args.max_field_limit:.2f} safety cliff")
        elif (it >= args.stall_after_iter
              and iter_log['mean_step_length_cm'] < args.stall_step_length_cm
              and iter_log['mean_goal_err_mm_random'] > args.stall_goal_limit_mm):
            abort_reason = (
                f"policy stalled by iter {it}: mean step length "
                f"{iter_log['mean_step_length_cm']:.2f} cm < "
                f"{args.stall_step_length_cm:.2f} cm AND random goal err "
                f"{iter_log['mean_goal_err_mm_random']:.0f} mm > "
                f"{args.stall_goal_limit_mm:.0f} mm")

        if abort_reason is not None:
            log['aborted'] = True
            log['abort_reason'] = abort_reason
            # Save the current policy as a "_aborted" checkpoint
            abort_ckpt = Path(_ROOT / args.checkpoint_dir
                              / f'{args.checkpoint_prefix}_aborted_iter{it}.pt')
            torch.save(policy.state_dict(), abort_ckpt)
            with open(log_path, 'w') as f:
                json.dump(log, f, indent=2)
            print(f"\n[ppo2-train] ABORT iter {it}: {abort_reason}")
            print(f"[ppo2-train] saved aborted checkpoint -> {abort_ckpt}")
            break

    total_wall = time.time() - t_train_start
    log['total_wallclock_seconds'] = float(total_wall)
    with open(_ROOT / args.log_out, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\n[ppo2-train] DONE in {total_wall:.1f}s "
          f"({total_wall/3600:.2f} hr)")
    print(f"[ppo2-train] log -> {args.log_out}")


if __name__ == '__main__':
    main()
