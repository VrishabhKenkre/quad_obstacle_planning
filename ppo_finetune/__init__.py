"""ppo_finetune -- phase 1 PPO/AWR fine-tuning of the v2 diffusion student.

This package builds the scaffolding for advantage-weighted regression on
top of the BC diffusion student trained in `distillation/`. Phase 1
validates the infrastructure end-to-end with a 3-iteration training run
on the existing v2 EMA checkpoint; phase 2 (next sprint) tunes reward
weights and may swap AWR for DPPO if AWR plateaus.

Public API:
  rollout.collect_rollouts
  reward.compute_step_reward / compute_episode_return
  advantage.ValueNet / estimate_advantages
  awr_update.awr_step
  train_ppo.main
  eval_finetuned.main
"""
__version__ = "0.1.0"
