"""hybrid -- planner-online + learned-tracker controller.

The hierarchical A*+min-snap planner runs once per episode to produce a
reference trajectory; a small learned MLP tracker (27-D observation,
~6.3k params, ~30 us inference) then follows that reference at a high
control rate. This decouples slow global planning from fast local
tracking and is the standard aerial-robotics deployment pattern.

Public API:
  planner_runner.PlannerRunner  -- plan once, expose the reference
  mlp_tracker.MLPTracker        -- the learned reference-tracking policy
  hybrid_ctrl.run_hybrid        -- orchestrator (plan -> track loop)
  eval_hybrid.main              -- 10 random + 10 dp eval harness
"""
__version__ = "0.1.0"
