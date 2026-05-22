"""
find_showcase_seed.py -- pick a dp seed where PPO iter15 visually
outperforms BC v2 K=1, for the side-by-side and iter-progression videos.

The aggregate PPO improvement is real (dp p95 max_field 1.357 -> 0.319,
-76%), but on a per-seed basis the gain is not uniform: some seeds the
BC student already navigates perfectly. dp seed 6 was the original
pick (it is the seed used in the existing multi-modal MLP-vs-diffusion
video) but BC happens to handle it well (max_field 0.045), so a
side-by-side there visually contradicts the paper claim.

This script:
  1. Loads per-seed eval JSONs for BC v2 K=1, PPO iter5, iter10, iter15,
     and PPO T=0.5 iter15.
  2. For each of the 10 dp seeds, computes the metric tuple
     (BC max_field, PPO iter15 max_field, PPO iter15 goal err) and
     scores the seed against the brief's filters:
        BC max_field > 0.6        # BC visibly grazes obstacle
        PPO iter15 max_field<0.3  # PPO is clean
        PPO iter15 goal err<80cm  # PPO actually arrives (=800 mm)
  3. Among the seeds that pass, ranks by:
        a) monotonicity of PPO max_field decrease across iter5/10/15
           (a perfect monotonic drop scores higher)
        b) gap = BC max_field - PPO iter15 max_field (bigger gap is more
           visually compelling)
  4. Prints a table of all seeds with each metric and the chosen
     showcase seed.

Usage:
  python distillation/find_showcase_seed.py
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_RES = _ROOT / 'results'


def _per_seed_max_and_goal(path: Path, controller: str | None = None) -> dict:
    """Return {seed_str -> (max_field, goal_err_mm)}.

    `controller` is the key into per_seed_summary when the file is the
    multi-controller dp eval (e.g. decision_point_eval_v2_single.json);
    None means the file is single-controller (per_seed directly).
    """
    if not path.exists():
        return {}
    d = json.load(open(path))
    if controller is None:
        # single-controller phase-2 eval
        per_seed = d.get('per_seed', {})
        out = {}
        for s, row in per_seed.items():
            out[s] = (float(row['max_field']), float(row['goal_err_mm']))
        return out
    else:
        per = d.get('per_seed_summary', d.get('per_seed', {}))
        out = {}
        for s, rows in per.items():
            if controller in rows:
                r = rows[controller]
                out[s] = (float(r['max_field']), float(r['goal_err_mm']))
        return out


def main():
    bc = _per_seed_max_and_goal(
        _RES / 'decision_point_eval_v2_single.json',
        controller='Diffusion_BC')
    ppo5 = _per_seed_max_and_goal(_RES / 'diffusion_v2_ppo_phase2_iter5_dp.json')
    ppo10 = _per_seed_max_and_goal(_RES / 'diffusion_v2_ppo_phase2_iter10_dp.json')
    ppo15 = _per_seed_max_and_goal(_RES / 'diffusion_v2_ppo_phase2_dp.json')
    ppo15_t05 = _per_seed_max_and_goal(
        _RES / 'diffusion_v2_ppo_T05_iter15_dp.json')

    seeds = sorted(set(bc) & set(ppo5) & set(ppo10) & set(ppo15) & set(ppo15_t05),
                   key=lambda s: int(s))
    print(f"{'seed':>4} | {'BC field':>9} {'BC goal':>9} | "
          f"{'i5 field':>9} {'i10 field':>10} {'i15 field':>10} {'i15 goal':>9} | "
          f"{'T05 field':>10} {'T05 goal':>9}")
    print('-' * 110)
    rows = []
    for s in seeds:
        bc_f, bc_g = bc[s]
        f5, _ = ppo5[s]
        f10, _ = ppo10[s]
        f15, g15 = ppo15[s]
        f5t, g5t = ppo15_t05[s]
        passes = (bc_f > 0.6 and f15 < 0.3 and g15 < 800.0)
        mono = (f5 >= f10 >= f15)
        gap = bc_f - f15
        rows.append(dict(
            seed=int(s), bc_field=bc_f, bc_goal=bc_g,
            i5_field=f5, i10_field=f10, i15_field=f15, i15_goal=g15,
            t05_field=f5t, t05_goal=g5t,
            passes=passes, monotonic=mono, gap=gap,
        ))
        marker = ''
        if passes:
            marker += ' PASS'
        if mono:
            marker += ' MONO'
        print(f"{s:>4} | {bc_f:9.3f} {bc_g:9.1f} | "
              f"{f5:9.3f} {f10:10.3f} {f15:10.3f} {g15:9.1f} | "
              f"{f5t:10.3f} {g5t:9.1f}{marker}")

    # Filter and rank
    candidates = [r for r in rows if r['passes']]
    if not candidates:
        # Relax: drop the goal cap if no seed satisfies all 3
        candidates = [r for r in rows
                      if r['bc_field'] > 0.6 and r['i15_field'] < 0.3]
        print(f"\n[relaxed: no seed met all 3 filters; "
              f"dropped goal-err cap]")
    if not candidates:
        candidates = [r for r in rows
                      if r['bc_field'] > 0.6 and r['i15_field'] < 0.4]
        print(f"[relaxed: also widened i15 field cap to 0.4]")
    if not candidates:
        print('NO CANDIDATE FOUND -- pick by largest gap instead')
        candidates = sorted(rows, key=lambda r: -r['gap'])[:3]

    # Rank: monotonic first, then by gap
    candidates.sort(key=lambda r: (not r['monotonic'], -r['gap']))
    print(f"\n[candidates]")
    for r in candidates:
        print(f"  seed {r['seed']:>4d}: BC field {r['bc_field']:.3f} -> "
              f"PPO i15 field {r['i15_field']:.3f} (gap {r['gap']:.3f}); "
              f"monotonic={r['monotonic']}; i15 goal {r['i15_goal']:.0f} mm")
    pick = candidates[0]
    print(f"\nSHOWCASE SEED: {pick['seed']}")
    print(f"  BC v2 K=1:        max_field {pick['bc_field']:.3f}  "
          f"goal {pick['bc_goal']:.1f} mm")
    print(f"  PPO iter5:        max_field {pick['i5_field']:.3f}")
    print(f"  PPO iter10:       max_field {pick['i10_field']:.3f}")
    print(f"  PPO iter15 T=0.1: max_field {pick['i15_field']:.3f}  "
          f"goal {pick['i15_goal']:.1f} mm")
    print(f"  PPO iter15 T=0.5: max_field {pick['t05_field']:.3f}  "
          f"goal {pick['t05_goal']:.1f} mm")
    print(f"  gap (BC - PPO):   {pick['gap']:.3f}; monotonic? {pick['monotonic']}")

    # Save the analysis
    out = _RES / 'showcase_seed.json'
    json.dump(dict(
        showcase_seed=int(pick['seed']),
        showcase_metrics={k: v for k, v in pick.items()
                          if k not in ('passes', 'monotonic')},
        showcase_monotonic=bool(pick['monotonic']),
        showcase_gap=float(pick['gap']),
        all_seeds=rows,
    ), open(out, 'w'), indent=2)
    print(f"\n -> {out}")
    return pick['seed']


if __name__ == '__main__':
    main()
