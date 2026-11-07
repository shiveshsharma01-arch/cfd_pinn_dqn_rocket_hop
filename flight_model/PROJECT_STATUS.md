# Project status — 7 November 2026

## Coding stage

The 500 m vertical takeoff-to-touchdown simulator, DQN training run, guided
landing controller, comparison, and Blender animations are complete.
The recommended **hybrid** uses the trained DQN for ascent pressure commands
and a model-based feedback rule for descent. The DQN alone is not a reliable
landing controller. The final report remains.

| Component | Evidence and status |
|---|---|
| CFD nozzle fields | 600–1000 kPa cases, including 950 kPa; thrust validation remains provisional because mass imbalance is 1.07–3.70% and the 750 kPa outlet profile is incomplete. |
| Frozen PINN | Independent 950 kPa field relative L2 errors: 1.029% axial speed, 2.346% radial speed, 0.531% static pressure, 0.277% static temperature. Direct interpolation is more accurate at 950 kPa and must be disclosed. |
| Flight model | Two-dimensional point mass with gravity, drag, atmospheric pressure, propellant depletion and a simplified actuator; the landing mission fixes pitch vertically. |
| DQN | One 120,000-step, reference-seeded/behavior-cloning-assisted DQN checkpoint. It controls five pressure settings plus off and is frozen for evaluation. |
| Guided descent | 900 kPa stopping-distance rule; below 20 m, a downward-speed envelope improves terminal control. |

## Deterministic 500 m validation sweep

All three policies use the same flight dynamics and nozzle map. The sweep has
21 initial masses from 99.0 to 101.0 kg in 0.1 kg increments. A case succeeds
if peak height is within **10%** of 500 m, touchdown is at or below 5 m/s,
and at least 5 kg fuel remains. The sweep is simulator evidence, not flight or
CFD validation. It was used during controller tuning, so it is not a strictly
untouched test set.

| Policy | Success | Mean peak-height error | Mean touchdown speed as % of 5 m/s limit | Worst touchdown speed as % of limit | Minimum fuel left |
|---|---:|---:|---:|---:|---:|
| DQN alone | 4.76% | 3.36% | 215.47% | 288.36% | 5.75 kg |
| **DQN ascent + guided descent** | **100%** | **3.36%** | **24.15%** | **38.04%** | **10.40 kg** |
| Timed ascent + same guided descent | 100% | 2.40% | 28.25% | 45.86% | 10.35 kg |

The hybrid lands more slowly than the timed-ascent reference in this sweep,
but the reference has lower mean peak-height error. Do not claim the DQN is
overall better or that a pure DQN lands the rocket.

For the nominal 100 kg case, halving the physics step from 0.01 to 0.005 s
changes peak height by 0.00009 m (**0.000018% of the 500 m target**) and
touchdown speed by 0.00158 m/s (**0.032 percentage points of the 5 m/s
limit**). Eight flight-model and landing integration tests pass.

## Files to use next

- `finalize_landing.py` recreates the three-policy evaluation and the final
  package under `results/landing/final/`.
- `results/landing/final/selected_dqn.pt` is the frozen trained checkpoint.
- `results/landing/final/policy_manifest.json` records the controller and
  nozzle-map provenance.
- `results/landing/final/validation_cases.csv` and
  `validation_summary.json` contain the full comparison.
- `results/landing/final/*_nominal_trace.csv` are ready to drive animation;
  each row has time, height, vertical speed, mass/fuel, actuator opening and
  pressure command.
- `results/landing/seed0_v3/step*_trace.csv` records the training milestones
  for a learning-progress animation. These traces include unsuccessful early
  and final pure-DQN runs.

## Limits before any real-rocket claim

The nozzle map comes from 300 K air CFD at a fixed CFD ambient condition.
The flight model corrects pressure thrust as ambient pressure changes but
assumes the internal nozzle solution does not change. The vehicle, pitch,
actuator lag and drag are demonstration assumptions. No uncertainty sweep of
thrust, sensors or actuator timing, no flight test, and no independent CFD
thrust validation have been completed. The current claim is a **CFD-informed
vertical-hop control demonstrator**, suitable for an honest animation and
report with the limitations shown.
