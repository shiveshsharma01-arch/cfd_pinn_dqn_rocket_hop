# Proposed DQN mission: vertical hop and controlled landing

**Status:** historical mission-design proposal, written before the landing
controller was implemented. See `PROJECT_STATUS.md` and
`results/landing/final/validation_summary.json` for the current measured
outcome. All results use the provisional PINN-derived 300 K air nozzle map,
not CFD-validated thrust or real flight data.

## Mission objective

Launch vertically, reach a requested peak height of **500 m ± 50 m**, then
touch down at **≤ 5 m/s downward speed** with **at least 5 kg of propellant
remaining**. The stretch touchdown goal is **≤ 2 m/s**. The flight continues
through descent; apogee is not the end of the episode. This is a vertical-hop
demonstrator, not orbital insertion or a real reusable booster design.

For the first version, use the current illustrative 100 kg initial mass, 50 kg
dry mass, 0.5 drag coefficient and 0.01 m² frontal area. Keep pitch at zero
throughout the hop, so the nozzle thrust points vertically. The two-dimensional
flight model remains available for later horizontal-flight studies.

## What the current model can reach

With a 1000 kPa vertical ascent burn and no landing burn, the model gives:

| Ascent burn | Approximate peak height | Fuel used before coast |
|---:|---:|---:|
| 3 s | 155 m | 12.8 kg |
| 4 s | 291 m | 17.0 kg |
| 5 s | 482 m | 21.3 kg |
| 6 s | 736 m | 25.5 kg |

Burning all 50 kg at 1000 kPa reaches about **3.91 km** in this vertical
model, but the unpowered return reaches the ground at about **251 m/s**. That
maximum-height case does not reserve any landing propellant.

The previous apogee DQN used roughly 47.6 kg of the available 50 kg on average
by apogee, which leaves too little margin for this landing task. It remains an
apogee-control baseline, not a landing policy.

An **idealized hand-coded braking rule** was tested with a 5.0 s ascent burn,
900 kPa braking pressure, 0.05 s control decisions and 0.01 s physics steps.
It reached **482.25 m**, touched down at about **2.33 m/s**, and retained about
**13.09 kg** of propellant. It started braking five times. Halving the physics
step while keeping the same control interval changed touchdown speed by about
0.002 m/s. This demonstrates nominal feasibility of the 5 m/s goal within the
current simulator; it does not establish robustness or actuator feasibility.
The peak-height error is **3.55% of the 500 m target**; the touchdown speed is
**46.6% of the allowed 5 m/s limit**. Remaining propellant is **26.2% of the
initial 50 kg supply**.

The earlier apparent ≤2 m/s result was sensitive to floating-point timing of
the ascent cutoff. The cutoff was changed to use the control-tick index, and
the corrected 300-case sweep with 0.05 s decisions and 0.01 s physics steps
found **no** ≤2 m/s touchdown. Do not present that stricter goal as achieved.

## Controller and training design

- **Actions:** engine off, or 600, 700, 800, 900 or 1000 kPa inlet pressure.
- **Decision interval:** 0.05 s; **physics step:** 0.01 s. These are simulation
  choices. A real valve would need a measured response time.
- **Observations:** height, vertical speed, vehicle mass/fuel, requested peak
  height, peak height so far, ambient pressure, elapsed time, and actual inlet
  pressure if actuator lag is introduced.
- **Episode termination:** ground contact, fuel exhaustion with no recoverable
  landing, or a time limit. The vehicle must not be reset at apogee.
- **Reward priorities:** (1) touchdown speed and avoiding a crash, (2) peak
  height accuracy, (3) adequate fuel reserve, (4) total fuel and needless
  pressure switching. Weights need to be set and tested before training.
- **Baselines:** the hand-coded braking rule, a fixed-pressure ascent followed
  by coast, and the existing apogee DQN as a diagnostic only. All landing
  comparisons must share the same vehicle, map, atmosphere and actuator model.

Use a staged training curriculum with one eventual policy: (1) reliable
vertical ascent with fuel reservation, (2) descent and braking examples,
(3) complete launch-to-touchdown episodes, (4) randomized vehicle mass,
target height, thrust-map error and sensor/actuator effects. Keep training,
development selection and final evaluation scenarios separate. Train several
seeds and report all of them, including failures.

## Validation gates before a landing claim

1. Replace or check the provisional PINN thrust map against Fluent outlet-face
   thrust; the present Fluent mass reports have 1.07–3.70% imbalance.
2. Verify height, impact speed and fuel use with a smaller *physics* step while
   holding the control interval fixed.
3. Add a pressure actuator response and test whether the five braking starts
   remain possible. The current engine on/off response is instantaneous.
4. Test ±10% thrust-map variation, different launch masses, small sensor noise
   and at least a few initial temperature/ambient-pressure scenarios.
5. Compare DQN with the simple braking rule on the same held-out missions.

This scope retains the aerospace propulsion + AI integration while making the
flight objective achievable with the available low-temperature nozzle model.
