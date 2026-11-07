# Frozen nozzle field surrogate and exit performance

`best.pt` is the frozen 800-step mass-consistency continuation of the wall-aware
Stage 3 model. `field_best_stage3.pt` preserves the prior checkpoint with lower
local near-wall equation residuals. The selected model predicts axial/radial velocity, static pressure,
static temperature, density, turbulent kinetic energy and turbulent viscosity
for points in the nozzle at inlet total pressures from 600 to 1000 kPa.

`exit_performance.csv` contains PINN-derived exit mass flow, momentum flux and
pressure integral at nine inlet pressures. Its ambient absolute pressure is
**0 Pa**, so the `thrust_n` column is the formal vacuum-pressure result. For
another ambient pressure, use

`thrust(p_inlet, p_ambient) = thrust_vac(p_inlet) - p_ambient * exit_area_m2`.

This uses the rocket exhaust thrust balance for an onboard gas supply. It is
not a direct calculation of force on the nozzle wall alone.

The exit area is 0.00313706876017 m² for the 31.6 mm exit radius. The model
samples the radial profile with 256-point Gauss-Legendre quadrature and uses
the 2πr axisymmetric area factor. Pressure must be **absolute**, in pascals.
For engine-off, set mass flow and thrust to zero separately; the PINN has no
training data below 600 kPa.

At the 650 kPa development case, radial-velocity relative L2 error is 1.899%
and static-pressure relative L2 error is 0.654%. The same case was used for
checkpoint selection. The largest cross-section mass-flow mismatch at the
checked pressures is below 0.8%, but the sampled near-wall local continuity
and momentum residuals remain large. See `physics_regions.json` and
`mass_consistency_comparison.json`. The CFD energy and SST transport equations
are not fully enforced by this model.

**Status:** reproducible cold-air field surrogate and provisional thrust map.
The independent 950 kPa **field** test is recorded in `independent_950.json`.
Exit-integrated mass flow and thrust have not been validated against Fluent outlet
surface reports. The
model's ambient-pressure adjustment assumes the internal exit profile remains
unchanged as ambient pressure varies; shocks or separation may violate this.
