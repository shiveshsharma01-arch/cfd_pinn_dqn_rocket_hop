# Why the current nozzle PINN misses radial velocity

Audit date: 2026-11-07. This is a diagnosis and test plan; no new model was
trained. Evidence comes from the saved Stage 4 checkpoint and the original CFD
field exports. The complete read-only measurements are in
`results/checkpoints/focused_output/diagnostics.json` and can be reproduced with
`diagnose_focused.py`.

## Measured findings

The selected Stage 4 checkpoint has 18.81% radial relative L2 error and
4.483 m/s radial RMSE at the 650 kPa development case. Stage 5 has 18.10%
radial error, but pressure error rises to 2.27%, so Stage 4 remains selected.
For a 10% radial relative L2 target at 650 kPa, RMSE must fall below about
2.383 m/s.

Radial relative L2 remains 18.63–19.21% on the seven *training* pressures.
The near-equal training and validation errors show that the dominant present
problem is fitting the available fields, rather than a shortage of pressure
operating points. Direct interpolation between the 600 and 700 kPa CFD fields
has 0.566% radial error at 650 kPa.

At 650 kPa, the core (radius below half the reconstructed local wall radius)
has 8.46% radial error, while the outer 20% of radius has 22.24%. The wall
boundary probe predicts 105.94 m/s axial RMS and 20.68 m/s radial RMS at the
*reconstructed cell-envelope wall*. The no-slip target is zero. These probe
coordinates are approximate because actual wall-face coordinates were not
exported; nonetheless the boundary condition is plainly not well enforced in
the current implementation. At cell centres within 0.001% of the reconstructed
wall radius, the reference axial RMS is about 14.84 m/s, indicating the wall
approximation is not the only cause of the large predicted speed.

## Likely causes, in order of evidence

1. **Weak and approximate wall constraint.** The current boundary loss divides
   wall speeds by 500 m/s, then averages wall, axis and inlet losses. This
   makes even a large wall speed a modest loss. Wall locations come from the
   outermost *cell centres*, rather than the Fluent wall faces. The largest
   radial error occurs where the wall and boundary layer matter most.
2. **Under-resolved radial structure in the learned function.** The one-piece
   Tanh network uses only normalized x, r and pressure, without local wall
   distance or a boundary-layer coordinate. The lower core error and high
   near-wall error are consistent with this limitation. This is a hypothesis
   to test, not a demonstrated sole cause.
3. **Approximate physics unlike the Fluent target.** The PINN omits energy and
   SST k-omega transport, learns turbulent viscosity without enforcing its
   transport closure, and uses constant molecular viscosity. Its momentum
   residual therefore need not match Fluent's equations and material settings.
   Physics collocation uses 64 sampled cell centres per optimizer step and
   excludes the axis neighbourhood; this may leave local constraints weak.
4. **Loss competition.** The phased schedule helps, but the joint objective can
   still trade wall, data and momentum accuracy against each other. Increasing
   continuity weight alone previously failed to improve radial accuracy.

## Targeted experiments while keeping phased training

1. **Export exact wall-face coordinates and confirm Fluent model settings.**
   Preserve all current CFD field cases. Check axis, inlet and exit conditions,
   viscosity law, and turbulence model against the PINN equations.
2. **Run one wall-focused ablation.** Use exact wall points, report wall-speed
   RMS after every stage, and strengthen wall loss in physical velocity units.
   Compare with the current checkpoint at equal training duration. A
   distance-function construction that makes velocity zero at the true wall is
   an alternative if the geometry is accurate and remains compatible with the
   inlet/axis constraints.
3. **Add a wall-distance feature or local wall-normal coordinate.** Keep the
   phased schedule. Test a modest wider network or separate near-wall branch,
   with the same data split and training budget. Compare near-wall and core
   errors, not only the full-field mean.
4. **Then test physics changes one at a time.** Increase/adapt physics points
   near the wall and throat. Check residuals separately by region. Implement
   exact material and governing-equation terms only after confirming Fluent
   settings. More epochs alone are a weak first experiment.
5. **Only request new CFD cases if pressure interpolation becomes limiting.**
   The current training-case radial errors are already around 19%. Additional
   pressures do not directly repair this spatial fitting problem. A new 950
   kPa run is valuable as an independent test after the design is frozen.

Success gate: radial relative L2 below 10% on 650 kPa *and* a separate unseen
pressure case, with acceptable pressure and exit-integrated mass flow/thrust
errors. The 650 kPa case has already guided model selection, so it alone cannot
establish final generalization. A sub-10% result is a target, not a guarantee
from any listed modification.

## Research basis

- Krishnapriyan et al., [Characterizing possible failure modes in physics-informed neural networks](https://papers.neurips.cc/paper_files/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html): difficult PINN optimization and curriculum regularization.
- Wang, Teng and Perdikaris, [Understanding and Mitigating Gradient Flow Pathologies in PINNs](https://epubs.siam.org/doi/10.1137/20M1318043): imbalance among physics-loss gradients.
- Sukumar and Srivastava, [Exact imposition of boundary conditions with distance functions](https://www.sciencedirect.com/science/article/pii/S0045782521006186): enforcing boundary values by construction.
- Xiao et al., [RANS-based PINN for turbulent rocket nozzle flow](https://www.sciencedirect.com/science/article/pii/S1270963825016621): nozzle-specific use of SST equations, wall-distance weighting and coarse-to-fine data.
- [Ansys Fluent Theory Guide](https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/flu_th/flu_th.html): energy and turbulence transport in Fluent.
