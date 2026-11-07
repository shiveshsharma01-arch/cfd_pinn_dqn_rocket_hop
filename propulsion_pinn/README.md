# Nozzle field surrogate: first training stage

The complete Python and checkpoint archive is in [`results/`](results/README.md).
The frozen current model is `results/checkpoints/final/best.pt`.
The original Fluent field exports are needed for retraining or validation and
are included under `data/raw/fields` in the GitHub staging package.

See `FINAL_PINN_RESULTS.md` for the exit thrust integration, current errors,
residual audit and validation limits.

This first model uses the seven CFD field exports at 600, 700, 750, 800, 850,
900 and 1000 kPa. The 650 kPa field is held out for validation. The network
predicts axial/radial velocity, static pressure and static temperature from
axial/radial position and inlet total pressure. Density is calculated from the
ideal-gas relation. Its physics loss uses automatic differentiation to penalize
violation of **steady, axisymmetric mass continuity**.

This is a partial physics-informed surrogate. It does **not** yet implement
compressible momentum, energy or SST turbulence transport equations. It must
not be described as a full RANS PINN or a validated rocket engine model.

Run from the Intern folder:

```powershell
python propulsion_pinn/results/code/nozzle_pinn.py --data-dir data/raw/fields --steps 2000
```

The script saves the trained weights and a JSON report under
`propulsion_pinn/results/checkpoints/output`. The report compares the PINN with simple interpolation
on the held-out 650 kPa case. Only the training cases determine normalization
and model weights. The original Fluent exports remain untouched.

For a staged comparison, run:

```powershell
python propulsion_pinn/results/code/train_phases.py --data-dir data/raw/fields
```

`train_phases.py` screens Tanh and SiLU with Adam and field-data loss, continues
the better validation candidate with a mass-continuity loss, then applies LBFGS
on a fixed batch. It saves each checkpoint and chooses the best stage by 650 kPa
field error. This 650 kPa case is a **development validation case**; the independent
950 kPa CFD field test is reported in `FINAL_PINN_RESULTS.md`. The report also
shows the error of direct pressure interpolation, which must remain the
benchmark for selecting a surrogate for DRL.

See `CODE_WALKTHROUGH.md` for a plain-language explanation and the measured
results of the first staged experiment.

`train_four_loss.py` runs the requested four-stage test with data, boundary,
continuity, and approximate compressible RANS momentum losses. Its results and
metric definitions are recorded in `FOUR_LOSS_TEST.md`.

`train_focused.py` runs the radial-then-pressure five-stage curriculum using the
same four loss groups and separate output heads. Its measured errors, settings,
and limitations are recorded in `FOCUSED_TEST.md`; the selected checkpoint is
`results/checkpoints/focused_output/best.pt`.

`train_wall_aware.py` continues the staged model with the actual exported wall
nodes, wall-distance features and a nearly exact no-slip output correction.
`WALL_AWARE_TEST.md` reports the under-10% field result and the remaining
near-wall physics limitation. `predict_wall_aware.py` loads the selected
checkpoint for field inference within the training pressure range.

Before DRL, evaluate whether the PINN predicts exit-integrated mass flow,
momentum and pressure accurately. The current cell-centre exports do not contain
exit face areas; use Fluent exit-surface reports or export the exit faces for
that check. The existing five-row workbook covers five operating points; add
750 and 850 kPa integrated results when available. A 950 kPa CFD run would be
useful as an independent final test, but is not required to start training.
