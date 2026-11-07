# Wall-aware PINN continuation: measured result and limits

This records the original Stage 3 wall-aware experiment. The later
mass-consistency continuation and the currently frozen checkpoint are in
`FINAL_PINN_RESULTS.md` and `results/checkpoints/final/`.

The wall geometry is the 353-node Fluent export in `results/data/wall_nodes.csv`. The seven
training fields are 600, 700, 750, 800, 850, 900, and 1000 kPa. The 650 kPa
field remains the development validation case. The model starts from the
previous Stage 4 checkpoint and retains a shared Tanh trunk with separate
velocity, thermodynamic and turbulence heads. It adds local radius and
log-distance-to-wall input features. A rapidly decaying correction subtracts
the model's velocity at the exact wall nodes, making the output nearly no-slip.
Float32 geometry evaluation leaves at most 0.16 m/s of measured wall speed.

## Staged training results

| Checkpoint | Radial relative L2 at 650 kPa | Pressure relative L2 at 650 kPa |
|---|---:|---:|
| Previous selected PINN | 18.81% | 1.180% |
| Wall correction, no retraining | 14.46% | 1.180% |
| Stage 1: wall-aware data and boundary continuation | 2.70% | 1.152% |
| Stage 2: ramp continuity and momentum | 2.22% | 0.827% |
| **Stage 3: coupled refinement; selected** | **2.03%** | **0.729%** |

Stage 1 ran for 1,000 Adam steps; Stage 2 for 1,500; Stage 3 for 2,500.
Radial-data weight was 4 then 3; pressure-data weight was 1 then 2.
Continuity and momentum terms entered gradually. The code checks the 650 kPa
field after each stage and saves the best balanced checkpoint as
`results/checkpoints/wall_aware_output/best.pt`.

Radial relative L2 on the seven *training* pressures ranges from 2.16% to
2.66%; pressure relative L2 ranges from 0.48% to 0.66%. At 650 kPa, the radial
RMSE and pressure RMSE are in `results/checkpoints/wall_aware_output/report.json`. The diagnostics
for all eight cases and wall speeds are in
`results/checkpoints/wall_aware_output/diagnostics.json`.

## Physics consistency warning

This achieves the requested **under-10% CFD-field error target**, but it is
not yet a fully physics-consistent nozzle solver. The adaptive continuity and
momentum weights at the selected stage are approximately 1.9e-7 and 1.7e-6.
On 650 kPa sampled points, normalized mean-squared continuity/momentum losses
were 1252/170 within 5 micrometres of the wall and 0.043/0.010 beyond 500
micrometres. Thus the near-wall derivative correction introduces large local
equation residuals, and the optimization effectively gives physics losses little
influence there. See `results/checkpoints/wall_aware_output/physics_regions.json`.

The model also omits the Fluent energy and SST transport equations and uses
approximate material properties. The 650 kPa case selected the checkpoint, so
it is development validation rather than an independent final test. No CFD
truth is available at 950 kPa. The saved 950 kPa wall prediction is an
unverified inference example, not a validation result.

## Usage

To predict the field at points inside the nozzle, provide a CSV containing
`x_m,r_m` in metres and run, for example:

```powershell
python propulsion_pinn/results/code/predict_wall_aware.py --input-points points.csv --inlet-kpa 950 --output predicted_950.csv
```

The script reads only the checkpoint and wall nodes. It restricts the inlet
parameter to the 600–1000 kPa training interval and points to the nozzle
fluid domain. Predictions still need comparison with a CFD case unseen during
model development. Before DQN uses model-derived thrust, compare exit mass
flow, momentum and pressure-thrust integrals with Fluent reports.
