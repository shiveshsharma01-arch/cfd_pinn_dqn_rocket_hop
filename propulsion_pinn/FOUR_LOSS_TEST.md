# Four-loss PINN: first staged test

Training cases: 600, 700, 750, 800, 850, 900, 1000 kPa. Development
validation: 650 kPa, excluded from training and normalization. All cases use
the same 30,892-cell mesh and 300 K inlet total temperature.

The first test used a 4-hidden-layer, width-96 Tanh network. It predicts axial
velocity, radial velocity, static pressure, static temperature, turbulent
kinetic energy, and turbulent viscosity. Density follows the ideal-gas law.

Losses are CFD field data, axisymmetric continuity, approximate compressible
RANS axial/radial momentum, and boundary conditions. The boundary points on the
wall were reconstructed from the outer envelope of the 800 kPa cell centres;
the reconstructed minimum radius is 24.300 mm. Inlet total conditions, no-slip
wall velocity, and axis radial velocity are applied. The momentum loss uses a
constant molecular viscosity and learned turbulent viscosity and k. Energy and
SST transport equations are not enforced.

## Results on 650 kPa

| Checkpoint | Radial velocity relative L2 | Radial RMSE | Static pressure relative L2 | Pressure RMSE | Pressure MAPE |
|---|---:|---:|---:|---:|---:|
| Stage I: data + BC | 20.38% | 4.856 m/s | 2.21% | 8,861 Pa | 2.12% |
| Stage II: add continuity and momentum | 22.70% | 5.409 m/s | 3.81% | 15,266 Pa | 3.02% |
| Stage III: adaptive weights | **19.30%** | **4.598 m/s** | **1.33%** | **5,336 Pa** | **1.16%** |
| Stage IV: L-BFGS | 19.38% | 4.619 m/s | 1.36% | 5,463 Pa | 1.29% |
| Direct interpolation of 600 and 700 kPa CFD | **0.566%** | **0.135 m/s** | **0.0326%** | **131 Pa** | — |

Relative L2 error is `100*||prediction-CFD||_2/||CFD||_2`; it remains defined
when individual radial velocities are near zero. Pressure MAPE is the mean
absolute percent error across cells. Stage III minimized the sum of radial and
pressure relative L2 percentages among the trained checkpoints. Stage IV
slightly worsened both, so `best.pt` contains Stage III.

At the end of Stage III, data and boundary weights were both 1.0; adaptive
continuity and momentum weights were 0.2828 and 0.1448. Stage I trained for
1,000 steps, Stage II for 2,000, Stage III for 3,000, and Stage IV allowed 100
L-BFGS iterations. The raw settings, metrics and checkpoints are in
`results/checkpoints/four_loss_output/report.json` and the adjacent `.pt` files.

## Decision

This result confirms the four-loss training pipeline runs, but the current PINN
does not beat pressure interpolation at 650 kPa. It is not yet suitable as the
thrust source for DQN. The 650 kPa case was used to select the checkpoint, so a
separate unseen CFD case would still be required for final model assessment.
The current tests do not assess exit-integrated thrust.

## Continuity-weight follow-up

The Stage III checkpoint was continued for 1,000 Adam steps with the same seed,
data/boundary weights (1.0), momentum weight (0.1448), learning rate (1e-4),
and batches. Only the continuity weight changed:

| Continuity weight | Radial relative L2 | Radial RMSE | Pressure relative L2 |
|---:|---:|---:|---:|
| 0.2828 (control) | **19.58%** | **4.667 m/s** | **1.139%** |
| 0.6 | 19.64% | 4.680 m/s | 1.237% |
| 1.0 | 19.70% | 4.695 m/s | 1.334% |

Increasing the continuity weight reduced its training-batch residual but did
not improve either held-out metric. The original Stage III checkpoint remains
the best overall checkpoint from the four-stage run (19.30% radial and 1.332%
pressure relative L2). The follow-up checkpoints and metrics are stored in
`results/checkpoints/continuity_sweep_output` and do not replace `results/checkpoints/four_loss_output/best.pt`.

For the original Stage III model, radial RMSE by nozzle region was 5.776 m/s
in the converging region (x < 55 mm), 3.287 m/s around the throat
(55–80 mm), and 3.791 m/s in the diverging region (x ≥ 80 mm). The throat
had the largest region-normalized relative L2 error, 27.05%.
