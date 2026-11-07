# Finalized PINN code and current verification status

The frozen inference checkpoint is `results/checkpoints/final/best.pt`.
`results/code/predict_wall_aware.py` predicts full fields, while
`results/code/nozzle_performance.py` integrates the exit profile to supply
mass flow and thrust for a specified absolute ambient pressure. Training and
diagnostic scripts remain in `results/code/`; the CFD exports remain in the
`data/raw/fields` folder in this package.

## Selected checkpoint

The final checkpoint is the 800-step mass-consistency continuation of the
wall-aware Stage 3 model. The earlier Stage 3 model is preserved as
`results/checkpoints/final/field_best_stage3.pt`. Both were evaluated on the
same 650 kPa CFD field and the same 128-point near-wall residual sample:

| Checkpoint | Radial relative L2 | Pressure relative L2 | Max section mass-flow mismatch at 800 kPa | Continuity / momentum normalized MSE, 0–5 µm |
|---|---:|---:|---:|---:|
| Original Stage 3 | 2.03% | 0.729% | 7.73% | 1,252 / 170 |
| Mass continuation, 400 steps | 2.04% | 0.908% | 0.37% | 1,959 / 408 |
| **Mass continuation, 800 steps; frozen** | **1.90%** | **0.654%** | **0.61%** | **3,452 / 570** |
| Mass continuation, 1200 steps | 1.84% | 0.702% | 0.14% | 5,999 / 676 |

At the 650 kPa development case the frozen model's radial RMSE is about
0.4526 m/s and pressure RMSE is about 2620 Pa. For the 600–1000 kPa training
pressures, radial relative L2 ranges from 2.03% to 2.45% and pressure relative
L2 from 0.43% to 0.57%. The 650 kPa checkpoint-selection case is
development validation, not an independent final test. Direct interpolation
between the adjacent 600/700 kPa CFD fields is still more accurate at 650 kPa.

An **independent 950 kPa CFD field** is now available and was evaluated after
the checkpoint was frozen. Over its 30,892 cells, the PINN relative L2 errors
are **1.029% axial velocity, 2.346% radial velocity, 0.531% static pressure,
and 0.277% static temperature**. Linear interpolation of the 900/1000 kPa CFD
fields has lower errors on the same case: **0.027%, 0.054%, 0.0014%, and
0.0055%**, respectively. Thus the PINN predicts fields reasonably in this
pressure range but is not the most accurate available surrogate at 950 kPa.
The saved comparison is `results/checkpoints/final/independent_950.json`.

The mass-consistency loss reduced inlet-to-exit mass-flow mismatch to below
0.8% for the checked 600, 650, 800 and 1000 kPa conditions. This is an
integral conservation check, distinct from the local PDE residual. The frozen
model's sampled near-wall continuity and momentum residuals **worsened**
relative to original Stage 3. The table's equation residuals are normalized
**mean-squared loss values, not percent errors**. The model remains a
data-assisted, partially physics-informed surrogate and does not solve
Fluent's full energy and SST transport system. See `WALL_AWARE_TEST.md` and
`results/checkpoints/final/mass_consistency_comparison.json`.

## Thrust calculation

For the axisymmetric exit radius 0.0316 m, the code computes

\[\dot m=2\pi\int_0^{R_e}\rho u_x r\,dr,\qquad
F=2\pi\int_0^{R_e}[\rho u_x^2+(p_e-p_a)]r\,dr.\]

This profile integration uses the PINN's density, axial velocity and **absolute
static pressure** at the exit. It uses 256 Gauss-Legendre points. At 800 kPa,
increasing the count from 256 to 512 changed mass flow by approximately
0.00000005 kg/s and thrust by less than 0.00001 N at 101325 Pa ambient.
The integration also passed uniform-profile analytic checks for mass flow,
momentum thrust, pressure thrust and the ambient-pressure effect.

| Inlet total pressure | PINN mass flow | PINN thrust at 101325 Pa ambient |
|---:|---:|---:|
| 600 kPa | 2.558 kg/s | 1,214 N |
| 800 kPa | 3.422 kg/s | 1,744 N |
| 1000 kPa | 4.255 kg/s | 2,253 N |

The full sea-level table is in
`results/checkpoints/final/thrust_map_sea_level.csv`. The frozen
ambient-independent map in `results/checkpoints/final/exit_performance.csv`
uses 0 Pa ambient. The controller can adjust any row by subtracting
`p_ambient * 0.00313706876017 m²`. This assumes ambient pressure does not
change the internal nozzle flow. Overexpanded, shocked or separated regimes
need additional CFD cases before this approximation can be trusted.

**These thrust values are PINN predictions, not validated CFD thrust values.**
The next verification is comparison with Fluent outlet-face integrated mass
flow, momentum and pressure force. The supplied Fluent inlet/outlet mass-flow
reports have 1.07–3.70% imbalance across 650–1000 kPa, and the 750 kPa outlet
profile contains coordinates only. Those issues limit a thrust-validation
claim even though the independent 950 kPa **field** comparison is complete.

## Commands from the Intern workspace

```powershell
python propulsion_pinn/results/code/predict_wall_aware.py --input-points points.csv --inlet-kpa 800 --output predicted_points.csv
python propulsion_pinn/results/code/nozzle_performance.py --inlet-kpa 600 800 1000 --ambient-pa 101325 --output propulsion_pinn/results/checkpoints/final/sea_level_example.csv
python propulsion_pinn/results/code/evaluate_checkpoints.py --data-dir data/raw/fields --run-dir propulsion_pinn/results/checkpoints/mass_consistency_output --model-report propulsion_pinn/results/checkpoints/final/report.json --stages source step400 step800 step1200
```

Thrust equation reference: [NASA Glenn](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/specific-impulse/).
