# Radial-then-pressure PINN training test

This experiment trained on the 600, 700, 750, 800, 850, 900, and 1000 kPa
cell-centre CFD fields. The 650 kPa field was held out for development
validation. All fields use the same 30,892-cell nozzle mesh. The run used seed 0
and a GPU. No CFD export was changed.

The model has a shared four-layer Tanh trunk (width 128) and separate output
heads for velocity, thermodynamic variables, and turbulence variables. Inputs
are axial position, radial position, and inlet total pressure. Outputs are
axial and radial velocity, static pressure, static temperature, turbulent
kinetic energy, and turbulent viscosity. Density follows the ideal-gas law.
The four loss groups are CFD data, reconstructed boundary conditions,
axisymmetric continuity, and approximate compressible RANS momentum. Energy
and SST transport equations are not enforced.

## Training schedule

The field-data weights below apply to standardized output errors. Each stage
also uses boundary weight 1. The continuity and momentum weights multiply
their own normalized residual losses, so their numbers are not directly
comparable with a single field-data weight.

| Stage | Optimizer steps | Radial data weight | Pressure data weight | Continuity weight | Momentum weight |
|---|---:|---:|---:|---:|---:|
| 1: field and boundaries | 1,000 Adam | 1 | 1 | 0 | 0 |
| 2: radial focus | 2,000 Adam | 4 | 1 | ramp 0 to 0.1 | 0 |
| 3: pressure focus | 2,000 Adam | 2 | 4 | 0.1 | ramp 0 to 0.05 |
| 4: coupled refinement | 4,000 Adam | 2 | 2 | adaptive, ends 0.2901 | adaptive, ends 0.1470 |
| 5: fixed-batch refinement | up to 100 L-BFGS iterations | 2 | 2 | 0.2901 | 0.1470 |

Other data weights: axial velocity 1, temperature 1, turbulent kinetic energy
0.25, turbulent viscosity 0.25. Stages 2–5 sample half their data uniformly,
one quarter in the throat region, and one quarter near the wall. Every stage
still supervises *all* outputs. The focused stages adjust relative emphasis;
they do not solve radial velocity and pressure independently.

## Results on 650 kPa

| Model/stage | Radial relative L2 | Radial RMSE | Pressure relative L2 | Pressure RMSE | Pressure MAPE |
|---|---:|---:|---:|---:|---:|
| 1: field and boundaries | 21.96% | 5.232 m/s | 2.035% | 8,152 Pa | 1.705% |
| 2: radial focus | 19.29% | 4.596 m/s | 2.303% | 9,226 Pa | 2.555% |
| 3: pressure focus | 19.35% | 4.611 m/s | 1.378% | 5,521 Pa | 1.178% |
| **4: coupled refinement; selected** | **18.81%** | **4.483 m/s** | **1.180%** | **4,726 Pa** | **1.026%** |
| 5: L-BFGS | 18.10% | 4.314 m/s | 2.266% | 9,076 Pa | 1.376% |
| Earlier four-loss model | 19.30% | 4.598 m/s | 1.332% | 5,336 Pa | 1.156% |
| Direct interpolation of 600/700 kPa CFD | **0.566%** | **0.135 m/s** | **0.0326%** | **131 Pa** | — |

Relative L2 is `100 × ||prediction − CFD||₂ / ||CFD||₂`, calculated over all
30,892 cells. Pressure MAPE averages absolute percentage errors per cell.
Stage 4 improves on the earlier four-loss model by about 0.48 percentage
points in radial relative L2 and 0.15 points in pressure relative L2. Stage 5
has the lowest radial error among these checkpoints, but its pressure error
rises sharply. Stage 4 minimizes the equal-relative score against the earlier
four-loss checkpoint and is saved as `results/checkpoints/focused_output/best.pt`.

## Interpretation and next decision

The staged emphasis worked directionally: stage 2 reduced radial error and
stage 3 reduced pressure error. It did not produce an accurate enough field
surrogate for the control task. Direct interpolation of the adjacent CFD cases
is substantially more accurate on this 650 kPa validation case. The 650 kPa
field also selected the checkpoint, so it is not an independent final test.

No exit-integrated thrust or mass-flow prediction was evaluated here. Before
using any surrogate as the DQN environment, obtain exit-face areas or Fluent
surface reports and compare mass flow, momentum flux, pressure thrust, and
derived thrust across the CFD cases. A separate pressure case such as 950 kPa
would provide an independent assessment after model choices are frozen.

The wall boundary in the loss is reconstructed from cell-centre coordinates,
not exact wall-face coordinates. Momentum uses learned turbulent viscosity and
k but omits turbulence transport. These limitations constrain any claim that
the network reproduces the full Fluent solution.

Detailed settings, all metrics, and checkpoints are in `results/checkpoints/focused_output`.
