# CFD outlet dataset audit — 7 November 2026

Source: `data/raw/outlets` and `data/raw/mass_flow_screenshots` in this package.
Each numbered `*kpa` file was read as an ASCII Fluent outlet-surface export.
Mass-flow values were transcribed from the correspondingly numbered Fluent
Flux Reports screenshots. The 600 kPa mass-flow screenshot was not supplied.

All nine exports have 91 outlet faces at approximately `x=0.170 m`, with
radial face centres from about `0.000251` to `0.031599 m`. Duplicate coordinate
columns in eight exports are harmless. **The 750 kPa export has only face
coordinates; density, axial velocity and absolute pressure are absent.**

Fluent uses a negative sign for flow leaving through the outlet in these
reports. Net = inlet + signed outlet. The imbalance percentage below uses
`abs(net)/inlet * 100`.

| Inlet pressure (kPa) | Inlet (kg/s) | Outlet signed (kg/s) | Net (kg/s) | Imbalance of inlet | PINN mass flow (kg/s) |
|---:|---:|---:|---:|---:|---:|
| 600 | — | — | — | — | 2.5584 |
| 650 | 2.776749 | -2.879525 | -0.102775 | 3.70% | 2.7650 |
| 700 | 2.992307 | -3.085293 | -0.092986 | 3.11% | 2.9865 |
| 750 | 3.208632 | -3.291604 | -0.082971 | 2.59% | 3.2064 |
| 800 | 3.426721 | -3.498845 | -0.072124 | 2.10% | 3.4219 |
| 850 | 3.645343 | -3.707188 | -0.061846 | 1.70% | 3.6355 |
| 900 | 3.863005 | -3.916675 | -0.053671 | 1.39% | 3.8482 |
| 950 | 4.078417 | -4.126841 | -0.048425 | 1.19% | 4.0566 |
| 1000 | 4.291527 | -4.337521 | -0.045993 | 1.07% | 4.2554 |

The PINN mass flow is closer to the reported *inlet* mass flow than the
reported outlet flow, but these screenshots alone cannot identify the cause.
Every reported case has outlet magnitude greater than inlet flow. A steady
nozzle with impermeable walls should not retain this discrepancy. Residual
convergence alone does not resolve the conservation check.

Before replacing the provisional PINN flight map or reporting validated CFD
thrust:

1. Re-export the **750 kPa** outlet with density, axial velocity and absolute
   pressure, keeping `Cell Center` and selecting only `outlet` under Surfaces.
2. Supply the **600 kPa** inlet/outlet mass-flow report.
3. Recompute inlet and outlet fluxes from the *same final case/data* for each
   pressure. Check that no mass source or additional open boundary is present.
   Continue or repair runs whose net imbalance persists. Aim below 0.5% at a
   minimum, preferably below 0.1%, while checking thrust and pressure monitors.
4. Supply outlet face areas or Fluent outlet reports for mass-weighted axial
   velocity, area-weighted **absolute** static pressure, and outlet area. The
   present profiles do not include face areas, so exact discrete thrust
   integration has not yet been established from these files alone.

`results/cfd_mass_flow_reports.csv` preserves the screenshot transcription;
`results/cfd_outlet_audit.json` preserves the parsed file checks and comparison
with the frozen PINN map. The flight-model map was **not** changed by this audit.
