# CFD data inventory

These are copies of the user's Fluent ASCII exports; their checksums are in `../PACKAGE_MANIFEST.csv`. The raw files retain Fluent's original, extensionless names.

| Folder | Contents | Use |
|---|---|---|
| `raw/fields/` | Nine full cell-centre cases, 600 to 1000 kPa in 50 kPa steps | Seven train cases, 650 kPa development selection, 950 kPa independent field test |
| `raw/outlets/` | Nine outlet-surface profiles | Outlet audit; the 750 kPa file has coordinates only |
| `raw/wall/` | Fluent wall-node coordinates | Wall-aware PINN geometry |
| `raw/mass_flow_screenshots/` | Supplied Fluent inlet/outlet reports | Evidence for the mass-balance audit; the 600 kPa screenshot is absent |

The full-field files contain x/r coordinates, density, pressure, temperature, turbulence quantities, axial/radial velocity and Mach number. The documented model uses 300 K air and a fixed CFD ambient condition. The original Fluent Workbench project and solver binaries are not included. Read `../flight_model/CFD_OUTLET_AUDIT.md` before interpreting thrust: the supplied mass-flow reports have 1.07–3.70% imbalance and do not validate the thrust map.
