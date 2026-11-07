# Nozzle PINN results package

This folder preserves the Python source, trained checkpoints, wall geometry,
and numerical reports for the nozzle surrogate experiments.

| Folder | Contents |
|---|---|
| `code/` | Python training, prediction, evaluation, and diagnostic scripts |
| `data/wall_nodes.csv` | 353 wall nodes exported from the shared Fluent mesh |
| `checkpoints/<run>/` | Experiment checkpoints with JSON reports and diagnostic files |
| `checkpoints/final/` | Frozen selected checkpoint and PINN-derived exit performance map |

The frozen field-surrogate checkpoint is `checkpoints/final/best.pt`. Its
training results, thrust calculation and physics-consistency limits are
described in `../FINAL_PINN_RESULTS.md`.
Directories containing `smoke` hold short setup tests; the other directories
hold the full experiments. No checkpoint has been discarded.

Run commands from the `Intern` workspace folder so the default paths resolve:

```powershell
python propulsion_pinn/results/code/predict_wall_aware.py --input-points points.csv --inlet-kpa 950 --output predicted_950.csv
python propulsion_pinn/results/code/nozzle_performance.py --inlet-kpa 600 800 1000 --ambient-pa 101325 --output thrust_example.csv
```

`points.csv` needs `x_m,r_m` columns in metres and points inside the nozzle.
Inference uses the checkpoint and wall nodes in this package. Retraining also
uses the original Fluent field exports in `data/raw/fields` at the project root. A prediction at 950 kPa
is not a CFD validation result.

The root `PACKAGE_MANIFEST.csv` lists packaged paths, sizes and SHA-256 checksums.
The recorded training environment was Python 3.13.15, NumPy 2.4.4, pandas
3.0.3, SciPy 1.16.2 and PyTorch 2.6.0+cu124. CUDA availability is optional
for inference; a compatible PyTorch installation is required.
