"""Compare saved wall-aware checkpoints on the same CFD and physics samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator

from nozzle_performance import performance
from predict_wall_aware import load_model
from train_four_loss import LX, LR, evaluate, load_data


CHECKPOINTS = ("transferred_no_training", "stage1", "stage2", "stage3")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--wall", type=Path, default=Path("propulsion_pinn/results/data/wall_nodes.csv"))
    p.add_argument("--run-dir", type=Path,
                   default=Path("propulsion_pinn/results/checkpoints/wall_aware_output"))
    p.add_argument("--validation-kpa", type=int, default=650)
    p.add_argument("--ambient-pa", type=float, default=101325.0)
    p.add_argument("--stages", nargs="+", default=list(CHECKPOINTS))
    p.add_argument("--model-report", type=Path,
                   help="model architecture report; defaults to the run's report.json")
    args = p.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cases, *_ = load_data(args.data_dir)
    if args.validation_kpa not in cases:
        p.error("validation pressure must be an existing CFD case")
    case = cases[args.validation_kpa]
    nodes = pd.read_csv(args.wall).sort_values("x_m")
    wall = PchipInterpolator(nodes.x_m, nodes.r_m)
    coords = case["x"]
    gap = wall(coords[:, 0] * LX) - coords[:, 1] * LR
    eligible = coords[:, 1] * LR > 0.001
    bands = {"near_wall_0_5_um": (gap >= 0) & (gap < 5e-6),
             "wall_5_50_um": (gap >= 5e-6) & (gap < 50e-6),
             "outer_50_500_um": (gap >= 50e-6) & (gap < 500e-6),
             "core_over_500_um": gap >= 500e-6}
    rng = np.random.default_rng(9)
    samples = {name: coords[rng.choice(np.flatnonzero(mask & eligible),
                                      size=min(128, int((mask & eligible).sum())),
                                      replace=False)]
               for name, mask in bands.items()}
    from train_four_loss import physics_losses  # expensive derivatives, imported on demand

    results = {"validation_kpa": args.validation_kpa, "ambient_pa": args.ambient_pa,
               "checkpoint_comparison": []}
    for name in args.stages:
        model, _, device = load_model(args.wall, args.run_dir / f"{name}.pt",
                                     args.model_report or args.run_dir / "report.json")
        metrics = evaluate(model, case, None, device)
        perf = performance(model, device, float(nodes.x_m.iloc[-1]),
                           float(nodes.r_m.iloc[-1]), args.validation_kpa,
                           args.ambient_pa)
        physics = {}
        for band, sample in samples.items():
            continuity, momentum = physics_losses(model, torch.from_numpy(sample).to(device))
            physics[band] = {"continuity_normalized_mse": float(continuity.detach()),
                             "momentum_normalized_mse": float(momentum.detach()),
                             "sampled_cells": len(sample)}
        results["checkpoint_comparison"].append({"stage": name,
            "radial_relative_l2_percent": metrics["radial_velocity"]["relative_l2_percent"],
            "radial_rmse_m_s": metrics["radial_velocity"]["rmse"],
            "pressure_relative_l2_percent": metrics["pressure"]["relative_l2_percent"],
            "pressure_rmse_pa": metrics["pressure"]["rmse"],
            "exit_mass_flow_kg_s": perf["mass_flow_kg_s"],
            "exit_thrust_n": perf["thrust_n"], "physics": physics})
    output = args.run_dir / "checkpoint_comparison.json"
    output.write_text(json.dumps(results, indent=2))
    for row in results["checkpoint_comparison"]:
        print(row["stage"], "radial_pct", round(row["radial_relative_l2_percent"], 3),
              "pressure_pct", round(row["pressure_relative_l2_percent"], 3),
              "near_wall_cont", round(row["physics"]["near_wall_0_5_um"]["continuity_normalized_mse"], 3),
              "near_wall_mom", round(row["physics"]["near_wall_0_5_um"]["momentum_normalized_mse"], 3))
    print("Saved", output)


if __name__ == "__main__":
    main()
