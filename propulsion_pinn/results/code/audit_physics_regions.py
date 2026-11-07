"""Measure PINN equation residual losses by distance from the exact wall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator

from train_wall_aware import WallAwareNet
from train_four_loss import LX, LR, load_data, physics_losses


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("propulsion_pinn/results/checkpoints/final/best.pt"))
    p.add_argument("--report", type=Path,
                   default=Path("propulsion_pinn/results/checkpoints/final/report.json"))
    p.add_argument("--output", type=Path,
                   default=Path("propulsion_pinn/results/checkpoints/final/physics_regions.json"))
    args = p.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cases, _, _, mean, std = load_data(args.data_dir)
    nodes = pd.read_csv("propulsion_pinn/results/data/wall_nodes.csv").sort_values("x_m")
    spline = PchipInterpolator(nodes.x_m, nodes.r_m)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = json.loads(args.report.read_text())
    model = WallAwareNet(mean, std, nodes.x_m.to_numpy(), nodes.r_m.to_numpy(),
                         settings["wall_scale_um"] * 1e-6).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    x = cases[650]["x"]
    gap = spline(x[:, 0] * LX) - x[:, 1] * LR
    eligible = x[:, 1] * LR > 0.001
    bands = {"0_5_um": (gap >= 0) & (gap < 5e-6),
             "5_50_um": (gap >= 5e-6) & (gap < 50e-6),
             "50_500_um": (gap >= 50e-6) & (gap < 500e-6),
             "over_500_um": gap >= 500e-6}
    rng = np.random.default_rng(9)
    out = {}
    for name, mask in bands.items():
        ids = np.flatnonzero(mask & eligible)
        chosen = rng.choice(ids, size=min(128, len(ids)), replace=False)
        c, m = physics_losses(model, torch.from_numpy(x[chosen]).to(device))
        out[name] = {"available_cells": len(ids), "sampled": len(chosen),
                     "mean_squared_continuity": float(c.detach()),
                     "mean_squared_momentum": float(m.detach())}
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
