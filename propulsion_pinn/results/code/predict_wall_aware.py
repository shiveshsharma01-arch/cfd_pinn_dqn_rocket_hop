"""Predict nozzle fields from the selected wall-aware surrogate checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator

from train_wall_aware import WallAwareNet
from train_four_loss import GAMMA, LX, LR, R_GAS


def load_model(wall_path: Path, checkpoint_path: Path, report_path: Path
               ) -> tuple[WallAwareNet, pd.DataFrame, torch.device]:
    """Load the frozen field surrogate without the original CFD datasets."""
    nodes = pd.read_csv(wall_path).sort_values("x_m")
    settings = json.loads(report_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WallAwareNet(np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32),
                         nodes.x_m.to_numpy(), nodes.r_m.to_numpy(),
                         settings["wall_scale_um"] * 1e-6).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                     weights_only=True)["state_dict"])
    model.eval()
    return model, nodes, device


def predict_fields(model: WallAwareNet, inputs: np.ndarray,
                   device: torch.device) -> np.ndarray:
    """Return ux, ur, p, T, rho, k and turbulent viscosity for normalized inputs."""
    arrays = []
    with torch.no_grad():
        for part in np.array_split(inputs, max(1, (len(inputs) + 4095) // 4096)):
            arrays.append(torch.stack(model.physical(torch.from_numpy(part).to(device)), 1).cpu().numpy())
    return np.concatenate(arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-points", type=Path, required=True,
                        help="CSV with x_m and r_m columns")
    parser.add_argument("--inlet-kpa", type=float, required=True,
                        help="Same inlet-total-pressure parameter as training, 600–1000 kPa")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall", type=Path, default=Path("propulsion_pinn/results/data/wall_nodes.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("propulsion_pinn/results/checkpoints/final/best.pt"))
    parser.add_argument("--report", type=Path, default=Path("propulsion_pinn/results/checkpoints/final/report.json"))
    args = parser.parse_args()
    if not 600 <= args.inlet_kpa <= 1000:
        parser.error("inlet-kpa must be within the 600–1000 kPa training range")
    points = pd.read_csv(args.input_points)
    if not {"x_m", "r_m"}.issubset(points):
        parser.error("input-points needs x_m and r_m columns in metres")
    model, nodes, device = load_model(args.wall, args.checkpoint, args.report)
    x_m, r_m = points.x_m.to_numpy(), points.r_m.to_numpy()
    radius = PchipInterpolator(nodes.x_m, nodes.r_m)(x_m)
    if not (np.isfinite(x_m).all() and np.isfinite(r_m).all()):
        parser.error("coordinates must be finite")
    if np.any((x_m < -1e-8) | (x_m > LX + 1e-8) |
              (r_m < -1e-8) | (r_m > radius + 1e-8)):
        parser.error("points must lie inside the nozzle fluid domain")
    inputs = np.column_stack((x_m / LX, r_m / LR,
                              np.full(len(points), args.inlet_kpa / 1000))).astype(np.float32)
    fields = predict_fields(model, inputs, device)
    out = points.copy()
    for i, name in enumerate(("axial_velocity_m_s", "radial_velocity_m_s",
                              "static_pressure_Pa", "static_temperature_K",
                              "density_kg_m3", "turbulent_kinetic_energy_m2_s2",
                              "turbulent_viscosity_Pa_s")):
        out[name] = fields[:, i]
    out["mach_number"] = np.sqrt(fields[:, 0] ** 2 + fields[:, 1] ** 2) / np.sqrt(
        GAMMA * R_GAS * fields[:, 3])
    out.to_csv(args.output, index=False)
    print(f"Predicted {len(out)} points at {args.inlet_kpa:g} kPa -> {args.output}")


if __name__ == "__main__":
    main()
