"""Read-only regional error audit of the selected focused PINN checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_focused import FocusedNet
from train_four_loss import LX, LR, TRAIN_KPA, VAL_KPA, boundary_points, load_data, raw_four, wall_profile


def stats(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    diff = pred - true
    return {
        "count": int(len(true)),
        "reference_rms_m_s": float(np.sqrt(np.mean(true**2))),
        "rmse_m_s": float(np.sqrt(np.mean(diff**2))),
        "relative_l2_percent": float(100 * np.linalg.norm(diff) / np.linalg.norm(true)),
        "mean_bias_m_s": float(np.mean(diff)),
    }


def predict(model: FocusedNet, case: dict, device: torch.device) -> np.ndarray:
    out = []
    model.eval()
    with torch.no_grad():
        for batch in np.array_split(case["x"], 32):
            out.append(torch.stack(model.physical(torch.from_numpy(batch).to(device))[:4], 1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("propulsion_pinn/results/checkpoints/focused_output/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("propulsion_pinn/results/checkpoints/focused_output/diagnostics.json"))
    args = parser.parse_args()
    cases, _, _, mean, std = load_data(args.data_dir)
    wall, _ = wall_profile(cases[800]["x"][:, :2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FocusedNet(mean, std).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True)["state_dict"])
    report = {"checkpoint": str(args.checkpoint), "cases": {}, "validation_regions": {}}
    with torch.no_grad():
        wall_pts, axis_pts, _ = boundary_points(2048, wall, np.random.default_rng(42), device)
        wux, wur, *_ = model.physical(wall_pts)
        _, aur, *_ = model.physical(axis_pts)
        report["boundary_probe"] = {
            "wall_axial_rms_m_s": float(torch.sqrt(torch.mean(wux.square()))),
            "wall_radial_rms_m_s": float(torch.sqrt(torch.mean(wur.square()))),
            "axis_radial_rms_m_s": float(torch.sqrt(torch.mean(aur.square()))),
        }
    for kpa in TRAIN_KPA + (VAL_KPA,):
        case = cases[kpa]
        pred = predict(model, case, device)
        true = raw_four(case)
        report["cases"][str(kpa)] = {
            "radial": stats(pred[:, 1], true[:, 1]),
            "pressure_relative_l2_percent": float(100 * np.linalg.norm(pred[:, 2] - true[:, 2]) / np.linalg.norm(true[:, 2])),
        }
        if kpa != VAL_KPA:
            continue
        x = case["x"][:, 0] * LX
        r = case["x"][:, 1] * LR
        relative_radius = r / wall(x)
        masks = {
            "converging_x_lt_55mm": x < 0.055,
            "throat_55_to_80mm": (x >= 0.055) & (x < 0.080),
            "diverging_x_ge_80mm": x >= 0.080,
            "core_r_over_wall_lt_0.5": relative_radius < 0.5,
            "outer_r_over_wall_ge_0.8": relative_radius >= 0.8,
            "low_radial_abs_lt_5ms": np.abs(true[:, 1]) < 5,
            "high_radial_abs_ge_5ms": np.abs(true[:, 1]) >= 5,
        }
        report["validation_regions"] = {name: stats(pred[mask, 1], true[mask, 1])
                                         for name, mask in masks.items() if mask.any()}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
