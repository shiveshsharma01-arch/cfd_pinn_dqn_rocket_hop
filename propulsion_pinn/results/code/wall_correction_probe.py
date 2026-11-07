"""Test whether an exact-wall velocity correction helps the existing PINN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator

from train_focused import FocusedNet
from train_four_loss import LX, LR, TRAIN_KPA, VAL_KPA, load_data, raw_four


def predict(model: FocusedNet, x: np.ndarray, device: torch.device) -> np.ndarray:
    out = []
    model.eval()
    with torch.no_grad():
        for part in np.array_split(x, 32):
            fields = model.physical(torch.from_numpy(part).to(device))
            out.append(torch.stack(fields[:4], dim=1).cpu().numpy())
    return np.concatenate(out)


def error(a: np.ndarray, b: np.ndarray) -> float:
    return float(100 * np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--wall", type=Path, default=Path("propulsion_pinn/results/data/wall_nodes.csv"))
    p.add_argument("--checkpoint", type=Path, default=Path("propulsion_pinn/results/checkpoints/focused_output/best.pt"))
    p.add_argument("--output", type=Path, default=Path("propulsion_pinn/results/checkpoints/focused_output/wall_correction_probe.json"))
    args = p.parse_args()
    cases, _, _, mean, std = load_data(args.data_dir)
    wall_df = pd.read_csv(args.wall).sort_values("x_m")
    wall = PchipInterpolator(wall_df.x_m, wall_df.r_m)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FocusedNet(mean, std).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True)["state_dict"])
    results = {}
    for kpa in TRAIN_KPA + (VAL_KPA,):
        case = cases[kpa]
        xy = case["x"].copy()
        wx = xy.copy()
        wx[:, 1] = (wall(xy[:, 0] * LX) / LR).astype(np.float32)
        pred = predict(model, xy, device)
        wall_pred = predict(model, wx, device)
        truth = raw_four(case)
        gap = np.maximum(wall(xy[:, 0] * LX) - xy[:, 1] * LR, 0)
        variants = {"uncorrected": {"radial_percent": error(pred[:, 1], truth[:, 1]),
                                    "axial_percent": error(pred[:, 0], truth[:, 0])}}
        for scale in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
            decay = np.exp(-gap / scale)
            corrected = pred[:, :2] - decay[:, None] * wall_pred[:, :2]
            variants[f"{scale:g}"] = {"radial_percent": error(corrected[:, 1], truth[:, 1]),
                                        "axial_percent": error(corrected[:, 0], truth[:, 0])}
        results[str(kpa)] = variants
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results[str(VAL_KPA)], indent=2))


if __name__ == "__main__":
    main()
