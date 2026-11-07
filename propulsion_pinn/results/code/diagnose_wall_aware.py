"""Check the selected wall-aware checkpoint across cases and boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_wall_aware import WallAwareNet
from train_four_loss import LX, LR, TRAIN_KPA, VAL_KPA, evaluate, load_data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--wall", type=Path, default=Path("propulsion_pinn/results/data/wall_nodes.csv"))
    p.add_argument("--checkpoint", type=Path, default=Path("propulsion_pinn/results/checkpoints/wall_aware_output/best.pt"))
    p.add_argument("--report", type=Path, default=Path("propulsion_pinn/results/checkpoints/wall_aware_output/report.json"))
    p.add_argument("--output", type=Path, default=Path("propulsion_pinn/results/checkpoints/wall_aware_output/diagnostics.json"))
    args = p.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cases, _, _, mean, std = load_data(args.data_dir)
    nodes = pd.read_csv(args.wall).sort_values("x_m")
    settings = json.loads(args.report.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WallAwareNet(mean, std, nodes.x_m.to_numpy(), nodes.r_m.to_numpy(),
                         settings["wall_scale_um"] * 1e-6).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True)["state_dict"])
    result = {"selected_stage": settings["best_stage"], "cases": {}, "wall": {}}
    for kpa in TRAIN_KPA + (VAL_KPA,):
        metrics = evaluate(model, cases[kpa], None, device)
        result["cases"][str(kpa)] = {field: metrics[field]
                                      for field in ("axial_velocity", "radial_velocity", "pressure", "temperature")}
        pts = np.column_stack((nodes.x_m.to_numpy() / LX, nodes.r_m.to_numpy() / LR,
                               np.full(len(nodes), kpa / 1000))).astype(np.float32)
        with torch.no_grad():
            ux, ur, *_ = model.physical(torch.from_numpy(pts).to(device))
            result["wall"][str(kpa)] = {
                "axial_rms_m_s": float(torch.sqrt(torch.mean(ux.square()))),
                "radial_rms_m_s": float(torch.sqrt(torch.mean(ur.square()))),
                "max_speed_m_s": float(torch.sqrt(ux.square() + ur.square()).max()),
            }
    args.output.write_text(json.dumps(result, indent=2))
    print("selected", settings["best_stage"])
    for kpa in TRAIN_KPA + (VAL_KPA,):
        c = result["cases"][str(kpa)]
        w = result["wall"][str(kpa)]
        print(kpa, "radial_pct", round(c["radial_velocity"]["relative_l2_percent"], 4),
              "pressure_pct", round(c["pressure"]["relative_l2_percent"], 4),
              "wall_max_speed", round(w["max_speed_m_s"], 6))


if __name__ == "__main__":
    main()
