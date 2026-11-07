"""Test stronger continuity weights from the same Stage III nozzle checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_four_loss import (
    NozzleNet, VAL_KPA, boundary_loss, boundary_points, data_loss, evaluate,
    load_data, physics_losses, raw_four, sample_batch, sample_physics,
    wall_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("propulsion_pinn/results/checkpoints/four_loss_output/stage3.pt"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("propulsion_pinn/results/checkpoints/continuity_sweep_output"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases, train_x, train_y, mean, std = load_data(args.data_dir)
    wall, wall_info = wall_profile(cases[800]["x"][:, :2])
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    initial_cont = float(saved["weights"]["continuity"])
    momentum_weight = float(saved["weights"]["momentum"])
    weights = (initial_cont, 0.6, 1.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = {600: raw_four(cases[600]), 700: raw_four(cases[700])}
    report = {"source_checkpoint": str(args.checkpoint), "steps_per_candidate": args.steps,
              "momentum_weight_fixed": momentum_weight, "data_weight": 1.0,
              "boundary_weight": 1.0, "wall": wall_info,
              "validation_pressure_kpa": VAL_KPA, "candidates": []}
    for continuity_weight in weights:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = NozzleNet(mean, std).to(device)
        model.load_state_dict(saved["state_dict"])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for step in range(1, args.steps + 1):
            model.train()
            bx, by = sample_batch(train_x, train_y, mean, std, rng, 1024, device)
            px = sample_physics(train_x, rng, 64, device)
            bp = boundary_points(128, wall, rng, device)
            dloss = data_loss(model, bx, by)
            bloss = boundary_loss(model, bp)
            closs, mloss = physics_losses(model, px)
            total = dloss + bloss + continuity_weight * closs + momentum_weight * mloss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            if step == args.steps or step % 500 == 0:
                print("lambda_cont", round(continuity_weight, 4), "step", step,
                      "data", round(float(dloss.detach()), 5),
                      "continuity", round(float(closs.detach()), 5),
                      "momentum", round(float(mloss.detach()), 5), flush=True)
        metrics = evaluate(model, cases[VAL_KPA], baseline, device)
        item = {"continuity_weight": continuity_weight, "momentum_weight": momentum_weight,
                "metrics": metrics}
        report["candidates"].append(item)
        torch.save({"state_dict": model.state_dict(), "weights": {"data": 1.0,
                    "boundary": 1.0, "continuity": continuity_weight,
                    "momentum": momentum_weight}, "validation": metrics},
                   args.output_dir / f"continuity_{continuity_weight:.3f}.pt")
        print("lambda_cont", round(continuity_weight, 4), "radial_rel_l2_pct",
              round(metrics["radial_velocity"]["relative_l2_percent"], 4),
              "pressure_rel_l2_pct",
              round(metrics["pressure"]["relative_l2_percent"], 4), flush=True)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    best = min(report["candidates"], key=lambda item:
               item["metrics"]["radial_velocity"]["relative_l2_percent"] +
               item["metrics"]["pressure"]["relative_l2_percent"])
    report["best_continuity_weight_by_joint_metric"] = best["continuity_weight"]
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
