"""Continue the wall-aware PINN with an integral mass-conservation loss."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator

from predict_wall_aware import load_model
from train_focused import batches, weighted_data_loss
from train_four_loss import LX, LR, TRAIN_KPA, VAL_KPA, boundary_loss, boundary_points, evaluate, load_data, raw_four


def section_mass_flow(model, pressures_kpa: torch.Tensor, section_x_m: torch.Tensor,
                      section_r_m: torch.Tensor, quadrature_r: torch.Tensor,
                      quadrature_w: torch.Tensor) -> torch.Tensor:
    """Return [pressure, axial-section] mass-flow matrix in kg/s."""
    npres, nx, nr = pressures_kpa.numel(), section_x_m.numel(), quadrature_r.numel()
    radius = section_r_m[None, :, None] * quadrature_r[None, None, :]
    radial_weights = section_r_m[None, :, None] * quadrature_w[None, None, :]
    radius = radius.expand(npres, -1, -1)
    radial_weights = radial_weights.expand(npres, -1, -1)
    x = section_x_m[None, :, None].expand(npres, nx, nr)
    pressure = (pressures_kpa / 1000)[:, None, None].expand(npres, nx, nr)
    locations = torch.stack((x / LX, radius / LR, pressure), dim=-1).reshape(-1, 3)
    ux, _, _, _, rho, *_ = model.physical(locations)
    area_weights = 2 * torch.pi * radius * radial_weights
    return torch.sum((rho * ux).reshape(npres, nx, nr) * area_weights, dim=-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--wall", type=Path, default=Path("propulsion_pinn/results/data/wall_nodes.csv"))
    p.add_argument("--input-checkpoint", type=Path,
                   default=Path("propulsion_pinn/results/checkpoints/final/best.pt"))
    p.add_argument("--report", type=Path,
                   default=Path("propulsion_pinn/results/checkpoints/final/report.json"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("propulsion_pinn/results/checkpoints/mass_consistency_output"))
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--mass-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()
    if args.steps < 1 or args.mass_weight <= 0:
        p.error("steps and mass-weight must be positive")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    rng = np.random.default_rng(args.seed)
    cases, train_x, train_y, mean, std = load_data(args.data_dir)
    model, nodes, device = load_model(args.wall, args.input_checkpoint, args.report)
    wall = PchipInterpolator(nodes.x_m, nodes.r_m)
    wall_r = wall(train_x[:, 0] * LX)
    x_samples = np.array((0, 0.005, 0.010, 0.030, 0.060, 0.100, LX))
    section_x = torch.tensor(x_samples, dtype=torch.float32, device=device)
    section_r = torch.tensor(wall(x_samples), dtype=torch.float32, device=device)
    quad_x, quad_w = np.polynomial.legendre.leggauss(32)
    quad_r = torch.tensor(0.5 * (quad_x + 1), dtype=torch.float32, device=device)
    quad_w = torch.tensor(0.5 * quad_w, dtype=torch.float32, device=device)
    base = {600: raw_four(cases[600]), 700: raw_four(cases[700])}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    record = {"source_checkpoint": str(args.input_checkpoint), "seed": args.seed,
              "steps": args.steps, "mass_weight": args.mass_weight,
              "section_x_m": x_samples.tolist(), "checkpoints": []}
    best_score, best_state, best_name = float("inf"), None, None

    def snapshot(name: str) -> None:
        nonlocal best_score, best_state, best_name
        metrics = evaluate(model, cases[VAL_KPA], base, device)
        with torch.no_grad():
            mass = section_mass_flow(model, torch.tensor([600., 650., 800., 1000.], device=device),
                                     section_x, section_r, quad_r, quad_w).cpu().numpy()
        mismatch = 100 * np.max(np.abs(mass[:, :-1] / mass[:, -1:] - 1), axis=1)
        rad = metrics["radial_velocity"]["relative_l2_percent"]
        press = metrics["pressure"]["relative_l2_percent"]
        score = rad / 2.0329647064208984 + press / 0.7286771535873413 + float(np.mean(mismatch)) / 10
        row = {"name": name, "radial_percent": rad, "pressure_percent": press,
               "max_section_mismatch_percent": dict(zip((600, 650, 800, 1000), mismatch.tolist())),
               "joint_score": score}
        record["checkpoints"].append(row)
        torch.save({"state_dict": model.state_dict(), "stage": name,
                    "metrics": metrics}, output / f"{name}.pt")
        if score < best_score:
            best_score, best_state, best_name = score, copy.deepcopy(model.state_dict()), name
        (output / "report.json").write_text(json.dumps(record, indent=2))
        print(name, "radial_pct", round(rad, 4), "pressure_pct", round(press, 4),
              "max_mismatch_pct_800", round(float(mismatch[2]), 3), flush=True)

    snapshot("source")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for step in range(1, args.steps + 1):
        bx, by, _ = batches(train_x, train_y, mean, std, wall_r, rng, 1536, 32, True, device)
        bp = boundary_points(64, wall, rng, device)
        pressures = torch.tensor(rng.choice(TRAIN_KPA, size=3, replace=False),
                                 dtype=torch.float32, device=device)
        flows = section_mass_flow(model, pressures, section_x, section_r, quad_r, quad_w)
        mass_loss = torch.mean(((flows[:, :-1] - flows[:, -1:]) / flows[:, -1:]) ** 2)
        data_loss = weighted_data_loss(model, bx, by, 3.0, 2.0)
        bc_loss = boundary_loss(model, bp)
        mass_weight = args.mass_weight * min(1.0, step / max(1, args.steps // 2))
        loss = data_loss + bc_loss + mass_weight * mass_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 400 == 0 or step == args.steps:
            print(step, "data", round(float(data_loss.detach()), 6),
                  "mass", round(float(mass_loss.detach()), 6), flush=True)
            snapshot(f"step{step}")
    model.load_state_dict(best_state)
    best_row = next(row for row in record["checkpoints"] if row["name"] == best_name)
    torch.save({"state_dict": model.state_dict(), "stage": best_name,
                "metrics": best_row}, output / "best.pt")
    record["best_stage"] = best_name
    (output / "report.json").write_text(json.dumps(record, indent=2))
    print("best", best_name, flush=True)


if __name__ == "__main__":
    main()
