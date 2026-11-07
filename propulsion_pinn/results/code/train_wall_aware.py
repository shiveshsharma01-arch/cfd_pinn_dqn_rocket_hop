"""Continue the staged nozzle PINN with exact wall nodes and hard no-slip output."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator
from torch import nn

from train_focused import FocusedNet, batches, weighted_data_loss
from train_four_loss import (
    LX, LR, TRAIN_KPA, VAL_KPA, boundary_loss, boundary_points, evaluate,
    gradient_norm, load_data, physics_losses, raw_four,
)


class WallAwareNet(FocusedNet):
    """Spatial wall features plus exact no-slip at the supplied wall curve."""

    def __init__(self, mean: np.ndarray, std: np.ndarray,
                 wall_x: np.ndarray, wall_r: np.ndarray, wall_scale_m: float):
        super().__init__(mean, std, width=128)
        self.trunk[0] = nn.Linear(5, 128)
        spline = PchipInterpolator(wall_x, wall_r)
        self.register_buffer("wall_knots", torch.tensor(spline.x, dtype=torch.float32))
        self.register_buffer("wall_coeff", torch.tensor(spline.c, dtype=torch.float32))
        self.wall_scale_m = wall_scale_m

    def wall_radius(self, x_m: torch.Tensor) -> torch.Tensor:
        index = torch.searchsorted(self.wall_knots[1:-1], x_m.contiguous()).clamp(
            0, self.wall_knots.numel() - 2)
        dx = x_m - self.wall_knots[index]
        coeff = self.wall_coeff[:, index]
        return ((coeff[0] * dx + coeff[1]) * dx + coeff[2]) * dx + coeff[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        physical = self.raw_physical(x)
        r_m = x[:, 1] * LR
        radius = self.wall_radius(x[:, 0] * LX)
        dist = (radius - r_m).clamp_min(0)
        wall_x = torch.stack((x[:, 0], radius / LR, x[:, 2]), dim=1)
        wall_physical = self.raw_physical(wall_x)
        decay = torch.exp(-dist / self.wall_scale_m)
        velocity = physical[:, :2] - decay[:, None] * wall_physical[:, :2]
        physical = torch.cat((velocity, physical[:, 2:]), dim=1)
        return (physical - self.mean) / self.std

    def raw_physical(self, x: torch.Tensor) -> torch.Tensor:
        r_m = x[:, 1] * LR
        radius = self.wall_radius(x[:, 0] * LX)
        dist = (radius - r_m).clamp_min(0)
        local_radius = r_m / radius
        log_distance = torch.log1p(dist / self.wall_scale_m) / math.log1p(LR / self.wall_scale_m)
        features = torch.cat((x, local_radius[:, None], log_distance[:, None]), dim=1)
        z = self.trunk(features)
        raw = torch.cat((self.velocity(z), self.thermo(z), self.turbulence(z)), dim=1)
        return raw * self.std + self.mean


def transfer(old: FocusedNet, new: WallAwareNet) -> None:
    old_state = old.state_dict()
    new_state = new.state_dict()
    for key, value in old_state.items():
        if key == "trunk.0.weight":
            new_state[key][:, :3] = value
            new_state[key][:, 3:] = 0
        else:
            new_state[key] = value
    new.load_state_dict(new_state)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--wall", type=Path, default=Path("propulsion_pinn/results/data/wall_nodes.csv"))
    p.add_argument("--input-checkpoint", type=Path, default=Path("propulsion_pinn/results/checkpoints/focused_output/best.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("propulsion_pinn/results/checkpoints/wall_aware_output"))
    p.add_argument("--wall-scale-um", type=float, default=3.0)
    p.add_argument("--stage1", type=int, default=1000)
    p.add_argument("--stage2", type=int, default=1500)
    p.add_argument("--stage3", type=int, default=2500)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if min(args.stage1, args.stage2, args.stage3) < 1 or args.wall_scale_um <= 0:
        p.error("positive stage steps and wall scale required")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases, train_x, train_y, mean, std = load_data(args.data_dir)
    nodes = pd.read_csv(args.wall).sort_values("x_m")
    wall_x, wall_r = nodes.x_m.to_numpy(), nodes.r_m.to_numpy()
    if len(nodes) < 10 or not np.all(np.diff(wall_x) > 0):
        raise ValueError("Wall nodes must have unique ascending x coordinates")
    if not (abs(wall_x[0]) < 1e-8 and abs(wall_x[-1] - LX) < 1e-8):
        raise ValueError("Wall must span x=0 to x=0.170 m")
    wall = PchipInterpolator(wall_x, wall_r)
    old = FocusedNet(mean, std).to(device)
    old.load_state_dict(torch.load(args.input_checkpoint, map_location=device, weights_only=True)["state_dict"])
    model = WallAwareNet(mean, std, wall_x, wall_r, args.wall_scale_um * 1e-6).to(device)
    transfer(old, model)
    del old
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = {600: raw_four(cases[600]), 700: raw_four(cases[700])}
    report = {"model": "wall-aware hard-no-slip staged PINN", "seed": args.seed,
              "train_kpa": TRAIN_KPA, "validation_kpa": VAL_KPA,
              "wall_nodes": len(nodes), "wall_scale_um": args.wall_scale_um,
              "stage_steps": [args.stage1, args.stage2, args.stage3],
              "device": str(device), "stages": []}
    best_score, best_state, best_name = math.inf, None, None

    def snapshot(name: str, weights: dict) -> None:
        nonlocal best_score, best_state, best_name
        metrics = evaluate(model, cases[VAL_KPA], baseline, device)
        rad = metrics["radial_velocity"]["relative_l2_percent"]
        press = metrics["pressure"]["relative_l2_percent"]
        score = rad / 18.814144134521484 + press / 1.1798564195632935
        report["stages"].append({"name": name, "weights": weights,
                                 "joint_relative_score": score, "metrics": metrics})
        torch.save({"state_dict": model.state_dict(), "stage": name,
                    "metrics": metrics}, args.output_dir / f"{name}.pt")
        if score < best_score:
            best_score, best_state, best_name = score, copy.deepcopy(model.state_dict()), name
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(name, "radial_pct", round(rad, 4), "pressure_pct", round(press, 4),
              "joint_score", round(score, 4), flush=True)

    snapshot("transferred_no_training", {"data_ur": 0, "data_p": 0,
                                          "boundary": 0, "continuity": 0, "momentum": 0})
    adam = torch.optim.Adam(model.parameters(), lr=2e-4)
    for stage, steps, wr, wp in ((1, args.stage1, 4.0, 1.0),
                                  (2, args.stage2, 3.0, 2.0),
                                  (3, args.stage3, 3.0, 2.0)):
        if stage == 3:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=steps, eta_min=3e-5)
        for step in range(1, steps + 1):
            bx, by, px = batches(train_x, train_y, mean, std,
                                 wall(train_x[:, 0] * LX), rng, 1536, 128,
                                 True, device)
            bp = boundary_points(128, wall, rng, device)
            d = weighted_data_loss(model, bx, by, wr, wp)
            b = boundary_loss(model, bp)
            if stage == 1:
                c = m = torch.zeros((), device=device)
                lam_c = lam_m = 0.0
            else:
                c, m = physics_losses(model, px)
                if step % 100 == 0:
                    gd = gradient_norm(d, model)
                    gc = gradient_norm(c, model)
                    gm = gradient_norm(m, model)
                    ramp = step / steps if stage == 2 else 1.0
                    lam_c = ramp * float(np.clip(0.1 * gd / max(gc, 1e-12), 0, 0.10))
                    lam_m = ramp * float(np.clip(0.1 * gd / max(gm, 1e-12), 0, 0.05))
            loss = d + b + lam_c * c + lam_m * m
            adam.zero_grad(set_to_none=True)
            loss.backward()
            adam.step()
            if stage == 3:
                scheduler.step()
            if step == steps or step % 500 == 0:
                print(f"stage{stage}", step, "data", round(float(d.detach()), 5),
                      "bc", round(float(b.detach()), 5), "cont", round(float(c.detach()), 5),
                      "mom", round(float(m.detach()), 5), flush=True)
        snapshot(f"stage{stage}", {"data_ur": wr, "data_p": wp,
                                   "boundary": 1, "continuity": lam_c,
                                   "momentum": lam_m})
    model.load_state_dict(best_state)
    best_metrics = next(s["metrics"] for s in report["stages"] if s["name"] == best_name)
    torch.save({"state_dict": model.state_dict(), "stage": best_name,
                "metrics": best_metrics}, args.output_dir / "best.pt")
    report["best_stage"] = best_name
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("best", best_name, flush=True)


if __name__ == "__main__":
    main()
