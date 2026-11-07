"""Five-stage PINN experiment with radial-then-pressure data emphasis."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_four_loss import (
    LX, LR, R_GAS, TRAIN_KPA, VAL_KPA, boundary_loss, boundary_points,
    evaluate, gradient_norm, load_data, physics_losses, raw_four, wall_profile,
)


REFERENCE_RADIAL_PERCENT = 19.295625686645508
REFERENCE_PRESSURE_PERCENT = 1.3322477340698242


class FocusedNet(nn.Module):
    """Shared spatial trunk with separate velocity, thermodynamic, and turbulence heads."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, width: int = 128):
        super().__init__()
        trunk: list[nn.Module] = []
        for n_in in (3, width, width, width):
            trunk += [nn.Linear(n_in, width), nn.Tanh()]
        self.trunk = nn.Sequential(*trunk)
        self.velocity = self._head(width)
        self.thermo = self._head(width)
        self.turbulence = self._head(width)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    @staticmethod
    def _head(width: int) -> nn.Sequential:
        head = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(),
                             nn.Linear(width // 2, 2))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        return torch.cat((self.velocity(z), self.thermo(z), self.turbulence(z)), dim=1)

    def physical(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        y = self.forward(x) * self.std + self.mean
        ux, ur, logp, logt, logk, logmut = y.unbind(dim=1)
        p, t = torch.exp(logp), torch.exp(logt)
        k = (torch.exp(logk) - 1e-3).clamp_min(0)
        mut = (torch.exp(logmut) - 1e-8).clamp_min(0)
        rho = p / (R_GAS * t)
        return ux, ur, p, t, rho, k, mut


def weighted_data_loss(model: FocusedNet, x: torch.Tensor, y: torch.Tensor,
                       radial_weight: float, pressure_weight: float) -> torch.Tensor:
    weights = torch.tensor((1.0, radial_weight, pressure_weight, 1.0, 0.25, 0.25),
                           device=x.device)
    per_output = torch.mean((model(x) - y).square(), dim=0)
    return torch.dot(per_output, weights) / weights.sum()


def sample_indices(train_x: np.ndarray, wall_r: np.ndarray,
                   rng: np.random.Generator, n: int, focused: bool) -> np.ndarray:
    if not focused:
        return rng.integers(len(train_x), size=n)
    x_m, r_m = train_x[:, 0] * LX, train_x[:, 1] * LR
    throat = np.flatnonzero((x_m >= 0.055) & (x_m < 0.080))
    near_wall = np.flatnonzero(r_m / wall_r > 0.80)
    n_uniform = n // 2
    n_throat = n // 4
    return np.concatenate((rng.integers(len(train_x), size=n_uniform),
                           rng.choice(throat, size=n_throat),
                           rng.choice(near_wall, size=n - n_uniform - n_throat)))


def batches(train_x: np.ndarray, train_y: np.ndarray, mean: np.ndarray,
            std: np.ndarray, wall_r: np.ndarray, rng: np.random.Generator,
            n_data: int, n_phys: int, focused: bool, device: torch.device
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = sample_indices(train_x, wall_r, rng, n_data, focused)
    pids = sample_indices(train_x, wall_r, rng, n_phys * 2, focused)
    pids = pids[train_x[pids, 1] * LR > 0.001][:n_phys]
    if len(pids) < n_phys:
        eligible = np.flatnonzero(train_x[:, 1] * LR > 0.001)
        pids = np.r_[pids, rng.choice(eligible, size=n_phys - len(pids))]
    return (torch.from_numpy(train_x[ids]).to(device),
            torch.from_numpy((train_y[ids] - mean) / std).to(device),
            torch.from_numpy(train_x[pids]).to(device))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("propulsion_pinn/results/checkpoints/focused_output"))
    parser.add_argument("--stage1", type=int, default=1000)
    parser.add_argument("--stage2", type=int, default=2000)
    parser.add_argument("--stage3", type=int, default=2000)
    parser.add_argument("--stage4", type=int, default=4000)
    parser.add_argument("--lbfgs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.stage1, args.stage2, args.stage3, args.stage4, args.lbfgs) < 1:
        parser.error("all stages need at least one optimizer step")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases, train_x, train_y, mean, std = load_data(args.data_dir)
    wall, wall_info = wall_profile(cases[800]["x"][:, :2])
    wall_r = wall(train_x[:, 0] * LX)
    model = FocusedNet(mean, std).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = {600: raw_four(cases[600]), 700: raw_four(cases[700])}
    report = {"model": "shared Tanh trunk with separate output heads",
              "train_kpa": TRAIN_KPA, "validation_kpa": VAL_KPA,
              "wall": wall_info, "seed": args.seed, "device": str(device),
              "stage_steps": {"stage1": args.stage1, "stage2": args.stage2,
                              "stage3": args.stage3, "stage4": args.stage4,
                              "lbfgs_max_iter": args.lbfgs}, "stages": []}
    best_score, best_state, best_name = math.inf, None, None

    def save_stage(name: str, weights: dict[str, float]) -> None:
        nonlocal best_score, best_state, best_name
        metrics = evaluate(model, cases[VAL_KPA], baseline, device)
        radial = metrics["radial_velocity"]["relative_l2_percent"]
        pressure = metrics["pressure"]["relative_l2_percent"]
        # Equal *relative* emphasis compared with the prior four-loss model.
        score = 0.5 * (radial / REFERENCE_RADIAL_PERCENT +
                       pressure / REFERENCE_PRESSURE_PERCENT)
        entry = {"name": name, "weights": weights,
                 "joint_relative_score": score, "metrics": metrics}
        report["stages"].append(entry)
        torch.save({"state_dict": model.state_dict(), "stage": name,
                    "weights": weights, "metrics": metrics},
                   args.output_dir / f"{name}.pt")
        if score < best_score:
            best_score, best_state, best_name = score, copy.deepcopy(model.state_dict()), name
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(name, "radial_pct", round(radial, 4),
              "pressure_pct", round(pressure, 4), "joint_score", round(score, 4), flush=True)

    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = ((1, args.stage1, 1.0, 1.0),
                (2, args.stage2, 4.0, 1.0),
                (3, args.stage3, 2.0, 4.0),
                (4, args.stage4, 2.0, 2.0))
    for stage, count, wr, wp in schedule:
        if stage == 4:
            lam_c, lam_m = 0.10, 0.05
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=count, eta_min=1e-4)
        for step in range(1, count + 1):
            model.train()
            bx, by, px = batches(train_x, train_y, mean, std, wall_r, rng,
                                 1024, 64, stage > 1, device)
            bp = boundary_points(128, wall, rng, device)
            dloss = weighted_data_loss(model, bx, by, wr, wp)
            bloss = boundary_loss(model, bp)
            if stage == 1:
                lam_c = lam_m = 0.0
                closs = mloss = torch.zeros((), device=device)
            else:
                closs, mloss = physics_losses(model, px)
                if stage == 2:
                    lam_c, lam_m = 0.10 * step / count, 0.0
                elif stage == 3:
                    lam_c, lam_m = 0.10, 0.05 * step / count
                elif step % 100 == 0:
                    gd = gradient_norm(dloss, model)
                    gc = gradient_norm(closs, model)
                    gm = gradient_norm(mloss, model)
                    lam_c = 0.9 * lam_c + 0.1 * float(np.clip(gd / max(gc, 1e-12), 0.03, 0.30))
                    lam_m = 0.9 * lam_m + 0.1 * float(np.clip(gd / max(gm, 1e-12), 0.01, 0.15))
            total = dloss + bloss + lam_c * closs + lam_m * mloss
            adam.zero_grad(set_to_none=True)
            total.backward()
            adam.step()
            if stage == 4:
                scheduler.step()
            if step == count or step % 500 == 0:
                print(f"stage{stage}", step, "data", round(float(dloss.detach()), 5),
                      "bc", round(float(bloss.detach()), 5),
                      "cont", round(float(closs.detach()), 5),
                      "mom", round(float(mloss.detach()), 5),
                      "lambdas", round(lam_c, 4), round(lam_m, 4), flush=True)
        save_stage(f"stage{stage}", {"data_ux": 1.0, "data_ur": wr,
                                 "data_p": wp, "data_t": 1.0,
                                 "data_k": 0.25, "data_mut": 0.25,
                                 "boundary": 1.0, "continuity": lam_c,
                                 "momentum": lam_m})

    # L-BFGS starts from the best Adam stage and uses a fixed objective/batch.
    model.load_state_dict(best_state)
    chosen = next(s["weights"] for s in report["stages"] if s["name"] == best_name)
    bx, by, px = batches(train_x, train_y, mean, std, wall_r, rng, 2048, 128, True, device)
    bp = boundary_points(128, wall, rng, device)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=args.lbfgs,
                              history_size=20, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        d = weighted_data_loss(model, bx, by, chosen["data_ur"], chosen["data_p"])
        b = boundary_loss(model, bp)
        c, m = physics_losses(model, px)
        loss = d + b + chosen["continuity"] * c + chosen["momentum"] * m
        loss.backward()
        return loss

    lbfgs.step(closure)
    save_stage("stage5_lbfgs", chosen)
    model.load_state_dict(best_state)
    best_entry = next(s for s in report["stages"] if s["name"] == best_name)
    torch.save({"state_dict": model.state_dict(), "stage": best_name,
                "weights": best_entry["weights"], "metrics": best_entry["metrics"]},
               args.output_dir / "best.pt")
    report["best_stage"] = best_name
    report["best_joint_relative_score"] = best_score
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("best", best_name, "joint_score", round(best_score, 4), flush=True)


if __name__ == "__main__":
    main()
