"""Four-stage, four-loss test for the cold-air axisymmetric nozzle fields.

This is a data-assisted approximate RANS PINN. It enforces continuity and
momentum with a learned eddy-viscosity field; it does not enforce energy or SST
transport. Wall coordinates are reconstructed from the exported cell envelope.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from torch import nn


TRAIN_KPA = (600, 700, 750, 800, 850, 900, 1000)
VAL_KPA = 650
LX, LR = 0.170, 0.050
U_REF, RHO_REF = 500.0, 3.0
R_GAS, GAMMA = 287.04, 1.4
MU_LAM = 1.7894e-5  # Pa s; approximate constant-air viscosity for first test
COLS = ("cellnumber", "x-coordinate", "y-coordinate", "axial-velocity",
        "radial-velocity", "pressure", "temperature", "density",
        "turb-kinetic-energy", "viscosity-turb")


def read_case(data_dir: Path, kpa: int) -> dict[str, np.ndarray]:
    with (data_dir / f"{kpa}kpa").open(newline="") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        if reader.fieldnames is None or not set(COLS).issubset(reader.fieldnames):
            raise ValueError(f"Missing columns in {kpa} kPa export")
        rows = list(reader)
    v = {key: np.array([float(row[key]) for row in rows], dtype=np.float32) for key in COLS}
    if not all(np.isfinite(a).all() for a in v.values()):
        raise ValueError(f"Nonfinite values in {kpa} kPa export")
    x = np.column_stack((v["x-coordinate"] / LX, v["y-coordinate"] / LR,
                         np.full(len(rows), kpa / 1000, dtype=np.float32))).astype(np.float32)
    y = np.column_stack((v["axial-velocity"], v["radial-velocity"],
                         np.log(v["pressure"]), np.log(v["temperature"]),
                         np.log(v["turb-kinetic-energy"] + 1e-3),
                         np.log(v["viscosity-turb"] + 1e-8))).astype(np.float32)
    return {"x": x, "y": y, "id": v["cellnumber"], "rho": v["density"]}


def load_data(data_dir: Path) -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cases = {p: read_case(data_dir, p) for p in TRAIN_KPA + (VAL_KPA,)}
    reference = cases[TRAIN_KPA[0]]
    for p, case in cases.items():
        if not np.array_equal(reference["id"], case["id"]) or not np.allclose(
                reference["x"][:, :2], case["x"][:, :2], atol=1e-7, rtol=0):
            raise ValueError(f"Cell alignment differs at {p} kPa")
    train_x = np.concatenate([cases[p]["x"] for p in TRAIN_KPA])
    train_y = np.concatenate([cases[p]["y"] for p in TRAIN_KPA])
    return cases, train_x, train_y, train_y.mean(axis=0), train_y.std(axis=0).clip(min=1e-5)


def wall_profile(cell_xy: np.ndarray) -> tuple[PchipInterpolator, dict[str, float]]:
    """Approximate wall from outermost cell centres; no exact wall faces supplied."""
    x, r = cell_xy[:, 0] * LX, cell_xy[:, 1] * LR
    edges = np.linspace(0, LX, 341)
    chosen = []
    for left, right in zip(edges[:-1], edges[1:]):
        ids = np.flatnonzero((x >= left) & (x < right))
        if len(ids):
            chosen.append(ids[np.argmax(r[ids])])
    wx = np.r_[0.0, x[chosen], LX]
    wr = np.r_[0.050, r[chosen], 0.0316]
    order = np.argsort(wx)
    wx, wr = wx[order], wr[order]
    keep = np.r_[True, np.diff(wx) > 1e-9]
    wx, wr = wx[keep], wr[keep]
    fn = PchipInterpolator(wx, wr)
    grid = np.linspace(0, LX, 2001)
    return fn, {"wall_source": "upper envelope of 800 kPa cell centres",
                "wall_min_radius_m": float(fn(grid).min()),
                "wall_inlet_radius_m": float(fn(0)),
                "wall_exit_radius_m": float(fn(LX))}


class NozzleNet(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, width: int = 96):
        super().__init__()
        layers: list[nn.Module] = []
        for n_in in (3, width, width, width):
            layers += [nn.Linear(n_in, width), nn.Tanh()]
        layers.append(nn.Linear(width, 6))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def physical(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        y = self.forward(x) * self.std + self.mean
        ux, ur, logp, logt, logk, logmut = y.unbind(dim=1)
        p, t = torch.exp(logp), torch.exp(logt)
        k = (torch.exp(logk) - 1e-3).clamp_min(0)
        mut = (torch.exp(logmut) - 1e-8).clamp_min(0)
        rho = p / (R_GAS * t)
        return ux, ur, p, t, rho, k, mut


def derivative(value: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    return gradient[:, 0] / LX, gradient[:, 1] / LR


def physics_losses(model: NozzleNet, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.detach().requires_grad_(True)
    ux, ur, p, _, rho, k, mut = model.physical(x)
    radius = (x[:, 1] * LR).clamp_min(0.001)
    ux_x, ux_r = derivative(ux, x)
    ur_x, ur_r = derivative(ur, x)
    p_x, p_r = derivative(p, x)
    mass_x, _ = derivative(rho * ux, x)
    _, mass_r = derivative(rho * ur, x)
    continuity = mass_x + mass_r + rho * ur / radius
    l_cont = torch.mean((continuity / (RHO_REF * U_REF / LX)) ** 2)

    div_u = ux_x + ur_r + ur / radius
    mu = MU_LAM + mut
    turbulent_pressure = (2.0 / 3.0) * rho * k
    tau_xx = 2 * mu * ux_x - (2.0 / 3.0) * mu * div_u - turbulent_pressure
    tau_rr = 2 * mu * ur_r - (2.0 / 3.0) * mu * div_u - turbulent_pressure
    tau_tt = 2 * mu * ur / radius - (2.0 / 3.0) * mu * div_u - turbulent_pressure
    tau_xr = mu * (ux_r + ur_x)
    d_txx_dx, _ = derivative(tau_xx, x)
    _, d_txr_dr = derivative(tau_xr, x)
    d_txr_dx, _ = derivative(tau_xr, x)
    _, d_trr_dr = derivative(tau_rr, x)
    rx = rho * (ux * ux_x + ur * ux_r) + p_x - (d_txx_dx + d_txr_dr + tau_xr / radius)
    rr = rho * (ux * ur_x + ur * ur_r) + p_r - (d_txr_dx + d_trr_dr + (tau_rr - tau_tt) / radius)
    l_mom = 0.5 * torch.mean((rx / (RHO_REF * U_REF**2 / LX)) ** 2
                              + (rr / (RHO_REF * U_REF**2 / LR)) ** 2)
    return l_cont, l_mom


def boundary_points(n: int, wall: PchipInterpolator, rng: np.random.Generator,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = rng.choice(TRAIN_KPA, size=n).astype(np.float32) / 1000
    wall_x = rng.uniform(0.001, LX - 0.001, size=n)
    wall_r = wall(wall_x)
    wall_pts = np.column_stack((wall_x / LX, wall_r / LR, p)).astype(np.float32)
    axis_pts = np.column_stack((rng.uniform(0.001, LX - 0.001, size=n) / LX,
                                np.zeros(n), p)).astype(np.float32)
    inlet_pts = np.column_stack((np.zeros(n), rng.uniform(0.001, 0.049, size=n) / LR,
                                 p)).astype(np.float32)
    return tuple(torch.from_numpy(a).to(device) for a in (wall_pts, axis_pts, inlet_pts))


def boundary_loss(model: NozzleNet, points: tuple[torch.Tensor, ...]) -> torch.Tensor:
    wall, axis, inlet = points
    wux, wur, *_ = model.physical(wall)
    _, aur, *_ = model.physical(axis)
    iux, iur, ip, it, *_ = model.physical(inlet)
    speed2 = iux.square() + iur.square()
    mach2 = speed2 / (GAMMA * R_GAS * it)
    total_t = it * (1 + 0.5 * (GAMMA - 1) * mach2)
    total_p = ip * (1 + 0.5 * (GAMMA - 1) * mach2) ** (GAMMA / (GAMMA - 1))
    wall_loss = torch.mean((wux / U_REF) ** 2 + (wur / U_REF) ** 2)
    axis_loss = torch.mean((aur / U_REF) ** 2)
    inlet_loss = torch.mean(((total_t - 300) / 300) ** 2 +
                            ((total_p - inlet[:, 2] * 1e6) / 1e6) ** 2)
    return (wall_loss + axis_loss + inlet_loss) / 3


def data_loss(model: NozzleNet, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    error = (model(x) - y).square()
    # Main flow variables carry full weight; k and eddy viscosity support momentum.
    return error[:, :4].mean() + 0.25 * error[:, 4:].mean()


def sample_batch(train_x: np.ndarray, train_y: np.ndarray, mean: np.ndarray,
                 std: np.ndarray, rng: np.random.Generator, n: int,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = rng.integers(len(train_x), size=n)
    return (torch.from_numpy(train_x[ids]).to(device),
            torch.from_numpy((train_y[ids] - mean) / std).to(device))


def sample_physics(train_x: np.ndarray, rng: np.random.Generator, n: int,
                   device: torch.device) -> torch.Tensor:
    # Avoid the cylindrical 1/r singularity on the axis; axis is handled by BC.
    eligible = np.flatnonzero(train_x[:, 1] * LR > 0.001)
    return torch.from_numpy(train_x[rng.choice(eligible, size=n)]).to(device)


def gradient_norm(loss: torch.Tensor, model: NozzleNet) -> float:
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))


def evaluate(model: NozzleNet, case: dict[str, np.ndarray], baseline: dict | None,
             device: torch.device) -> dict[str, dict[str, float]]:
    model.eval()
    preds = []
    with torch.no_grad():
        for part in np.array_split(case["x"], 16):
            fields = model.physical(torch.from_numpy(part).to(device))
            preds.append(torch.stack(fields[:4], dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    true = np.column_stack((case["y"][:, 0], case["y"][:, 1],
                            np.exp(case["y"][:, 2]), np.exp(case["y"][:, 3])))
    out = {}
    for i, name in enumerate(("axial_velocity", "radial_velocity", "pressure", "temperature")):
        diff = pred[:, i] - true[:, i]
        out[name] = {"rmse": float(np.sqrt(np.mean(diff**2))),
                     "relative_l2_percent": float(100 * np.linalg.norm(diff) / np.linalg.norm(true[:, i]))}
        if name == "pressure":
            out[name]["mean_absolute_percent_error"] = float(100 * np.mean(np.abs(diff / true[:, i])))
    if baseline is not None:
        low, high = baseline[600], baseline[700]
        interp = 0.5 * (low + high)
        out["interpolation"] = {}
        for i, name in enumerate(("axial_velocity", "radial_velocity", "pressure", "temperature")):
            diff = interp[:, i] - true[:, i]
            out["interpolation"][name] = {"rmse": float(np.sqrt(np.mean(diff**2))),
                                          "relative_l2_percent": float(100 * np.linalg.norm(diff) / np.linalg.norm(true[:, i]))}
    return out


def raw_four(case: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack((case["y"][:, 0], case["y"][:, 1],
                            np.exp(case["y"][:, 2]), np.exp(case["y"][:, 3])))


def checkpoint(path: Path, model: NozzleNet, stage: str, losses: dict[str, float],
               metrics: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "stage": stage, "weights": losses,
                "metrics": metrics, "train_kpa": TRAIN_KPA, "validation_kpa": VAL_KPA}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("propulsion_pinn/results/checkpoints/four_loss_output"))
    parser.add_argument("--stage1", type=int, default=1000)
    parser.add_argument("--stage2", type=int, default=2000)
    parser.add_argument("--stage3", type=int, default=3000)
    parser.add_argument("--lbfgs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--physics-batch", type=int, default=64)
    parser.add_argument("--boundary-batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.stage1, args.stage2, args.stage3, args.lbfgs,
           args.batch, args.physics_batch, args.boundary_batch) < 1:
        parser.error("all stage and batch sizes must be positive")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases, train_x, train_y, mean, std = load_data(args.data_dir)
    wall, wall_info = wall_profile(cases[800]["x"][:, :2])
    model = NozzleNet(mean, std).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": "data-assisted approximate axisymmetric RANS PINN",
              "caveat": "wall reconstructed from cell centres; energy and SST transport absent",
              "train_kpa": TRAIN_KPA, "validation_kpa": VAL_KPA,
              "wall": wall_info, "device": str(device), "torch": torch.__version__,
              "configuration": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
              "stages": []}
    baseline = {600: raw_four(cases[600]), 700: raw_four(cases[700])}
    best_score, best_state, best_stage = math.inf, None, None

    def end_stage(name: str, weights: dict[str, float]) -> None:
        nonlocal best_score, best_state, best_stage
        metrics = evaluate(model, cases[VAL_KPA], baseline, device)
        score = metrics["radial_velocity"]["relative_l2_percent"] + metrics["pressure"]["relative_l2_percent"]
        report["stages"].append({"name": name, "weights": weights,
                                  "radial_plus_pressure_score": score, "metrics": metrics})
        checkpoint(args.output_dir / f"{name}.pt", model, name, weights, metrics)
        if score < best_score:
            best_score, best_state, best_stage = score, copy.deepcopy(model.state_dict()), name
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(name, "radial_rel_l2_pct", round(metrics["radial_velocity"]["relative_l2_percent"], 3),
              "pressure_rel_l2_pct", round(metrics["pressure"]["relative_l2_percent"], 3), flush=True)

    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    for stage_num, count in ((1, args.stage1), (2, args.stage2), (3, args.stage3)):
        if stage_num == 3:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=count, eta_min=1e-4)
            lam_c, lam_m = 0.1, 0.05
        for step in range(1, count + 1):
            model.train()
            bx, by = sample_batch(train_x, train_y, mean, std, rng, args.batch, device)
            bp = boundary_points(args.boundary_batch, wall, rng, device)
            dloss = data_loss(model, bx, by)
            bloss = boundary_loss(model, bp)
            if stage_num == 1:
                lam_c = lam_m = 0.0
                closs = mloss = torch.zeros((), device=device)
            else:
                px = sample_physics(train_x, rng, args.physics_batch, device)
                closs, mloss = physics_losses(model, px)
                if stage_num == 2:
                    halfway = max(1, count // 2)
                    lam_c = 0.1 * min(1, step / halfway)
                    lam_m = 0.05 * max(0, step - halfway) / max(1, count - halfway)
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
            if stage_num == 3:
                scheduler.step()
            if step == count or step % 500 == 0:
                print(f"stage{stage_num}", step, "data", round(float(dloss.detach()), 5),
                      "bc", round(float(bloss.detach()), 5),
                      "continuity", round(float(closs.detach()), 5),
                      "momentum", round(float(mloss.detach()), 5),
                      "weights", round(lam_c, 4), round(lam_m, 4), flush=True)
        end_stage(f"stage{stage_num}", {"data": 1.0, "boundary": 1.0,
                                     "continuity": lam_c, "momentum": lam_m})

    # Deterministic objective for L-BFGS, starting from the best Adam checkpoint.
    model.load_state_dict(best_state)
    bx, by = sample_batch(train_x, train_y, mean, std, rng, 2048, device)
    px = sample_physics(train_x, rng, 128, device)
    bp = boundary_points(128, wall, rng, device)
    chosen_weights = next(s["weights"] for s in report["stages"] if s["name"] == best_stage)
    lc, lm = chosen_weights["continuity"], chosen_weights["momentum"]
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=args.lbfgs,
                                  history_size=20, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        dloss = data_loss(model, bx, by)
        bloss = boundary_loss(model, bp)
        closs, mloss = physics_losses(model, px)
        total = dloss + bloss + lc * closs + lm * mloss
        total.backward()
        return total

    optimizer.step(closure)
    end_stage("stage4_lbfgs", chosen_weights)
    model.load_state_dict(best_state)
    checkpoint(args.output_dir / "best.pt", model, best_stage,
               next(s["weights"] for s in report["stages"] if s["name"] == best_stage),
               next(s["metrics"] for s in report["stages"] if s["name"] == best_stage))
    report["best_stage"] = best_stage
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("best", best_stage, "score", round(best_score, 3), flush=True)


if __name__ == "__main__":
    main()
