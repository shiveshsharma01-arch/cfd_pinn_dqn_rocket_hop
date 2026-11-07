"""Data-assisted, continuity-informed surrogate for the 2-D axisymmetric nozzle.

The model learns steady fields from Fluent cell-centre exports. Its physics loss
enforces axisymmetric mass continuity and ideal-gas density. It does not solve
the complete compressible SST RANS system or represent a hot-gas rocket engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


TRAIN_PRESSURES_KPA = (600, 700, 750, 800, 850, 900, 1000)
VAL_PRESSURE_KPA = 650
R_AIR = 287.04
LENGTH_M = 0.170
RADIUS_SCALE_M = 0.050
PRESSURE_SCALE_PA = 1_000_000.0
FIELDS = (
    "cellnumber",
    "x-coordinate",
    "y-coordinate",
    "axial-velocity",
    "radial-velocity",
    "pressure",
    "density",
    "temperature",
)
TARGET_NAMES = ("axial_velocity_m_s", "radial_velocity_m_s", "log_pressure_pa", "log_temperature_k")


def read_case(path: Path, pressure_kpa: int) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        if reader.fieldnames is None or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}: {reader.fieldnames}")
        records = list(reader)
    if not records:
        raise ValueError(f"Empty CFD export: {path}")
    values = {name: np.array([float(row[name]) for row in records], dtype=np.float32) for name in FIELDS}
    if any(not np.isfinite(column).all() for column in values.values()):
        raise ValueError(f"Nonfinite value in {path}")
    if len(np.unique(values["cellnumber"])) != len(records):
        raise ValueError(f"Duplicate cell numbers in {path}")
    if np.any(values["pressure"] <= 0) or np.any(values["temperature"] <= 0):
        raise ValueError(f"Pressure and temperature must be absolute and positive: {path}")
    x = np.column_stack(
        [values["x-coordinate"] / LENGTH_M,
         values["y-coordinate"] / RADIUS_SCALE_M,
         np.full(len(records), pressure_kpa * 1000 / PRESSURE_SCALE_PA, dtype=np.float32)]
    ).astype(np.float32)
    y = np.column_stack(
        [values["axial-velocity"], values["radial-velocity"],
         np.log(values["pressure"]), np.log(values["temperature"])]
    ).astype(np.float32)
    return {"x": x, "y": y, "density": values["density"], "cellnumber": values["cellnumber"]}


def load_cases(data_dir: Path, pressures_kpa: tuple[int, ...]) -> dict[int, dict[str, np.ndarray]]:
    cases = {p: read_case(data_dir / f"{p}kpa", p) for p in pressures_kpa}
    first = cases[pressures_kpa[0]]
    for p, case in cases.items():
        if len(case["x"]) != len(first["x"]):
            raise ValueError(f"Different cell count at {p} kPa")
        if not np.array_equal(case["cellnumber"], first["cellnumber"]):
            raise ValueError(f"Cell IDs differ at {p} kPa; align by coordinates before interpolation")
        if not np.allclose(case["x"][:, :2], first["x"][:, :2], rtol=0, atol=1e-7):
            raise ValueError(f"Mesh coordinates differ at {p} kPa")
    return cases


class FieldNet(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, activation: str = "tanh"):
        super().__init__()
        if activation not in ("tanh", "silu"):
            raise ValueError(f"Unsupported activation: {activation}")
        layers: list[nn.Module] = []
        width = 64
        for n_in in (3, width, width):
            layers.extend((nn.Linear(n_in, width), nn.Tanh() if activation == "tanh" else nn.SiLU()))
        layers.append(nn.Linear(width, 4))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("target_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def physical(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raw = self.forward(x) * self.target_std + self.target_mean
        ux, ur, logp, logt = raw.unbind(dim=1)
        pressure = torch.exp(logp)
        temperature = torch.exp(logt)
        density = pressure / (R_AIR * temperature)
        return ux, ur, pressure, temperature, density


def continuity_loss(model: FieldNet, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    ux, ur, _, _, rho = model.physical(x)
    mx = rho * ux
    mr = rho * ur
    dmx = torch.autograd.grad(mx.sum(), x, create_graph=True)[0][:, 0] / LENGTH_M
    dmr = torch.autograd.grad(mr.sum(), x, create_graph=True)[0][:, 1] / RADIUS_SCALE_M
    radius = x[:, 1] * RADIUS_SCALE_M
    residual = dmx + dmr + mr / radius.clamp_min(0.00025)
    # Scale by a representative density*speed/length, so this loss is dimensionless.
    scale = 3.0 * 500.0 / LENGTH_M
    return torch.mean((residual / scale) ** 2)


def predict(model: FieldNet, x: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for part in np.array_split(x, max(1, math.ceil(len(x) / 4096))):
            result = model.physical(torch.from_numpy(part).to(device))
            chunks.append(torch.stack(result, dim=1).cpu().numpy())
    full = np.concatenate(chunks)
    return dict(zip(("ux", "ur", "pressure", "temperature", "density"), full.T))


def truth(case: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    y = case["y"]
    return {"ux": y[:, 0], "ur": y[:, 1], "pressure": np.exp(y[:, 2]),
            "temperature": np.exp(y[:, 3]), "density": case["density"]}


def errors(pred: dict[str, np.ndarray], actual: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    units = {"ux": "m/s", "ur": "m/s", "pressure": "Pa", "temperature": "K", "density": "kg/m^3"}
    return {
        name: {"rmse": float(np.sqrt(np.mean((pred[name] - actual[name]) ** 2))),
               "mae": float(np.mean(np.abs(pred[name] - actual[name]))), "unit": units[name]}
        for name in units
    }


def interpolated_validation(cases: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    low, high = truth(cases[600]), truth(cases[700])
    return {name: 0.5 * (low[name] + high[name]) for name in low}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("propulsion_pinn/results/checkpoints/output"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--physics-batch", type=int, default=256)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activation", choices=("tanh", "silu"), default="tanh")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.physics_batch < 1:
        parser.error("step and batch counts must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pressures = TRAIN_PRESSURES_KPA + (VAL_PRESSURE_KPA,)
    cases = load_cases(args.data_dir, pressures)
    train_x = np.concatenate([cases[p]["x"] for p in TRAIN_PRESSURES_KPA])
    train_y = np.concatenate([cases[p]["y"] for p in TRAIN_PRESSURES_KPA])
    mean = train_y.mean(axis=0)
    std = train_y.std(axis=0).clip(min=1e-6)
    model = FieldNet(mean, std, args.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(args.seed)
    history = []
    for step in range(1, args.steps + 1):
        idx = rng.integers(len(train_x), size=args.batch_size)
        x = torch.from_numpy(train_x[idx]).to(device)
        y = torch.from_numpy((train_y[idx] - mean) / std).to(device)
        physics_idx = rng.integers(len(train_x), size=args.physics_batch)
        physics_x = torch.from_numpy(train_x[physics_idx]).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        data_loss = torch.mean((model(x) - y) ** 2)
        mass_loss = continuity_loss(model, physics_x)
        loss = data_loss + args.physics_weight * mass_loss
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            item = {"step": step, "data_loss": float(data_loss.detach().cpu()),
                    "continuity_loss": float(mass_loss.detach().cpu())}
            history.append(item)
            print(item, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "train_pressures_kpa": TRAIN_PRESSURES_KPA, "seed": args.seed,
                "activation": args.activation},
               args.output_dir / "continuity_pinn.pt")
    observed = truth(cases[VAL_PRESSURE_KPA])
    report = {
        "model": "supervised field surrogate with ideal-gas and axisymmetric continuity constraints",
        "physics_not_included": "compressible momentum, energy and SST transport equations",
        "training_pressures_kpa": TRAIN_PRESSURES_KPA,
        "activation": args.activation,
        "validation_pressure_kpa": VAL_PRESSURE_KPA,
        "validation_pinn": errors(predict(model, cases[VAL_PRESSURE_KPA]["x"], device), observed),
        "validation_linear_interpolation": errors(interpolated_validation(cases), observed),
        "loss_history": history,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("validation_pinn", "validation_linear_interpolation")}, indent=2))


if __name__ == "__main__":
    main()
