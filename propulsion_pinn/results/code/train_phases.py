"""Compare activations, then refine a nozzle PINN using staged optimization."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from nozzle_pinn import (
    TRAIN_PRESSURES_KPA, VAL_PRESSURE_KPA, FieldNet, continuity_loss,
    errors, interpolated_validation, load_cases, predict, truth,
)


def validation(model: FieldNet, case: dict, mean: np.ndarray, std: np.ndarray,
               device: torch.device) -> tuple[float, dict]:
    model.eval()
    target = torch.from_numpy((case["y"] - mean) / std).to(device)
    with torch.no_grad():
        pieces = [model(torch.from_numpy(part).to(device)) for part in np.array_split(case["x"], 8)]
        pred = torch.cat(pieces)
        score = float(torch.mean((pred - target) ** 2).cpu())
    return score, errors(predict(model, case["x"], device), truth(case))


def save(path: Path, model: FieldNet, mean: np.ndarray, std: np.ndarray,
         activation: str, stage: str, score: float) -> None:
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "activation": activation, "stage": stage, "validation_score": score,
                "train_pressures_kpa": TRAIN_PRESSURES_KPA,
                "validation_pressure_kpa": VAL_PRESSURE_KPA}, path)


def adam_phase(model: FieldNet, train_x: np.ndarray, train_y: np.ndarray,
               mean: np.ndarray, std: np.ndarray, rng: np.random.Generator,
               steps: int, physics_weight: float, batch_size: int,
               physics_batch: int, device: torch.device, label: str) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for step in range(1, steps + 1):
        ids = rng.integers(len(train_x), size=batch_size)
        x = torch.from_numpy(train_x[ids]).to(device)
        y = torch.from_numpy((train_y[ids] - mean) / std).to(device)
        optimizer.zero_grad(set_to_none=True)
        data_loss = torch.mean((model(x) - y) ** 2)
        if physics_weight:
            physics_ids = rng.integers(len(train_x), size=physics_batch)
            physics_x = torch.from_numpy(train_x[physics_ids]).to(device)
            mass_loss = continuity_loss(model, physics_x)
            loss = data_loss + physics_weight * mass_loss
        else:
            mass_loss = torch.zeros((), device=device)
            loss = data_loss
        loss.backward()
        optimizer.step()
        if step == steps or step % 500 == 0:
            print(label, step, "data", round(float(data_loss.detach()), 6),
                  "continuity", round(float(mass_loss.detach()), 6), flush=True)


def lbfgs_phase(model: FieldNet, train_x: np.ndarray, train_y: np.ndarray,
                mean: np.ndarray, std: np.ndarray, rng: np.random.Generator,
                steps: int, physics_weight: float, device: torch.device) -> None:
    # A fixed batch is required because LBFGS reevaluates the closure repeatedly.
    ids = rng.choice(len(train_x), size=min(2048, len(train_x)), replace=False)
    pids = rng.choice(len(train_x), size=min(256, len(train_x)), replace=False)
    x = torch.from_numpy(train_x[ids]).to(device)
    y = torch.from_numpy((train_y[ids] - mean) / std).to(device)
    px = torch.from_numpy(train_x[pids]).to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=5,
                                  history_size=10, line_search_fn="strong_wolfe")
    model.train()
    for step in range(1, steps + 1):
        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x) - y) ** 2) + physics_weight * continuity_loss(model, px)
            loss.backward()
            return loss
        result = optimizer.step(closure)
        if step == steps or step % 10 == 0:
            print("lbfgs", step, "fixed_batch_loss", round(float(result.detach()), 6), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("propulsion_pinn/results/checkpoints/phased_output"))
    parser.add_argument("--screen-steps", type=int, default=1500)
    parser.add_argument("--joint-steps", type=int, default=2000)
    parser.add_argument("--lbfgs-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--physics-batch", type=int, default=256)
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.screen_steps, args.joint_steps, args.lbfgs_steps,
           args.batch_size, args.physics_batch) < 1 or args.physics_weight < 0:
        parser.error("step and batch counts must be positive; physics weight cannot be negative")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = load_cases(args.data_dir, TRAIN_PRESSURES_KPA + (VAL_PRESSURE_KPA,))
    train_x = np.concatenate([cases[p]["x"] for p in TRAIN_PRESSURES_KPA])
    train_y = np.concatenate([cases[p]["y"] for p in TRAIN_PRESSURES_KPA])
    mean, std = train_y.mean(axis=0), train_y.std(axis=0).clip(min=1e-6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"train_pressures_kpa": TRAIN_PRESSURES_KPA,
              "validation_pressure_kpa": VAL_PRESSURE_KPA,
              "interpolation_errors": errors(interpolated_validation(cases), truth(cases[VAL_PRESSURE_KPA])),
              "stages": [], "torch_version": torch.__version__}
    candidates = {}
    for activation in ("tanh", "silu"):
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = FieldNet(mean, std, activation).to(device)
        adam_phase(model, train_x, train_y, mean, std, rng, args.screen_steps, 0.0,
                   args.batch_size, args.physics_batch, device, f"screen-{activation}")
        score, field_errors = validation(model, cases[VAL_PRESSURE_KPA], mean, std, device)
        stage = f"screen_{activation}"
        save(args.output_dir / f"{stage}.pt", model, mean, std, activation, stage, score)
        report["stages"].append({"stage": stage, "validation_normalized_mse": score,
                                  "field_errors": field_errors})
        print(stage, "validation_normalized_mse", score, flush=True)
        candidates[activation] = (score, copy.deepcopy(model.state_dict()))
    chosen = min(candidates, key=lambda name: candidates[name][0])
    model = FieldNet(mean, std, chosen).to(device)
    model.load_state_dict(candidates[chosen][1])
    best_stage, best_score, best_state = f"screen_{chosen}", candidates[chosen][0], copy.deepcopy(model.state_dict())
    rng = np.random.default_rng(args.seed + 1)
    adam_phase(model, train_x, train_y, mean, std, rng, args.joint_steps,
               args.physics_weight, args.batch_size, args.physics_batch, device, "joint")
    score, field_errors = validation(model, cases[VAL_PRESSURE_KPA], mean, std, device)
    save(args.output_dir / "joint.pt", model, mean, std, chosen, "joint", score)
    report["stages"].append({"stage": "joint", "activation": chosen,
                              "validation_normalized_mse": score, "field_errors": field_errors})
    if score < best_score:
        best_stage, best_score, best_state = "joint", score, copy.deepcopy(model.state_dict())
    lbfgs_phase(model, train_x, train_y, mean, std, rng, args.lbfgs_steps,
                args.physics_weight, device)
    score, field_errors = validation(model, cases[VAL_PRESSURE_KPA], mean, std, device)
    save(args.output_dir / "lbfgs.pt", model, mean, std, chosen, "lbfgs", score)
    report["stages"].append({"stage": "lbfgs", "activation": chosen,
                              "validation_normalized_mse": score, "field_errors": field_errors})
    if score < best_score:
        best_stage, best_score, best_state = "lbfgs", score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    save(args.output_dir / "best.pt", model, mean, std, chosen, best_stage, best_score)
    report["best_stage"] = best_stage
    report["best_validation_normalized_mse"] = best_score
    report["configuration"] = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("best_stage", best_stage, "score", best_score, flush=True)


if __name__ == "__main__":
    main()
