"""Evaluate the frozen PINN on independent 950 kPa CFD cell data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from predict_wall_aware import load_model
from train_four_loss import evaluate, raw_four, read_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("propulsion_pinn/results/checkpoints/final/independent_950.json"))
    parser.add_argument("--wall", type=Path,
                        default=Path("propulsion_pinn/results/data/wall_nodes.csv"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("propulsion_pinn/results/checkpoints/final/best.pt"))
    parser.add_argument("--report", type=Path,
                        default=Path("propulsion_pinn/results/checkpoints/final/report.json"))
    args = parser.parse_args()
    case = read_case(args.data_dir, 950)
    low = read_case(args.data_dir, 900)
    high = read_case(args.data_dir, 1000)
    for reference in (low, high):
        if not np.array_equal(case["id"], reference["id"]) or not np.allclose(
                case["x"][:, :2], reference["x"][:, :2], atol=1e-7, rtol=0):
            raise ValueError("950 kPa CFD cells do not align with neighbouring cases")
    model, _, device = load_model(args.wall, args.checkpoint, args.report)
    pinn_metrics = evaluate(model, case, None, device)
    true = raw_four(case)
    interpolation = 0.5 * (raw_four(low) + raw_four(high))
    names = ("axial_velocity", "radial_velocity", "pressure", "temperature")
    interpolation_metrics = {}
    for index, name in enumerate(names):
        delta = interpolation[:, index] - true[:, index]
        interpolation_metrics[name] = {
            "relative_l2_percent": float(100 * np.linalg.norm(delta) / np.linalg.norm(true[:, index])),
        }
    result = {
        "case": "950 kPa CFD field; not used in PINN training or checkpoint selection",
        "cells": len(case["id"]),
        "pinn": pinn_metrics,
        "linear_interpolation_900_1000": interpolation_metrics,
        "caveat": "Field errors are against Fluent cell data, not experimental truth or integrated thrust",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
