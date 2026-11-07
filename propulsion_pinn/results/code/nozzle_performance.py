"""Integrate frozen PINN exit fields into mass flow and cold-air nozzle thrust."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from predict_wall_aware import load_model, predict_fields
from train_four_loss import LX, LR


def integrate_exit_profile(radius_m: np.ndarray, radial_weights_m: np.ndarray,
                           density_kg_m3: np.ndarray, axial_velocity_m_s: np.ndarray,
                           static_pressure_pa: np.ndarray, ambient_pressure_pa: float
                           ) -> dict[str, float]:
    """Integrate an axisymmetric exit profile; weights already have radial units."""
    if ambient_pressure_pa < 0 or not math.isfinite(ambient_pressure_pa):
        raise ValueError("ambient pressure must be a finite nonnegative absolute pressure")
    area_weights = 2 * math.pi * radius_m * radial_weights_m
    if not all(np.isfinite(a).all() for a in
               (area_weights, density_kg_m3, axial_velocity_m_s, static_pressure_pa)):
        raise ValueError("nonfinite exit profile")
    area = float(np.sum(area_weights))
    mdot = float(np.sum(density_kg_m3 * axial_velocity_m_s * area_weights))
    momentum_force = float(np.sum(density_kg_m3 * axial_velocity_m_s**2 * area_weights))
    pressure_integral = float(np.sum(static_pressure_pa * area_weights))
    pressure_force = pressure_integral - ambient_pressure_pa * area
    if area <= 0 or mdot <= 0:
        raise ValueError("exit area and outward mass flow must be positive")
    return {
        "exit_area_m2": area,
        "mass_flow_kg_s": mdot,
        "exit_velocity_mass_weighted_m_s": momentum_force / mdot,
        "exit_pressure_area_weighted_pa": pressure_integral / area,
        "momentum_force_n": momentum_force,
        "pressure_force_n": pressure_force,
        "thrust_n": momentum_force + pressure_force,
    }


def exit_quadrature(exit_radius_m: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 8:
        raise ValueError("use at least 8 radial quadrature points")
    x, weight = np.polynomial.legendre.leggauss(n)
    return 0.5 * exit_radius_m * (x + 1), 0.5 * exit_radius_m * weight


def performance(model, device, exit_x_m: float, exit_radius_m: float,
                inlet_kpa: float, ambient_pressure_pa: float,
                n: int = 256) -> dict[str, float]:
    """Evaluate the PINN on a radial exit quadrature and integrate its outputs."""
    if not 600 <= inlet_kpa <= 1000:
        raise ValueError("inlet pressure outside the trained 600–1000 kPa range")
    radius, weights = exit_quadrature(exit_radius_m, n)
    inputs = np.column_stack((np.full(n, exit_x_m / LX), radius / LR,
                              np.full(n, inlet_kpa / 1000))).astype(np.float32)
    fields = predict_fields(model, inputs, device)
    result = integrate_exit_profile(radius, weights, fields[:, 4], fields[:, 0],
                                    fields[:, 2], ambient_pressure_pa)
    result.update({"inlet_total_pressure_kpa": inlet_kpa,
                   "ambient_absolute_pressure_pa": ambient_pressure_pa,
                   "radial_quadrature_points": n})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inlet-kpa", type=float, nargs="+",
                        default=[600, 650, 700, 750, 800, 850, 900, 950, 1000])
    parser.add_argument("--ambient-pa", type=float, required=True,
                        help="absolute ambient pressure in pascals")
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--output", type=Path,
                        default=Path("propulsion_pinn/results/checkpoints/wall_aware_output/thrust_map.csv"))
    parser.add_argument("--wall", type=Path, default=Path("propulsion_pinn/results/data/wall_nodes.csv"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("propulsion_pinn/results/checkpoints/final/best.pt"))
    parser.add_argument("--report", type=Path,
                        default=Path("propulsion_pinn/results/checkpoints/final/report.json"))
    args = parser.parse_args()
    model, nodes, device = load_model(args.wall, args.checkpoint, args.report)
    exit_x_m, exit_radius_m = float(nodes.x_m.iloc[-1]), float(nodes.r_m.iloc[-1])
    rows = [performance(model, device, exit_x_m, exit_radius_m, p,
                        args.ambient_pa, args.points) for p in args.inlet_kpa]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "exit_radius_m": exit_radius_m,
                      "results": rows}, indent=2))


if __name__ == "__main__":
    main()
