"""Run an illustrative flight with the frozen, as-yet-unvalidated PINN map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from flight_model import DEFAULT_MAP, NozzleMap, Vehicle, simulate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--pressure-kpa", type=float, default=800.0)
    parser.add_argument("--wet-mass-kg", type=float, default=100.0)
    parser.add_argument("--dry-mass-kg", type=float, default=50.0)
    parser.add_argument("--drag-coefficient", type=float, default=0.5)
    parser.add_argument("--frontal-area-m2", type=float, default=0.01)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results/example_flight.csv")
    args = parser.parse_args()
    nozzle = NozzleMap(args.map)
    vehicle = Vehicle(args.wet_mass_kg, args.dry_mass_kg,
                      args.drag_coefficient, args.frontal_area_m2)
    history = simulate(nozzle, vehicle, lambda _: args.pressure_kpa,
                       args.duration_s, args.dt_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "status": "provisional PINN-based flight illustration; CFD thrust not yet checked",
        "nozzle_map": str(nozzle.path),
        "nozzle_map_sha256": nozzle.sha256,
        "command_kpa": args.pressure_kpa,
        "wet_mass_kg": vehicle.initial_mass_kg,
        "dry_mass_kg": vehicle.dry_mass_kg,
        "duration_s": args.duration_s,
        "dt_s": args.dt_s,
        "trajectory_csv": str(args.output.resolve()),
        "max_altitude_m": max(row["altitude_m"] for row in history),
        "max_dynamic_pressure_pa": max(row["dynamic_pressure_pa"] for row in history),
        "final_time_s": history[-1]["time_s"],
        "final_altitude_m": history[-1]["altitude_m"],
        "final_downrange_m": history[-1]["downrange_m"],
        "final_mass_kg": history[-1]["mass_kg"],
        "fuel_exhausted": history[-1]["mass_kg"] <= vehicle.dry_mass_kg + 1e-9,
        "terminated_at_ground": history[-1]["time_s"] < args.duration_s - 1e-9,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
