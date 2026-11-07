"""Audit Fluent outlet exports and transcribed inlet/outlet mass-flow reports."""

from __future__ import annotations

import csv
import argparse
import json
import math
from pathlib import Path

from flight_model import DEFAULT_MAP, NozzleMap


DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "data/raw/outlets"
REPORTS = Path(__file__).resolve().parent / "results/cfd_mass_flow_reports.csv"
OUTPUT = Path(__file__).resolve().parent / "results/cfd_outlet_audit.json"


def read_profile(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = [column.strip() for column in next(reader)]
        rows = [list(map(float, row)) for row in reader]
    needed = ("density", "axial-velocity", "absolute-pressure")
    missing = [name for name in needed if name not in header]
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError(f"Malformed outlet export: {path}")
    x = [row[1] for row in rows]
    radius = [row[2] for row in rows]
    result = {
        "rows": len(rows),
        "columns": header,
        "missing_required_fields": missing,
        "face_x_min_m": min(x),
        "face_x_max_m": max(x),
        "face_radius_min_m": min(radius),
        "face_radius_max_m": max(radius),
    }
    for name in needed:
        if name in header:
            values = [row[header.index(name)] for row in rows]
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite {name}: {path}")
            result[name + "_min"] = min(values)
            result[name + "_max"] = max(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Directory containing the numbered Fluent outlet exports")
    source = parser.parse_args().source
    nozzle = NozzleMap(DEFAULT_MAP)
    with REPORTS.open(newline="", encoding="utf-8") as stream:
        reports = {
            int(row["inlet_total_pressure_kpa"]): row
            for row in csv.DictReader(stream)
        }
    cases = []
    for pressure in range(600, 1001, 50):
        profile = read_profile(source / f"{pressure}kpa")
        case = {"inlet_total_pressure_kpa": pressure, "profile": profile}
        if pressure in reports:
            report = reports[pressure]
            inlet = float(report["inlet_mass_flow_kg_s"])
            outlet_signed = float(report["outlet_mass_flow_signed_kg_s"])
            _, pinn_mdot = nozzle.evaluate(float(pressure), 101325.0)
            case["mass_report"] = {
                "inlet_kg_s": inlet,
                "outlet_signed_kg_s": outlet_signed,
                "net_kg_s": inlet + outlet_signed,
                "imbalance_percent_of_inlet": 100 * abs(inlet + outlet_signed) / inlet,
                "pinn_mass_flow_kg_s": pinn_mdot,
                "pinn_vs_cfd_outlet_percent": 100 * (pinn_mdot - abs(outlet_signed)) / abs(outlet_signed),
                "source_image": report["source_image"],
            }
        cases.append(case)
    OUTPUT.write_text(json.dumps({"source_dir": str(source), "cases": cases}, indent=2),
                      encoding="utf-8")
    print(OUTPUT)
    for case in cases:
        p = case["inlet_total_pressure_kpa"]
        q = case["profile"]
        report = case.get("mass_report", {})
        print(f"{p}: {q['rows']} faces; missing={q['missing_required_fields']}; "
              f"imbalance={report.get('imbalance_percent_of_inlet', 'no report')}%")


if __name__ == "__main__":
    main()
