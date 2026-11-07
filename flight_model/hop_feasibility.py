"""Explore vertical hop and braking feasibility before defining a landing task.

This is an idealized, hand-coded controller sweep, not DQN training or a real
landing guarantee. It uses the existing provisional nozzle map and flight model.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from flight_model import (
    DEFAULT_MAP, EARTH_MU_M3_S2, EARTH_RADIUS_M, NozzleMap,
    State, Vehicle, atmosphere, step,
)


RESULTS = Path(__file__).resolve().parent / "results"


def run_hop(ascent_burn_s: float, margin_m: float, landing_pressure_kpa: float,
            dt_s: float = 0.05, integration_step_s: float | None = None
            ) -> tuple[dict, list[dict]]:
    nozzle = NozzleMap(DEFAULT_MAP)
    vehicle = Vehicle()
    state = State(0.0, EARTH_RADIUS_M, 0.0, 0.0, 0.0, vehicle.initial_mass_kg)
    phase = "ascent"
    peak = 0.0
    trace = []
    brake_switches = 0
    last_brake = False
    integration_step_s = integration_step_s or dt_s
    if dt_s <= 0 or integration_step_s <= 0 or integration_step_s > dt_s:
        raise ValueError("Use positive steps with integration step <= control interval")
    for tick in range(math.ceil(120.0 / dt_s)):
        altitude = max(0.0, state.altitude_m)
        peak = max(peak, altitude)
        ambient, density = atmosphere(altitude)
        if tick * dt_s < ascent_burn_s - 1e-12:
            command = 1000.0
        elif state.radial_velocity_m_s >= 0:
            phase = "coast"
            command = None
        else:
            phase = "descent"
            thrust, _ = nozzle.evaluate(landing_pressure_kpa, ambient)
            g = EARTH_MU_M3_S2 / state.radius_m**2
            speed = abs(state.radial_velocity_m_s)
            drag_assist = 0.5 * density * vehicle.drag_coefficient * vehicle.frontal_area_m2 * speed**2 / state.mass_kg
            upward_acceleration = thrust / state.mass_kg - g + drag_assist
            stopping_distance = speed**2 / (2 * upward_acceleration) if upward_acceleration > 0 else float("inf")
            command = landing_pressure_kpa if altitude <= stopping_distance + margin_m else None
        if state.mass_kg <= vehicle.dry_mass_kg + 1e-9:
            command = None
        braking = phase == "descent" and command is not None
        if braking and not last_brake:
            brake_switches += 1
        last_brake = braking
        trace.append({
            "time_s": state.time_s,
            "altitude_m": altitude,
            "vertical_speed_m_s": state.radial_velocity_m_s,
            "mass_kg": state.mass_kg,
            "phase": phase,
            "command_kpa": command,
        })
        advanced = 0.0
        while advanced < dt_s - 1e-12:
            sub_dt = min(integration_step_s, dt_s - advanced)
            new = step(state, command, sub_dt, nozzle, vehicle, lambda _: 0.0)
            peak = max(peak, new.altitude_m)
            if new.radius_m <= EARTH_RADIUS_M and new.radial_velocity_m_s < 0:
                fraction = (state.radius_m - EARTH_RADIUS_M) / (state.radius_m - new.radius_m)
                impact_speed = state.radial_velocity_m_s + fraction * (
                    new.radial_velocity_m_s - state.radial_velocity_m_s
                )
                impact_mass = state.mass_kg + fraction * (new.mass_kg - state.mass_kg)
                result = {
                    "ascent_burn_s": ascent_burn_s,
                    "margin_m": margin_m,
                    "landing_pressure_kpa": landing_pressure_kpa,
                    "control_interval_s": dt_s,
                    "integration_step_s": integration_step_s,
                    "peak_altitude_m": peak,
                    "impact_speed_m_s": abs(impact_speed),
                    "fuel_used_kg": vehicle.initial_mass_kg - impact_mass,
                    "fuel_remaining_kg": impact_mass - vehicle.dry_mass_kg,
                    "braking_burn_starts": brake_switches,
                    "touchdown_time_s": state.time_s + fraction * sub_dt,
                    "soft_landing": abs(impact_speed) <= 2.0,
                }
                return result, trace
            state = new
            advanced += sub_dt
    raise RuntimeError("No ground return within 120 s")


def main() -> None:
    rows = []
    for burn in (4.0, 4.5, 5.0, 5.5, 6.0):
        for margin in (-30.0, 0.0, 30.0, 60.0):
            for pressure in (800.0, 900.0, 1000.0):
                result, _ = run_hop(burn, margin, pressure)
                rows.append(result)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "hop_feasibility_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidates = sorted(rows, key=lambda row: (
        not row["soft_landing"], abs(row["peak_altitude_m"] - 500.0),
        row["impact_speed_m_s"], row["fuel_used_kg"]
    ))
    report = {
        "status": "idealized heuristic feasibility sweep, provisional nozzle map",
        "tested_cases": len(rows),
        "soft_landings": sum(row["soft_landing"] for row in rows),
        "near_500m_soft_landings": [row for row in candidates if row["soft_landing"]
                                     and abs(row["peak_altitude_m"] - 500) < 100][:10],
        "closest_to_500m": min(rows, key=lambda row: abs(row["peak_altitude_m"] - 500)),
        "lowest_impact_speed": min(rows, key=lambda row: row["impact_speed_m_s"]),
        "nozzle_map_sha256": NozzleMap(DEFAULT_MAP).sha256,
    }
    (RESULTS / "hop_feasibility_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
