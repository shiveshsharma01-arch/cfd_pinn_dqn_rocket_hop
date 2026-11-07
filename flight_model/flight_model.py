"""Provisional 2-D rocket flight model driven by a frozen nozzle performance map.

This is a point-mass, nonrotating-Earth demonstrator, not a launch-vehicle design.
Pitch is measured from local vertical toward the downrange direction.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


EARTH_RADIUS_M = 6_371_000.0
EARTH_MU_M3_S2 = 3.986004418e14
SEA_LEVEL_PRESSURE_PA = 101_325.0
ISOTHERMAL_TEMPERATURE_K = 288.15
AIR_GAS_CONSTANT_J_KG_K = 287.05
SEA_LEVEL_GRAVITY_M_S2 = EARTH_MU_M3_S2 / EARTH_RADIUS_M**2
ATMOSPHERE_SCALE_HEIGHT_M = (
    AIR_GAS_CONSTANT_J_KG_K * ISOTHERMAL_TEMPERATURE_K / SEA_LEVEL_GRAVITY_M_S2
)
DEFAULT_MAP = (
    Path(__file__).resolve().parents[1]
    / "propulsion_pinn/results/checkpoints/final/exit_performance.csv"
)


@dataclass(frozen=True)
class NozzlePoint:
    pressure_kpa: float
    mass_flow_kg_s: float
    vacuum_thrust_n: float


class NozzleMap:
    """Piecewise-linear pressure map; ambient changes only the pressure force."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        with self.path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.points = tuple(
            NozzlePoint(
                pressure_kpa=float(row["inlet_total_pressure_kpa"]),
                mass_flow_kg_s=float(row["mass_flow_kg_s"]),
                vacuum_thrust_n=float(row["thrust_n"]),
            )
            for row in rows
        )
        if len(self.points) < 2 or any(
            a.pressure_kpa >= b.pressure_kpa
            for a, b in zip(self.points, self.points[1:])
        ):
            raise ValueError("Nozzle map requires increasing pressure points")
        if any(p.mass_flow_kg_s <= 0 or p.vacuum_thrust_n <= 0 for p in self.points):
            raise ValueError("Nozzle map requires positive flow and vacuum thrust")
        areas = [float(row["exit_area_m2"]) for row in rows]
        if any(not math.isclose(area, areas[0], rel_tol=1e-8) for area in areas):
            raise ValueError("Exit area changes between map rows")
        self.exit_area_m2 = areas[0]
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def evaluate(self, inlet_kpa: float, ambient_pa: float) -> tuple[float, float]:
        """Return positive thrust N and positive fuel consumption kg/s."""
        if not math.isfinite(ambient_pa) or ambient_pa < 0:
            raise ValueError("Ambient pressure must be nonnegative absolute Pa")
        if not math.isfinite(inlet_kpa):
            raise ValueError("Inlet pressure must be finite")
        if inlet_kpa < self.points[0].pressure_kpa or inlet_kpa > self.points[-1].pressure_kpa:
            raise ValueError("Command outside measured 600–1000 kPa map range")
        for left, right in zip(self.points, self.points[1:]):
            if left.pressure_kpa <= inlet_kpa <= right.pressure_kpa:
                fraction = (inlet_kpa - left.pressure_kpa) / (
                    right.pressure_kpa - left.pressure_kpa
                )
                mdot = left.mass_flow_kg_s + fraction * (
                    right.mass_flow_kg_s - left.mass_flow_kg_s
                )
                vacuum_thrust = left.vacuum_thrust_n + fraction * (
                    right.vacuum_thrust_n - left.vacuum_thrust_n
                )
                return vacuum_thrust - ambient_pa * self.exit_area_m2, mdot
        raise AssertionError("Pressure range check and interpolation disagree")


@dataclass(frozen=True)
class Vehicle:
    initial_mass_kg: float = 100.0
    dry_mass_kg: float = 50.0
    drag_coefficient: float = 0.5
    frontal_area_m2: float = 0.01

    def __post_init__(self):
        if self.dry_mass_kg <= 0 or self.initial_mass_kg <= self.dry_mass_kg:
            raise ValueError("Vehicle must start with positive propellant and dry mass")
        if self.drag_coefficient < 0 or self.frontal_area_m2 < 0:
            raise ValueError("Drag inputs cannot be negative")


@dataclass(frozen=True)
class State:
    time_s: float
    radius_m: float
    downrange_angle_rad: float
    radial_velocity_m_s: float
    tangential_velocity_m_s: float
    mass_kg: float

    @property
    def altitude_m(self) -> float:
        return self.radius_m - EARTH_RADIUS_M

    @property
    def speed_m_s(self) -> float:
        return math.hypot(self.radial_velocity_m_s, self.tangential_velocity_m_s)


def atmosphere(altitude_m: float) -> tuple[float, float]:
    """Isothermal exponential atmosphere; intentionally simple."""
    pressure = SEA_LEVEL_PRESSURE_PA * math.exp(
        -max(0.0, altitude_m) / ATMOSPHERE_SCALE_HEIGHT_M
    )
    return pressure, pressure / (AIR_GAS_CONSTANT_J_KG_K * ISOTHERMAL_TEMPERATURE_K)


def pitch_schedule(time_s: float) -> float:
    """Illustrative guidance: vertical 5 s, then rotate to 60 degrees by 25 s."""
    return math.radians(60.0) * min(1.0, max(0.0, (time_s - 5.0) / 20.0))


def _rates(
    values: tuple[float, float, float, float, float],
    time_s: float,
    command_kpa: float | None,
    nozzle: NozzleMap,
    vehicle: Vehicle,
    pitch: Callable[[float], float],
) -> tuple[float, float, float, float, float]:
    radius, _, vr, vt, mass = values
    ambient, density = atmosphere(radius - EARTH_RADIUS_M)
    # The caller splits a step exactly at burnout. Keep the final RK4 stage
    # powered even when its trial mass reaches the dry-mass boundary.
    powered = command_kpa is not None
    thrust, mdot = nozzle.evaluate(command_kpa, ambient) if powered else (0.0, 0.0)
    speed = math.hypot(vr, vt)
    drag_over_mass = 0.5 * density * vehicle.drag_coefficient * vehicle.frontal_area_m2 * speed / mass
    angle = pitch(time_s)
    return (
        vr,
        vt / radius,
        vt * vt / radius - EARTH_MU_M3_S2 / radius**2
        + thrust * math.cos(angle) / mass - drag_over_mass * vr,
        -vr * vt / radius + thrust * math.sin(angle) / mass - drag_over_mass * vt,
        -mdot,
    )


def _rk4(
    state: State,
    dt_s: float,
    command_kpa: float | None,
    nozzle: NozzleMap,
    vehicle: Vehicle,
    pitch: Callable[[float], float],
) -> State:
    y = (
        state.radius_m, state.downrange_angle_rad, state.radial_velocity_m_s,
        state.tangential_velocity_m_s, state.mass_kg,
    )

    def add(base, slope, scale):
        return tuple(a + scale * b for a, b in zip(base, slope))

    k1 = _rates(y, state.time_s, command_kpa, nozzle, vehicle, pitch)
    k2 = _rates(add(y, k1, dt_s / 2), state.time_s + dt_s / 2,
                command_kpa, nozzle, vehicle, pitch)
    k3 = _rates(add(y, k2, dt_s / 2), state.time_s + dt_s / 2,
                command_kpa, nozzle, vehicle, pitch)
    k4 = _rates(add(y, k3, dt_s), state.time_s + dt_s,
                command_kpa, nozzle, vehicle, pitch)
    new = tuple(
        value + dt_s * (a + 2*b + 2*c + d) / 6
        for value, a, b, c, d in zip(y, k1, k2, k3, k4)
    )
    return State(state.time_s + dt_s, *new)


def step(
    state: State,
    command_kpa: float | None,
    dt_s: float,
    nozzle: NozzleMap,
    vehicle: Vehicle,
    pitch: Callable[[float], float] = pitch_schedule,
) -> State:
    """Advance one command; split a step at fuel exhaustion."""
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Time step must be positive and finite")
    if command_kpa is None or state.mass_kg <= vehicle.dry_mass_kg:
        return _rk4(state, dt_s, None, nozzle, vehicle, pitch)
    ambient, _ = atmosphere(state.altitude_m)
    _, mdot = nozzle.evaluate(command_kpa, ambient)
    burn_s = min(dt_s, (state.mass_kg - vehicle.dry_mass_kg) / mdot)
    if burn_s <= 0:
        return _rk4(state, dt_s, None, nozzle, vehicle, pitch)
    burned = _rk4(state, burn_s, command_kpa, nozzle, vehicle, pitch)
    burned = State(burned.time_s, burned.radius_m, burned.downrange_angle_rad,
                   burned.radial_velocity_m_s, burned.tangential_velocity_m_s,
                   max(vehicle.dry_mass_kg, burned.mass_kg))
    if burn_s < dt_s:
        return _rk4(burned, dt_s - burn_s, None, nozzle, vehicle, pitch)
    return burned


def simulate(
    nozzle: NozzleMap,
    vehicle: Vehicle,
    command: Callable[[State], float | None],
    duration_s: float = 180.0,
    dt_s: float = 0.1,
    pitch: Callable[[float], float] = pitch_schedule,
) -> list[dict[str, float | None]]:
    """Return time history; command is held constant within each step."""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Duration must be positive and finite")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Time step must be positive and finite")
    state = State(0.0, EARTH_RADIUS_M, 0.0, 0.0, 0.0, vehicle.initial_mass_kg)
    history = []

    def record(current: State) -> dict[str, float | None]:
        ambient, density = atmosphere(current.altitude_m)
        requested = command(current)
        active = requested if current.mass_kg > vehicle.dry_mass_kg + 1e-9 else None
        thrust, mdot = nozzle.evaluate(active, ambient) if active is not None else (0.0, 0.0)
        return {
            "time_s": current.time_s,
            "altitude_m": max(0.0, current.altitude_m),
            "downrange_m": EARTH_RADIUS_M * current.downrange_angle_rad,
            "radial_velocity_m_s": current.radial_velocity_m_s,
            "tangential_velocity_m_s": current.tangential_velocity_m_s,
            "speed_m_s": current.speed_m_s,
            "mass_kg": current.mass_kg,
            "ambient_pressure_pa": ambient,
            "density_kg_m3": density,
            "dynamic_pressure_pa": 0.5 * density * current.speed_m_s**2,
            "command_kpa": active,
            "thrust_n": thrust,
            "mass_flow_kg_s": mdot,
            "pitch_deg": math.degrees(pitch(current.time_s)),
        }

    count = math.ceil(duration_s / dt_s)
    for _ in range(count + 1):
        row = record(state)
        history.append(row)
        if state.time_s >= duration_s - 1e-10:
            break
        next_state = step(state, row["command_kpa"], min(dt_s, duration_s - state.time_s),
                          nozzle, vehicle, pitch)
        if next_state.radius_m <= EARTH_RADIUS_M:
            fraction = (state.radius_m - EARTH_RADIUS_M) / (
                state.radius_m - next_state.radius_m
            )
            impact = State(
                state.time_s + fraction * (next_state.time_s - state.time_s),
                EARTH_RADIUS_M,
                state.downrange_angle_rad + fraction * (
                    next_state.downrange_angle_rad - state.downrange_angle_rad
                ),
                state.radial_velocity_m_s + fraction * (
                    next_state.radial_velocity_m_s - state.radial_velocity_m_s
                ),
                state.tangential_velocity_m_s + fraction * (
                    next_state.tangential_velocity_m_s - state.tangential_velocity_m_s
                ),
                state.mass_kg + fraction * (next_state.mass_kg - state.mass_kg),
            )
            history.append(record(impact))
            break
        state = next_state
    return history
