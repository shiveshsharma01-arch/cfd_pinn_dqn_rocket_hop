"""Fixed-step pressure-control environment for the provisional nozzle flight model.

The API follows Gymnasium's reset/step return shape without requiring Gymnasium.
Episodes end at apogee, ground impact, or a time limit. The objective is apogee
tracking with a small propellant cost, not orbital insertion or landing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from flight_model import (
    DEFAULT_MAP, EARTH_MU_M3_S2, EARTH_RADIUS_M, NozzleMap, State,
    Vehicle, atmosphere, pitch_schedule, step,
)


ACTIONS_KPA: tuple[float | None, ...] = (None, 600.0, 700.0, 800.0, 900.0, 1000.0)
OBSERVATION_SIZE = 8


@dataclass(frozen=True)
class Mission:
    target_apogee_m: float = 2400.0
    initial_mass_kg: float = 100.0
    max_time_s: float = 90.0
    control_interval_s: float = 0.5
    integration_step_s: float = 0.1


def ballistic_apogee_estimate(state: State) -> float:
    """Two-body coast apogee; drag is deliberately omitted from this estimate."""
    r = state.radius_m
    vr = state.radial_velocity_m_s
    vt = state.tangential_velocity_m_s
    energy = 0.5 * (vr * vr + vt * vt) - EARTH_MU_M3_S2 / r
    if energy >= 0:
        return float("inf")
    angular_momentum = r * vt
    eccentricity_sq = max(0.0, 1.0 + 2.0 * energy * angular_momentum**2 / EARTH_MU_M3_S2**2)
    semi_major_axis = -EARTH_MU_M3_S2 / (2.0 * energy)
    return max(0.0, semi_major_axis * (1.0 + math.sqrt(eccentricity_sq)) - EARTH_RADIUS_M)


class NozzleFlightEnv:
    """One episode controls pressure until the vehicle reaches its apogee."""

    def __init__(self, nozzle: NozzleMap | None = None, seed: int = 0,
                 mission: Mission | None = None):
        self.nozzle = nozzle if nozzle is not None else NozzleMap(DEFAULT_MAP)
        self.rng = random.Random(seed)
        self.mission = mission or Mission()
        self.vehicle = Vehicle(initial_mass_kg=self.mission.initial_mass_kg)
        self.state = State(0.0, EARTH_RADIUS_M, 0.0, 0.0, 0.0,
                           self.vehicle.initial_mass_kg)
        self.previous_action = 0
        self.previous_potential = 0.0
        self.peak_altitude_m = 0.0
        self.done = False
        self.command_history: list[int] = []
        self._validate_mission()

    def _validate_mission(self) -> None:
        m = self.mission
        if not 1000 <= m.target_apogee_m <= 4000:
            raise ValueError("Target must be between 1000 and 4000 m for this demonstrator")
        if m.initial_mass_kg <= self.vehicle.dry_mass_kg:
            raise ValueError("Initial mass must exceed dry mass")
        if m.max_time_s <= 0 or m.control_interval_s <= 0 or m.integration_step_s <= 0:
            raise ValueError("Time settings must be positive")

    def _observation(self) -> np.ndarray:
        s = self.state
        ambient, _ = atmosphere(s.altitude_m)
        ballistic = ballistic_apogee_estimate(s)
        return np.array([
            s.altitude_m / 4000.0,
            s.radial_velocity_m_s / 250.0,
            s.tangential_velocity_m_s / 150.0,
            (s.mass_kg - self.vehicle.dry_mass_kg) / 60.0,
            self.mission.target_apogee_m / 4000.0,
            min(ballistic, 10_000.0) / 4000.0,
            ambient / 101_325.0,
            s.time_s / self.mission.max_time_s,
        ], dtype=np.float32)

    def _potential(self) -> float:
        predicted = ballistic_apogee_estimate(self.state)
        return -min(5.0, abs(predicted - self.mission.target_apogee_m) / 1000.0)

    def reset(self, *, target_apogee_m: float | None = None,
              initial_mass_kg: float | None = None) -> tuple[np.ndarray, dict]:
        target = target_apogee_m if target_apogee_m is not None else self.rng.uniform(1600.0, 2800.0)
        mass = initial_mass_kg if initial_mass_kg is not None else self.rng.uniform(95.0, 105.0)
        self.mission = Mission(target_apogee_m=target, initial_mass_kg=mass,
                               max_time_s=self.mission.max_time_s,
                               control_interval_s=self.mission.control_interval_s,
                               integration_step_s=self.mission.integration_step_s)
        self.vehicle = Vehicle(initial_mass_kg=mass)
        self._validate_mission()
        self.state = State(0.0, EARTH_RADIUS_M, 0.0, 0.0, 0.0, mass)
        self.previous_action = 0
        self.previous_potential = self._potential()
        self.peak_altitude_m = 0.0
        self.done = False
        self.command_history = []
        return self._observation(), self._info()

    def _info(self) -> dict:
        return {
            "target_apogee_m": self.mission.target_apogee_m,
            "peak_altitude_m": self.peak_altitude_m,
            "apogee_error_m": self.peak_altitude_m - self.mission.target_apogee_m,
            "fuel_used_kg": self.vehicle.initial_mass_kg - self.state.mass_kg,
            "time_s": self.state.time_s,
            "mass_kg": self.state.mass_kg,
            "altitude_m": max(0.0, self.state.altitude_m),
            "radial_velocity_m_s": self.state.radial_velocity_m_s,
            "command_count": len(self.command_history),
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("Reset environment before stepping a finished episode")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(ACTIONS_KPA):
            raise ValueError("Action must be an integer from 0 to 5")
        command = ACTIONS_KPA[action]
        self.command_history.append(int(action))
        initial_mass = self.state.mass_kg
        previous = self.state
        n = math.ceil(self.mission.control_interval_s / self.mission.integration_step_s)
        terminated = False
        truncated = False
        for index in range(n):
            remaining = self.mission.control_interval_s - index * self.mission.integration_step_s
            dt = min(self.mission.integration_step_s, remaining,
                     self.mission.max_time_s - self.state.time_s)
            if dt <= 1e-10:
                truncated = True
                break
            new_state = step(self.state, command, dt, self.nozzle, self.vehicle,
                             pitch_schedule)
            self.peak_altitude_m = max(self.peak_altitude_m, new_state.altitude_m)
            if new_state.radius_m <= EARTH_RADIUS_M and new_state.radial_velocity_m_s < 0:
                self.state = State(new_state.time_s, EARTH_RADIUS_M,
                                   new_state.downrange_angle_rad,
                                   new_state.radial_velocity_m_s,
                                   new_state.tangential_velocity_m_s, new_state.mass_kg)
                terminated = True
                break
            if self.state.radial_velocity_m_s > 0 and new_state.radial_velocity_m_s <= 0:
                self.state = new_state
                terminated = True
                break
            self.state = new_state
            if self.state.time_s >= self.mission.max_time_s - 1e-10:
                truncated = True
                break
        self.previous_action = int(action)
        potential = self._potential()
        fuel_used = initial_mass - self.state.mass_kg
        # Potential-difference shaping retains the terminal apogee objective.
        reward = 0.99 * potential - self.previous_potential - 0.015 * fuel_used
        self.previous_potential = potential
        if terminated or truncated:
            error = abs(self.peak_altitude_m - self.mission.target_apogee_m)
            reward += 8.0 * math.exp(-error / 300.0) - 0.003 * error
            if self.peak_altitude_m < 10.0:
                reward -= 3.0
        self.done = terminated or truncated
        info = self._info()
        info["terminated_reason"] = (
            "apogee" if terminated and self.peak_altitude_m > 10.0
            else "ground" if terminated else "time_limit" if truncated else None
        )
        return self._observation(), float(reward), terminated, truncated, info
