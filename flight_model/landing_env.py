"""Vertical launch-to-touchdown control environment using the frozen nozzle map.

This is an educational point-mass model. A first-order spool variable delays
thrust and mass-flow response; pressure within the 600–1000 kPa map changes
immediately when an on command is selected. Off uses the previous pressure
while spool decays. This actuator is an explicit assumption, not CFD data.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from dqn_env import ACTIONS_KPA, ballistic_apogee_estimate
from flight_model import (
    DEFAULT_MAP, EARTH_MU_M3_S2, EARTH_RADIUS_M, NozzleMap,
    State, Vehicle, atmosphere, step,
)


OBSERVATION_SIZE = 11


@dataclass(frozen=True)
class LandingConfig:
    target_height_m: float = 500.0
    initial_mass_kg: float = 100.0
    control_interval_s: float = 0.05
    integration_step_s: float = 0.01
    actuator_time_constant_s: float = 0.05
    max_time_s: float = 60.0
    touchdown_limit_m_s: float = 5.0
    height_tolerance_m: float = 50.0
    minimum_fuel_remaining_kg: float = 5.0


class ScaledNozzle:
    """Scale mapped thrust and mass flow by a held actuator opening."""

    def __init__(self, base: NozzleMap, opening: float):
        self.base = base
        self.opening = opening

    def evaluate(self, pressure_kpa: float, ambient_pa: float) -> tuple[float, float]:
        thrust, mass_flow = self.base.evaluate(pressure_kpa, ambient_pa)
        return self.opening * thrust, self.opening * mass_flow


class LandingEnv:
    def __init__(self, nozzle: NozzleMap | None = None, seed: int = 0,
                 config: LandingConfig | None = None):
        self.nozzle = nozzle if nozzle is not None else NozzleMap(DEFAULT_MAP)
        self.rng = random.Random(seed)
        self.config = config or LandingConfig()
        self.vehicle = Vehicle(initial_mass_kg=self.config.initial_mass_kg)
        self.state = State(0.0, EARTH_RADIUS_M, 0.0, 0.0, 0.0,
                           self.vehicle.initial_mass_kg)
        self.spool = 0.0
        self.active_pressure_kpa = 600.0
        self.peak_height_m = 0.0
        self.has_lifted = False
        self.done = False
        self.previous_action = 0
        self.last_info: dict = {}
        self._validate()

    def _validate(self):
        c = self.config
        if not 200 <= c.target_height_m <= 800:
            raise ValueError("Landing target outside 200–800 m model range")
        if c.initial_mass_kg <= self.vehicle.dry_mass_kg:
            raise ValueError("Initial mass must exceed dry mass")
        if min(c.control_interval_s, c.integration_step_s,
               c.actuator_time_constant_s, c.max_time_s) <= 0:
            raise ValueError("Time settings must be positive")
        if c.integration_step_s > c.control_interval_s:
            raise ValueError("Physics step must not exceed control interval")

    def reset(self, *, target_height_m: float | None = None,
              initial_mass_kg: float | None = None) -> tuple[np.ndarray, dict]:
        target = target_height_m if target_height_m is not None else self.config.target_height_m
        mass = initial_mass_kg if initial_mass_kg is not None else self.config.initial_mass_kg
        self.config = LandingConfig(
            target_height_m=target, initial_mass_kg=mass,
            control_interval_s=self.config.control_interval_s,
            integration_step_s=self.config.integration_step_s,
            actuator_time_constant_s=self.config.actuator_time_constant_s,
            max_time_s=self.config.max_time_s,
            touchdown_limit_m_s=self.config.touchdown_limit_m_s,
            height_tolerance_m=self.config.height_tolerance_m,
            minimum_fuel_remaining_kg=self.config.minimum_fuel_remaining_kg,
        )
        self.vehicle = Vehicle(initial_mass_kg=mass)
        self._validate()
        self.state = State(0.0, EARTH_RADIUS_M, 0.0, 0.0, 0.0, mass)
        self.spool = 0.0
        self.active_pressure_kpa = 600.0
        self.peak_height_m = 0.0
        self.has_lifted = False
        self.done = False
        self.previous_action = 0
        self.last_info = self._info()
        return self._observation(), self.last_info.copy()

    def _stopping_distance(self, pressure_kpa: float = 900.0) -> float:
        speed = max(0.0, -self.state.radial_velocity_m_s)
        if speed == 0:
            return 0.0
        ambient, density = atmosphere(self.state.altitude_m)
        thrust, _ = self.nozzle.evaluate(pressure_kpa, ambient)
        gravity = EARTH_MU_M3_S2 / self.state.radius_m**2
        drag = (0.5 * density * self.vehicle.drag_coefficient
                * self.vehicle.frontal_area_m2 * speed**2 / self.state.mass_kg)
        upward = thrust / self.state.mass_kg - gravity + drag
        if upward <= 0:
            return 5000.0
        return speed**2 / (2 * upward) + speed * self.config.actuator_time_constant_s

    def _observation(self) -> np.ndarray:
        ambient, _ = atmosphere(self.state.altitude_m)
        ballistic = ballistic_apogee_estimate(self.state)
        return np.array([
            self.state.altitude_m / 800.0,
            self.state.radial_velocity_m_s / 150.0,
            (self.state.mass_kg - self.vehicle.dry_mass_kg) / 60.0,
            self.peak_height_m / 800.0,
            self.config.target_height_m / 800.0,
            self.state.time_s / self.config.max_time_s,
            ambient / 101_325.0,
            self.spool,
            self.active_pressure_kpa / 1000.0,
            min(self._stopping_distance(), 2000.0) / 800.0,
            min(ballistic, 2000.0) / 800.0,
        ], dtype=np.float32)

    def _info(self) -> dict:
        fuel_remaining = max(0.0, self.state.mass_kg - self.vehicle.dry_mass_kg)
        return {
            "time_s": self.state.time_s,
            "height_m": max(0.0, self.state.altitude_m),
            "vertical_speed_m_s": self.state.radial_velocity_m_s,
            "mass_kg": self.state.mass_kg,
            "fuel_remaining_kg": fuel_remaining,
            "peak_height_m": self.peak_height_m,
            "target_height_m": self.config.target_height_m,
            "peak_height_error_percent": (
                100.0 * abs(self.peak_height_m - self.config.target_height_m)
                / self.config.target_height_m
            ),
            "spool": self.spool,
            "active_pressure_kpa": self.active_pressure_kpa,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("Reset environment after touchdown or time limit")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(ACTIONS_KPA):
            raise ValueError("Action must be an integer from 0 to 5")
        command = ACTIONS_KPA[action]
        if command is not None:
            self.active_pressure_kpa = command
        target_opening = 0.0 if command is None else 1.0
        initial_mass = self.state.mass_kg
        old_height = max(0.0, self.state.altitude_m)
        new_peak = self.peak_height_m
        impact_speed = None
        launch_failed = False
        terminated = False
        truncated = False
        n = math.ceil(self.config.control_interval_s / self.config.integration_step_s)
        for sub in range(n):
            remaining = self.config.control_interval_s - sub * self.config.integration_step_s
            dt = min(self.config.integration_step_s, remaining,
                     self.config.max_time_s - self.state.time_s)
            if dt <= 1e-12:
                truncated = True
                break
            decay = math.exp(-dt / self.config.actuator_time_constant_s)
            average_opening = (target_opening + (self.spool - target_opening)
                               * (1 - decay) * self.config.actuator_time_constant_s / dt)
            self.spool = target_opening + (self.spool - target_opening) * decay
            actual_pressure = self.active_pressure_kpa if average_opening > 1e-8 else None
            new_state = step(self.state, actual_pressure, dt,
                             ScaledNozzle(self.nozzle, average_opening),
                             self.vehicle, lambda _: 0.0)
            new_peak = max(new_peak, new_state.altitude_m)
            if new_state.radius_m <= EARTH_RADIUS_M and new_state.radial_velocity_m_s < 0:
                if not self.has_lifted:
                    # The launch pad holds the vehicle until thrust exceeds weight.
                    self.state = State(new_state.time_s, EARTH_RADIUS_M, 0.0,
                                       0.0, 0.0, new_state.mass_kg)
                    continue
                fraction = (self.state.radius_m - EARTH_RADIUS_M) / (
                    self.state.radius_m - new_state.radius_m
                )
                impact_speed = abs(self.state.radial_velocity_m_s + fraction * (
                    new_state.radial_velocity_m_s - self.state.radial_velocity_m_s
                ))
                mass = self.state.mass_kg + fraction * (
                    new_state.mass_kg - self.state.mass_kg
                )
                self.state = State(
                    self.state.time_s + fraction * dt, EARTH_RADIUS_M, 0.0,
                    -impact_speed, 0.0, max(self.vehicle.dry_mass_kg, mass),
                )
                terminated = True
                break
            self.state = new_state
            if self.state.altitude_m > 1.0:
                self.has_lifted = True
            if self.state.time_s >= self.config.max_time_s - 1e-10:
                truncated = True
                break
        if not self.has_lifted and self.state.time_s >= 2.0:
            terminated = True
            launch_failed = True
        self.peak_height_m = new_peak
        used = max(0.0, initial_mass - self.state.mass_kg)
        self.previous_action = int(action)
        # Dense shaping for ascent accuracy, with final touchdown judged below.
        if self.state.radial_velocity_m_s >= 0:
            ballistic = ballistic_apogee_estimate(self.state)
            reward = -0.00002 * abs(ballistic - self.config.target_height_m)
        else:
            # Once descending, discourage staying far inside the stopping region.
            stop_gap = self.state.altitude_m - self._stopping_distance()
            reward = -0.00005 * min(abs(stop_gap), 1000.0)
        reward -= 0.01 * used
        if self.peak_height_m > self.config.target_height_m + self.config.height_tolerance_m:
            reward -= 0.001 * (self.peak_height_m - self.config.target_height_m)
        if launch_failed:
            reward -= 500.0
            success = False
        elif terminated:
            height_error = abs(self.peak_height_m - self.config.target_height_m)
            remaining_fuel = self.state.mass_kg - self.vehicle.dry_mass_kg
            reached_height = self.peak_height_m >= (
                self.config.target_height_m - self.config.height_tolerance_m
            )
            reward += 12.0 * (1.0 - min(height_error / 100.0, 3.0))
            if reached_height:
                reward += (25.0 * (1.0 - min(impact_speed / 5.0, 5.0))
                           + 4.0 * min(remaining_fuel / 10.0, 1.0))
            else:
                reward -= 30.0 + 75.0 * (
                    self.config.target_height_m - self.peak_height_m
                ) / self.config.target_height_m
            success = (height_error <= self.config.height_tolerance_m
                       and impact_speed <= self.config.touchdown_limit_m_s
                       and remaining_fuel >= self.config.minimum_fuel_remaining_kg)
            if success:
                reward += 25.0
        else:
            success = False
        self.done = terminated or truncated
        info = self._info()
        info.update({
            "impact_speed_m_s": impact_speed,
            "touchdown_speed_percent_of_limit": (
                100.0 * impact_speed / self.config.touchdown_limit_m_s
                if impact_speed is not None else None
            ),
            "success": success,
            "terminal_reason": ("launch_failed" if launch_failed else
                                "touchdown" if terminated else
                                "time_limit" if truncated else None),
            "last_command_kpa": command,
            "height_change_m": max(0.0, self.state.altitude_m) - old_height,
        })
        self.last_info = info
        return self._observation(), float(reward), terminated, truncated, info


def landing_heuristic(env: LandingEnv, cutoff_offset_m: float | None = None,
                      brake_margin_m: float = 0.0,
                      brake_pressure_kpa: float = 900.0,
                      terminal_accel_m_s2: float | None = None) -> int:
    """Reference controller: timed or target-aware ascent, then braking."""
    if cutoff_offset_m is None:
        if env.state.time_s < 5.15 - 1e-9:
            return 5
    elif env.state.radial_velocity_m_s >= 0:
        if ballistic_apogee_estimate(env.state) < (
            env.config.target_height_m - cutoff_offset_m
        ):
            return 5
    if env.state.radial_velocity_m_s >= 0:
        return 0
    action = ACTIONS_KPA.index(brake_pressure_kpa)
    if terminal_accel_m_s2 is not None and env.state.altitude_m < 20.0:
        speed = -env.state.radial_velocity_m_s
        desired = 1.0 + math.sqrt(2.0 * terminal_accel_m_s2 * env.state.altitude_m)
        return action if speed > desired else 0
    return action if env.state.altitude_m <= env._stopping_distance(brake_pressure_kpa) + brake_margin_m else 0
