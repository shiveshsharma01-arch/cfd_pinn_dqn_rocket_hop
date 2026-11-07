"""Checks for nozzle-map coupling, burnout, and unforced orbital motion."""

import math
import unittest

from flight_model import (
    DEFAULT_MAP, EARTH_MU_M3_S2, EARTH_RADIUS_M, NozzleMap, State,
    Vehicle, atmosphere, simulate, step,
)


class FlightModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nozzle = NozzleMap(DEFAULT_MAP)

    def test_map_uses_ambient_pressure_and_rejects_extrapolation(self):
        sea, mdot = self.nozzle.evaluate(800.0, 101325.0)
        vacuum, mdot_vac = self.nozzle.evaluate(800.0, 0.0)
        self.assertAlmostEqual(sea, 1744.4408412851442, places=5)
        self.assertAlmostEqual(mdot, 3.4218757520047265, places=6)
        self.assertAlmostEqual(mdot, mdot_vac)
        self.assertAlmostEqual(vacuum - sea, 101325.0 * self.nozzle.exit_area_m2)
        with self.assertRaises(ValueError):
            self.nozzle.evaluate(1050.0, 101325.0)

    def test_burnout_splits_step_and_conserves_propellant(self):
        vehicle = Vehicle(initial_mass_kg=52.0, dry_mass_kg=50.0,
                          drag_coefficient=0, frontal_area_m2=0)
        initial = State(0.0, EARTH_RADIUS_M, 0.0, 0.0, 0.0, 52.0)
        final = step(initial, 800.0, 2.0, self.nozzle, vehicle, lambda _: 0.0)
        self.assertAlmostEqual(final.mass_kg, 50.0, places=9)
        self.assertGreater(final.radial_velocity_m_s, 0.0)
        self.assertAlmostEqual(final.time_s, 2.0)

    def test_unpowered_circular_orbit_stays_nearly_circular(self):
        vehicle = Vehicle(drag_coefficient=0, frontal_area_m2=0)
        radius = EARTH_RADIUS_M + 200_000.0
        speed = math.sqrt(EARTH_MU_M3_S2 / radius)
        state = State(0.0, radius, 0.0, 0.0, speed, vehicle.dry_mass_kg)
        for _ in range(600):
            state = step(state, None, 1.0, self.nozzle, vehicle)
        self.assertLess(abs(state.radius_m - radius), 0.01)
        self.assertLess(abs(state.radial_velocity_m_s), 0.001)
        self.assertAlmostEqual(state.mass_kg, vehicle.dry_mass_kg)

    def test_example_returns_to_ground_without_negative_altitude(self):
        history = simulate(self.nozzle, Vehicle(), lambda _: 800.0,
                           duration_s=120.0, dt_s=0.1)
        self.assertGreater(max(row["altitude_m"] for row in history), 1000.0)
        self.assertAlmostEqual(history[-1]["altitude_m"], 0.0)
        self.assertTrue(all(row["mass_kg"] >= 50.0 - 1e-8 for row in history))
        self.assertGreater(history[-1]["time_s"], 30.0)
        pressure, density = atmosphere(0.0)
        self.assertAlmostEqual(pressure, 101325.0)
        self.assertGreater(density, 1.0)


if __name__ == "__main__":
    unittest.main()
