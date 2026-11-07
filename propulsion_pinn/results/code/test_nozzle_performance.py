"""Independent checks of axisymmetric mass-flow and thrust integration."""

import math
import unittest

import numpy as np

from nozzle_performance import exit_quadrature, integrate_exit_profile


class ExitIntegrationTests(unittest.TestCase):
    def test_uniform_profile_matches_analytic_rocket_thrust(self):
        exit_radius = 0.0316
        radius, weights = exit_quadrature(exit_radius, 64)
        area = math.pi * exit_radius**2
        result = integrate_exit_profile(
            radius, weights, np.full_like(radius, 2.0),
            np.full_like(radius, 3.0), np.full_like(radius, 10.0), 4.0)
        self.assertAlmostEqual(result["exit_area_m2"], area, places=12)
        self.assertAlmostEqual(result["mass_flow_kg_s"], 6 * area, places=12)
        self.assertAlmostEqual(result["momentum_force_n"], 18 * area, places=12)
        self.assertAlmostEqual(result["pressure_force_n"], 6 * area, places=12)
        self.assertAlmostEqual(result["thrust_n"], 24 * area, places=12)
        self.assertAlmostEqual(result["exit_velocity_mass_weighted_m_s"], 3.0, places=12)

    def test_ambient_pressure_changes_thrust_by_exit_area(self):
        radius, weights = exit_quadrature(0.0316, 64)
        args = (radius, weights, np.ones_like(radius),
                np.ones_like(radius), np.full_like(radius, 100000.0))
        first = integrate_exit_profile(*args, 50000.0)
        second = integrate_exit_profile(*args, 60000.0)
        self.assertAlmostEqual(first["thrust_n"] - second["thrust_n"],
                               10000.0 * math.pi * 0.0316**2, places=9)


if __name__ == "__main__":
    unittest.main()
