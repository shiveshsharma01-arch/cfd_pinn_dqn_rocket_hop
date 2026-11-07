"""Meaningful checks for DQN action coupling and episode termination."""

import unittest

from dqn_env import ACTIONS_KPA, NozzleFlightEnv, ballistic_apogee_estimate
from flight_model import DEFAULT_MAP, NozzleMap


class DqnEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nozzle = NozzleMap(DEFAULT_MAP)

    def test_pressure_action_consumes_fuel_and_off_does_not(self):
        env = NozzleFlightEnv(self.nozzle)
        observation, _ = env.reset(target_apogee_m=2200, initial_mass_kg=100)
        self.assertEqual(observation.shape, (8,))
        _, _, terminated, _, powered = env.step(3)
        self.assertFalse(terminated)
        self.assertGreater(powered["fuel_used_kg"], 0)
        mass = powered["mass_kg"]
        _, _, _, _, unpowered = env.step(0)
        self.assertAlmostEqual(unpowered["mass_kg"], mass)
        self.assertEqual(ACTIONS_KPA[3], 800.0)

    def test_fixed_pressure_reaches_apogee_and_ends_episode(self):
        env = NozzleFlightEnv(self.nozzle)
        env.reset(target_apogee_m=2400, initial_mass_kg=100)
        for _ in range(180):
            _, reward, terminated, truncated, info = env.step(3)
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertEqual(info["terminated_reason"], "apogee")
        self.assertGreater(info["peak_altitude_m"], 2500)
        self.assertAlmostEqual(info["mass_kg"], 50.0, places=7)
        self.assertLess(abs(info["apogee_error_m"]), 1000)
        self.assertGreater(ballistic_apogee_estimate(env.state), 0)
        with self.assertRaises(RuntimeError):
            env.step(3)


if __name__ == "__main__":
    unittest.main()
