"""End-to-end checks for the packaged 500 m takeoff and landing."""

import unittest

import torch

from evaluate_landing import load_policy
from finalize_landing import FINAL_DIR
from flight_model import DEFAULT_MAP, NozzleMap
from guided_landing import GuidedLandingPolicy
from landing_env import LandingConfig, LandingEnv
from train_landing_dqn import run_episode


class LandingMissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.nozzle = NozzleMap(DEFAULT_MAP)
        cls.network = load_policy(FINAL_DIR / "selected_dqn.pt", cls.nozzle)

    def test_nominal_takeoff_and_touchdown(self):
        metrics, trace = run_episode(
            LandingEnv(self.nozzle), GuidedLandingPolicy(self.network),
            record=True,
        )
        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["terminal_reason"], "touchdown")
        self.assertLessEqual(metrics["peak_height_error_percent"], 10.0)
        self.assertLessEqual(metrics["touchdown_speed_percent_of_limit"], 100.0)
        self.assertGreaterEqual(metrics["fuel_remaining_kg"], 5.0)
        self.assertGreater(max(row["height_m"] for row in trace), 450.0)
        self.assertEqual(trace[-1]["height_m"], 0.0)

    def test_integration_step_sensitivity(self):
        results = []
        for dt in (0.01, 0.005):
            env = LandingEnv(self.nozzle, config=LandingConfig(integration_step_s=dt))
            metrics, _ = run_episode(env, GuidedLandingPolicy(self.network))
            self.assertTrue(metrics["success"])
            results.append(metrics)
        self.assertLess(abs(results[0]["peak_height_m"] - results[1]["peak_height_m"]),
                        0.1)
        self.assertLess(abs(results[0]["touchdown_speed_percent_of_limit"]
                            - results[1]["touchdown_speed_percent_of_limit"]), 1.0)


if __name__ == "__main__":
    unittest.main()
