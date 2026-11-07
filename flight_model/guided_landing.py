"""DQN ascent and model-based descent for the 500 m hop.

The DQN proposes ascent throttle. Descent is a separate feedback controller.
This distinction matters: the DQN checkpoint alone does not land reliably.
"""

from __future__ import annotations

import numpy as np

from landing_env import LandingEnv, landing_heuristic
from train_dqn import greedy_action
from train_landing_dqn import LandingQNetwork


class GuidedLandingPolicy:
    def __init__(self, network: LandingQNetwork):
        self.network = network

    def __call__(self, env: LandingEnv, observation: np.ndarray) -> int:
        if env.state.radial_velocity_m_s >= 0:
            return greedy_action(self.network, observation)
        return landing_heuristic(env, brake_pressure_kpa=900.0,
                                 terminal_accel_m_s2=2.0)
