"""Train and evaluate vanilla PyTorch DQN for inlet-pressure control.

Only the Q network is trained. The PINN-derived nozzle map and flight physics
remain frozen. The target network, replay memory, and epsilon-greedy exploration
follow the standard DQN pattern; this is not Double or Dueling DQN.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dqn_env import ACTIONS_KPA, OBSERVATION_SIZE, NozzleFlightEnv, ballistic_apogee_estimate
from flight_model import DEFAULT_MAP, NozzleMap


RESULTS = Path(__file__).resolve().parent / "results"


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, len(ACTIONS_KPA)),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def add(self, observation, action, reward, next_observation, done):
        self.memory.append((observation.copy(), action, reward,
                            next_observation.copy(), done))

    def sample(self, batch_size: int, rng: random.Random):
        sample = rng.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*sample)
        return (
            torch.from_numpy(np.stack(observations)),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.from_numpy(np.stack(next_observations)),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.memory)


def greedy_action(network: QNetwork, observation: np.ndarray) -> int:
    with torch.no_grad():
        q = network(torch.from_numpy(observation).unsqueeze(0))
    return int(q.argmax(dim=1).item())


def optimize(network: QNetwork, target: QNetwork, optimizer,
             batch, gamma: float) -> float:
    observation, actions, rewards, next_observation, dones = batch
    q_selected = network(observation).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target(next_observation).max(dim=1).values
        expected = rewards + gamma * (1.0 - dones) * next_q
    loss = F.smooth_l1_loss(q_selected, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def train(seed: int, steps: int, map_path: Path, output_dir: Path) -> tuple[QNetwork, dict]:
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    nozzle = NozzleMap(map_path)
    env = NozzleFlightEnv(nozzle, seed=seed)
    network = QNetwork()
    target = QNetwork()
    target.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    buffer = ReplayBuffer(30_000)
    gamma = 0.99
    batch_size = 128
    warmup = 1000
    episode_rows = []
    observation, _ = env.reset()
    episode_return = 0.0
    recent_losses = deque(maxlen=100)
    for index in range(steps):
        epsilon = max(0.05, 1.0 - 0.95 * index / max(1, int(0.8 * steps)))
        action = rng.randrange(len(ACTIONS_KPA)) if rng.random() < epsilon else greedy_action(network, observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.add(observation, action, reward, next_observation, done)
        episode_return += reward
        observation = next_observation
        if len(buffer) >= warmup and index % 4 == 0:
            recent_losses.append(optimize(network, target, optimizer,
                                          buffer.sample(batch_size, rng), gamma))
        if (index + 1) % 500 == 0:
            target.load_state_dict(network.state_dict())
        if done:
            episode_rows.append({
                "episode": len(episode_rows) + 1,
                "global_step": index + 1,
                "target_apogee_m": info["target_apogee_m"],
                "peak_altitude_m": info["peak_altitude_m"],
                "absolute_error_m": abs(info["apogee_error_m"]),
                "fuel_used_kg": info["fuel_used_kg"],
                "episode_return": episode_return,
                "command_count": info["command_count"],
                "terminal_reason": info["terminated_reason"],
                "epsilon": epsilon,
                "recent_loss": sum(recent_losses) / len(recent_losses) if recent_losses else None,
            })
            observation, _ = env.reset()
            episode_return = 0.0
        if (index + 1) % 10_000 == 0:
            recent = episode_rows[-30:]
            mean_error = sum(row["absolute_error_m"] for row in recent) / len(recent) if recent else float("nan")
            print(f"seed={seed} step={index+1}/{steps} episodes={len(episode_rows)} "
                  f"recent_abs_error_m={mean_error:.1f}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"dqn_seed{seed}.pt"
    torch.save({
        "network_state_dict": network.state_dict(),
        "seed": seed,
        "steps": steps,
        "nozzle_map_sha256": nozzle.sha256,
        "actions_kpa": ACTIONS_KPA,
        "observation_size": OBSERVATION_SIZE,
        "observation_schema": ["altitude", "radial_velocity", "tangential_velocity",
                               "remaining_fuel", "target_apogee", "ballistic_apogee",
                               "ambient_pressure", "time_fraction"],
        "algorithm": "vanilla DQN",
    }, checkpoint)
    episodes_path = output_dir / f"training_seed{seed}.csv"
    with episodes_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    return network, {
        "seed": seed, "steps": steps, "episodes": len(episode_rows),
        "checkpoint": str(checkpoint.resolve()),
        "training_log": str(episodes_path.resolve()),
    }


def run_policy(env: NozzleFlightEnv, policy, target_m: float,
               mass_kg: float) -> dict:
    observation, _ = env.reset(target_apogee_m=target_m,
                               initial_mass_kg=mass_kg)
    total_reward = 0.0
    for _ in range(500):
        action = policy(env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return {
                "target_apogee_m": target_m,
                "initial_mass_kg": mass_kg,
                "peak_altitude_m": info["peak_altitude_m"],
                "absolute_error_m": abs(info["apogee_error_m"]),
                "fuel_used_kg": info["fuel_used_kg"],
                "episode_return": total_reward,
                "terminal_reason": info["terminated_reason"],
                "command_count": info["command_count"],
            }
    raise RuntimeError("Episode exceeded evaluation guard")


def evaluate(network: QNetwork, nozzle: NozzleMap, seed: int,
             output_dir: Path) -> dict:
    scenarios = [(target, mass) for target in (1800.0, 2200.0, 2600.0)
                 for mass in (95.0, 100.0, 105.0)]
    policies = {
        "dqn": lambda env, obs: greedy_action(network, obs),
        "fixed_800": lambda env, obs: 3,
        "fixed_1000": lambda env, obs: 5,
        "ballistic_cutoff_1000": lambda env, obs: (
            0 if ballistic_apogee_estimate(env.state) >= env.mission.target_apogee_m else 5
        ),
    }
    rows = []
    for name, policy in policies.items():
        env = NozzleFlightEnv(nozzle, seed=seed + 10_000)
        for target_m, mass_kg in scenarios:
            rows.append({"policy": name, **run_policy(env, policy, target_m, mass_kg)})
    with (output_dir / f"evaluation_seed{seed}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name in policies:
        subset = [row for row in rows if row["policy"] == name]
        summary[name] = {
            "mean_absolute_apogee_error_m": sum(row["absolute_error_m"] for row in subset) / len(subset),
            "mean_fuel_used_kg": sum(row["fuel_used_kg"] for row in subset) / len(subset),
            "within_100m_count": sum(row["absolute_error_m"] <= 100.0 for row in subset),
            "scenarios": len(subset),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    network, run = train(args.seed, args.steps, args.map, args.output_dir)
    run["evaluation"] = evaluate(network, NozzleMap(args.map), args.seed,
                                  args.output_dir)
    run["torch_version"] = torch.__version__
    run["numpy_version"] = np.__version__
    run["nozzle_map_sha256"] = NozzleMap(args.map).sha256
    report_path = args.output_dir / f"dqn_report_seed{args.seed}.json"
    report_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
