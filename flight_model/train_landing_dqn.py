"""Train demonstration-seeded DQN for a complete 500 m vertical hop and landing.

The nozzle map and flight physics remain frozen. Early experience includes a
hand-coded reference controller; later actions are selected by epsilon-greedy
DQN alone. Intermediate checkpoints and identical-scenario traces are saved
for honest learning-progress animation.
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

from dqn_env import ACTIONS_KPA
from flight_model import DEFAULT_MAP, NozzleMap
from landing_env import LandingEnv, OBSERVATION_SIZE, landing_heuristic
from train_dqn import ReplayBuffer, greedy_action


ROOT = Path(__file__).resolve().parent / "results/landing"
OBSERVATION_SCHEMA = (
    "height", "vertical_speed", "remaining_fuel", "peak_height", "target_height",
    "time_fraction", "ambient_pressure", "actuator_spool", "active_pressure",
    "stopping_distance", "ballistic_apogee",
)


class LandingQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, len(ACTIONS_KPA)),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


def episode_metrics(info: dict, episode_return: float, steps: int) -> dict:
    return {
        "target_height_m": info["target_height_m"],
        "peak_height_m": info["peak_height_m"],
        "peak_height_error_percent": info["peak_height_error_percent"],
        "touchdown_speed_percent_of_limit": info["touchdown_speed_percent_of_limit"],
        "fuel_remaining_kg": info["fuel_remaining_kg"],
        "success": info["success"],
        "terminal_reason": info["terminal_reason"],
        "episode_return": episode_return,
        "control_steps": steps,
    }


def run_episode(env: LandingEnv, policy, target_m: float = 500.0,
                initial_mass_kg: float = 100.0,
                record: bool = False) -> tuple[dict, list[dict]]:
    observation, _ = env.reset(target_height_m=target_m,
                               initial_mass_kg=initial_mass_kg)
    trace = []
    total_reward = 0.0
    for count in range(1500):
        action = policy(env, observation)
        if record:
            trace.append({
                "time_s": env.state.time_s,
                "height_m": max(0.0, env.state.altitude_m),
                "vertical_speed_m_s": env.state.radial_velocity_m_s,
                "peak_height_m": env.peak_height_m,
                "target_height_m": target_m,
                "mass_kg": env.state.mass_kg,
                "fuel_remaining_kg": env.state.mass_kg - env.vehicle.dry_mass_kg,
                "actuator_spool": env.spool,
                "action_index": action,
                "command_kpa": ACTIONS_KPA[action],
            })
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            metrics = episode_metrics(info, total_reward, count + 1)
            metrics["initial_mass_kg"] = initial_mass_kg
            if record:
                trace.append({
                    "time_s": env.state.time_s,
                    "height_m": max(0.0, env.state.altitude_m),
                    "vertical_speed_m_s": env.state.radial_velocity_m_s,
                    "peak_height_m": env.peak_height_m,
                    "target_height_m": target_m,
                    "mass_kg": env.state.mass_kg,
                    "fuel_remaining_kg": env.state.mass_kg - env.vehicle.dry_mass_kg,
                    "actuator_spool": env.spool,
                    "action_index": None,
                    "command_kpa": None,
                })
            return metrics, trace
    raise RuntimeError("Landing episode exceeded guard")


def _save_checkpoint(network: LandingQNetwork, nozzle: NozzleMap, path: Path,
                     seed: int, global_step: int) -> None:
    torch.save({
        "state_dict": network.state_dict(),
        "seed": seed,
        "global_step": global_step,
        "nozzle_map_sha256": nozzle.sha256,
        "observation_schema": OBSERVATION_SCHEMA,
        "actions_kpa": ACTIONS_KPA,
        "algorithm": "DQN with reference-policy replay seeding and behavior-cloning auxiliary loss",
    }, path)


def _save_trace(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scenario(rng: random.Random, progress: float) -> tuple[float, float]:
    if progress < 0.50:
        return 500.0, 100.0
    if progress < 0.80:
        return rng.uniform(495.0, 505.0), rng.uniform(99.5, 100.5)
    return rng.uniform(480.0, 520.0), rng.uniform(99.0, 101.0)


def optimize_landing(network: LandingQNetwork, target: LandingQNetwork,
                     optimizer, experience, demonstrations,
                     class_weights: torch.Tensor, gamma: float = 0.999) -> float:
    obs, action, reward, next_obs, done = experience
    chosen = network(obs).gather(1, action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        expected = reward + gamma * (1.0 - done) * target(next_obs).max(1).values
    bellman = F.smooth_l1_loss(chosen, expected)
    expert_obs, expert_action = demonstrations[0], demonstrations[1]
    imitation = F.cross_entropy(network(expert_obs), expert_action,
                                weight=class_weights)
    loss = bellman + 0.5 * imitation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def train(seed: int, steps: int, output_dir: Path,
          map_path: Path = DEFAULT_MAP) -> dict:
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    nozzle = NozzleMap(map_path)
    env = LandingEnv(nozzle, seed=seed)
    network = LandingQNetwork()
    target = LandingQNetwork()
    target.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    buffer = ReplayBuffer(100_000)
    expert_buffer = ReplayBuffer(30_000)
    gamma = 0.999
    output_dir.mkdir(parents=True, exist_ok=True)

    # Replay seeding provides successful touchdown examples before exploration.
    demonstration_episodes = 30
    demonstration_outcomes = []
    for _ in range(demonstration_episodes):
        mass = rng.uniform(99.0, 101.0)
        observation, _ = env.reset(target_height_m=500.0,
                                   initial_mass_kg=mass)
        for _ in range(1500):
            action = landing_heuristic(env)
            next_observation, reward, terminated, truncated, info = env.step(action)
            buffer.add(observation, action, reward, next_observation,
                       terminated or truncated)
            expert_buffer.add(observation, action, reward, next_observation,
                              terminated or truncated)
            observation = next_observation
            if terminated or truncated:
                demonstration_outcomes.append(info["success"])
                break

    counts = np.bincount([item[1] for item in expert_buffer.memory],
                         minlength=len(ACTIONS_KPA))
    class_weights = torch.as_tensor(
        1.0 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32
    )
    class_weights /= class_weights[class_weights > 0].mean()
    for _ in range(1500):
        expert_obs, expert_action, *_ = expert_buffer.sample(256, rng)
        imitation_loss = F.cross_entropy(network(expert_obs), expert_action,
                                         weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        imitation_loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 10.0)
        optimizer.step()
    target.load_state_dict(network.state_dict())

    observation, _ = env.reset(target_height_m=500.0, initial_mass_kg=100.0)
    episode_return = 0.0
    episode_steps = 0
    episode_expert = True
    episode_rows = []
    last_losses = deque(maxlen=200)
    milestones = {0, steps // 4, steps // 2, 3 * steps // 4, steps}

    def save_milestone(index: int):
        name = f"step{index:06d}"
        _save_checkpoint(network, nozzle, output_dir / f"{name}.pt", seed, index)
        rollout_env = LandingEnv(nozzle, seed=seed + 1000)
        metrics, trace = run_episode(
            rollout_env, lambda e, obs: greedy_action(network, obs), record=True
        )
        _save_trace(output_dir / f"{name}_trace.csv", trace)
        (output_dir / f"{name}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

    save_milestone(0)
    for index in range(steps):
        progress = index / steps
        epsilon = max(0.05, 0.7 - 0.65 * index / max(1, int(0.85 * steps)))
        if episode_expert:
            action = landing_heuristic(env)
        elif rng.random() < epsilon:
            action = rng.randrange(len(ACTIONS_KPA))
        else:
            action = greedy_action(network, observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        buffer.add(observation, action, reward, next_observation,
                   terminated or truncated)
        observation = next_observation
        episode_return += reward
        episode_steps += 1
        if index % 2 == 0:
            ordinary = buffer.sample(96, rng)
            demonstrations = expert_buffer.sample(32, rng)
            batch = tuple(torch.cat((a, b), dim=0) for a, b in zip(ordinary, demonstrations))
            last_losses.append(optimize_landing(
                network, target, optimizer, batch, demonstrations,
                class_weights, gamma,
            ))
        if (index + 1) % 1000 == 0:
            target.load_state_dict(network.state_dict())
        if terminated or truncated:
            row = episode_metrics(info, episode_return, episode_steps)
            row.update({"episode": len(episode_rows) + 1, "global_step": index + 1,
                        "initial_mass_kg": env.vehicle.initial_mass_kg,
                        "epsilon": epsilon})
            episode_rows.append(row)
            target_height, mass = _scenario(rng, progress)
            observation, _ = env.reset(target_height_m=target_height,
                                       initial_mass_kg=mass)
            episode_expert = (progress < 0.65 and rng.random() < 0.35)
            episode_return = 0.0
            episode_steps = 0
        if index + 1 in milestones:
            save_milestone(index + 1)
        if (index + 1) % 20_000 == 0:
            recent = episode_rows[-20:]
            success_rate = sum(row["success"] for row in recent) / len(recent) if recent else 0.0
            print(f"seed={seed} step={index+1}/{steps} episodes={len(episode_rows)} "
                  f"recent_success={success_rate:.1%}", flush=True)
    logs_path = output_dir / "training_episodes.csv"
    with logs_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    result = {
        "seed": seed,
        "steps": steps,
        "training_episodes": len(episode_rows),
        "demonstration_episodes": demonstration_episodes,
        "demonstration_success_percent": 100 * sum(demonstration_outcomes) / len(demonstration_outcomes),
        "nozzle_map_sha256": nozzle.sha256,
        "final_checkpoint": str((output_dir / f"step{steps:06d}.pt").resolve()),
        "training_log": str(logs_path.resolve()),
        "recent_success_percent": 100 * sum(row["success"] for row in episode_rows[-20:])
        / max(1, len(episode_rows[-20:])),
        "recent_mean_loss": sum(last_losses) / max(1, len(last_losses)),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    (output_dir / "report.json").write_text(json.dumps(result, indent=2),
                                              encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or ROOT / f"seed{args.seed}"
    print(json.dumps(train(args.seed, args.steps, output, args.map), indent=2))


if __name__ == "__main__":
    main()
