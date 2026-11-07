"""Run the selected DQN and record its pressure commands through apogee."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from dqn_env import ACTIONS_KPA, NozzleFlightEnv, ballistic_apogee_estimate
from evaluate_dqn import load_network
from flight_model import DEFAULT_MAP, NozzleMap, atmosphere
from train_dqn import RESULTS, greedy_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=RESULTS / "final/selected_dqn.pt")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--target-m", type=float, default=2300.0)
    parser.add_argument("--initial-mass-kg", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=RESULTS / "dqn_demo.csv")
    args = parser.parse_args()
    torch.set_num_threads(1)
    nozzle = NozzleMap(args.map)
    policy = load_network(args.checkpoint, nozzle)
    env = NozzleFlightEnv(nozzle)
    observation, _ = env.reset(target_apogee_m=args.target_m,
                               initial_mass_kg=args.initial_mass_kg)
    rows = []
    for _ in range(500):
        action = greedy_action(policy, observation)
        ambient, _ = atmosphere(env.state.altitude_m)
        thrust, mdot = nozzle.evaluate(ACTIONS_KPA[action], ambient) if action else (0.0, 0.0)
        rows.append({
            "time_s": env.state.time_s,
            "altitude_m": max(0.0, env.state.altitude_m),
            "radial_velocity_m_s": env.state.radial_velocity_m_s,
            "tangential_velocity_m_s": env.state.tangential_velocity_m_s,
            "mass_kg": env.state.mass_kg,
            "ambient_pressure_pa": ambient,
            "ballistic_apogee_estimate_m": ballistic_apogee_estimate(env.state),
            "target_apogee_m": args.target_m,
            "action": action,
            "command_kpa": ACTIONS_KPA[action],
            "estimated_thrust_n": thrust,
            "estimated_mass_flow_kg_s": mdot,
        })
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    else:
        raise RuntimeError("DQN demo exceeded episode guard")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "provisional PINN-map DQN flight example",
        "checkpoint": str(args.checkpoint.resolve()),
        "nozzle_map_sha256": nozzle.sha256,
        "target_apogee_m": args.target_m,
        "peak_altitude_m": info["peak_altitude_m"],
        "absolute_error_m": abs(info["apogee_error_m"]),
        "fuel_used_kg": info["fuel_used_kg"],
        "terminal_reason": info["terminated_reason"],
        "commands": len(rows),
        "engine_off_commands": sum(row["action"] == 0 for row in rows),
        "trajectory_csv": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
