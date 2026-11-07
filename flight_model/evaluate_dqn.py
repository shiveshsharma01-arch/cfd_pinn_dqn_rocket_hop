"""Evaluate saved DQN checkpoints on scenarios reserved from model selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from dqn_env import NozzleFlightEnv, ballistic_apogee_estimate
from flight_model import DEFAULT_MAP, NozzleMap
from train_dqn import QNetwork, RESULTS, greedy_action, run_policy


FINAL_SCENARIOS = [(target, mass) for target in (1700.0, 2300.0, 2700.0)
                   for mass in (97.0, 103.0)]
CHECKPOINTS = {
    seed: RESULTS / f"time_obs_seed{seed}/dqn_seed{seed}.pt" for seed in (0, 1, 2)
}


def load_network(path: Path, nozzle: NozzleMap) -> QNetwork:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["nozzle_map_sha256"] != nozzle.sha256:
        raise ValueError("Checkpoint was trained with a different nozzle map")
    if saved.get("observation_schema", [None])[-1] != "time_fraction":
        raise ValueError("Checkpoint uses an obsolete observation schema")
    network = QNetwork()
    network.load_state_dict(saved["network_state_dict"])
    network.eval()
    return network


def main() -> None:
    torch.set_num_threads(1)
    nozzle = NozzleMap(DEFAULT_MAP)
    policies = {}
    for seed, path in CHECKPOINTS.items():
        network = load_network(path, nozzle)
        policies[f"dqn_seed{seed}"] = (
            lambda env, obs, network=network: greedy_action(network, obs)
        )
    policies["fixed_800"] = lambda env, obs: 3
    policies["ballistic_cutoff_1000"] = lambda env, obs: (
        0 if ballistic_apogee_estimate(env.state) >= env.mission.target_apogee_m else 5
    )
    rows = []
    for name, policy in policies.items():
        env = NozzleFlightEnv(nozzle, seed=50_000)
        for target, mass in FINAL_SCENARIOS:
            rows.append({"policy": name, **run_policy(env, policy, target, mass)})
    output_csv = RESULTS / "final_evaluation.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "provisional PINN-map comparison; target altitude objective",
        "selection_rule": "seed 2 selected by nine development scenarios before this evaluation",
        "nozzle_map_sha256": nozzle.sha256,
        "scenarios": FINAL_SCENARIOS,
        "policies": {},
    }
    for name in policies:
        subset = [row for row in rows if row["policy"] == name]
        summary["policies"][name] = {
            "mean_absolute_apogee_error_m": sum(row["absolute_error_m"] for row in subset) / len(subset),
            "mean_fuel_used_kg": sum(row["fuel_used_kg"] for row in subset) / len(subset),
            "within_100m_count": sum(row["absolute_error_m"] <= 100 for row in subset),
            "scenarios": len(subset),
        }
    output_json = RESULTS / "final_evaluation.json"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
