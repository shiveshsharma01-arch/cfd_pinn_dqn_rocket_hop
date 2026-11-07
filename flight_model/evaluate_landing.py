"""Compare pure DQN, DQN ascent + guided descent, and a reference controller."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from flight_model import DEFAULT_MAP, NozzleMap
from landing_env import LandingEnv, landing_heuristic
from train_dqn import greedy_action
from train_landing_dqn import LandingQNetwork, ROOT, run_episode


OUTPUT = ROOT / "comparison"
CHECKPOINT_DIR = ROOT / "seed0_v3"


def load_policy(path: Path, nozzle: NozzleMap) -> LandingQNetwork:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["nozzle_map_sha256"] != nozzle.sha256:
        raise ValueError("Nozzle map changed since controller training")
    network = LandingQNetwork()
    network.load_state_dict(saved["state_dict"])
    network.eval()
    return network


def evaluate_policy(nozzle: NozzleMap, policy, scenarios,
                    name: str) -> list[dict]:
    rows = []
    for target, mass in scenarios:
        env = LandingEnv(nozzle)
        metrics, trace = run_episode(env, policy, target, mass,
                                     record=(target == 500 and mass == 100))
        rows.append({"policy": name, **metrics})
        if trace:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            path = OUTPUT / f"{name}_trace.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
                writer.writeheader()
                writer.writerows(trace)
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "success_percent": 100 * sum(row["success"] for row in rows) / len(rows),
        "mean_peak_height_error_percent": sum(row["peak_height_error_percent"] for row in rows) / len(rows),
        "mean_touchdown_speed_percent_of_limit": sum(
            row["touchdown_speed_percent_of_limit"] or 0 for row in rows
        ) / len(rows),
        "mean_fuel_remaining_percent_of_initial_supply": sum(
            100 * row["fuel_remaining_kg"] / (row["initial_mass_kg"] - 50)
            for row in rows
        ) / len(rows),
        "scenarios": len(rows),
    }


def main() -> None:
    torch.set_num_threads(1)
    nozzle = NozzleMap(DEFAULT_MAP)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenarios = [(500.0, mass) for mass in (99.0, 100.0, 101.0)]
    rows = evaluate_policy(nozzle, lambda env, obs: landing_heuristic(env),
                           scenarios, "reference")
    for step_number in (0, 30_000, 60_000, 90_000, 120_000):
        network = load_policy(CHECKPOINT_DIR / f"step{step_number:06d}.pt", nozzle)
        for kind in ("pure", "hybrid"):
            name = f"{kind}_step{step_number:06d}"
            if kind == "pure":
                policy = lambda env, obs, network=network: greedy_action(network, obs)
            else:
                policy = lambda env, obs, network=network: (
                    landing_heuristic(env) if env.state.radial_velocity_m_s < 0
                    else greedy_action(network, obs)
                )
            rows.extend(evaluate_policy(nozzle, policy, scenarios, name))
    output_csv = OUTPUT / "development_comparison.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {name: summarize([r for r in rows if r["policy"] == name])
               for name in dict.fromkeys(r["policy"] for r in rows)}
    (OUTPUT / "development_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
