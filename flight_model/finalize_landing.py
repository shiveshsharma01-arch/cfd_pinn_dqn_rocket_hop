"""Package the reproducible 500 m takeoff-and-landing coding result.

Evaluates the selected DQN alone, DQN ascent with guided descent, and a
timed-ascent reference with the same guided descent. The validated mission is
500 m with initial mass 99–101 kg; wider generalization is not claimed.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import torch

from evaluate_landing import CHECKPOINT_DIR, load_policy
from flight_model import DEFAULT_MAP, NozzleMap
from guided_landing import GuidedLandingPolicy
from landing_env import LandingConfig, LandingEnv, landing_heuristic
from train_dqn import greedy_action
from train_landing_dqn import ROOT, run_episode


FINAL_DIR = ROOT / "final"
SOURCE_CHECKPOINT = CHECKPOINT_DIR / "step120000.pt"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_policy(env: LandingEnv, _observation) -> int:
    return landing_heuristic(env, brake_pressure_kpa=900.0,
                             terminal_accel_m_s2=2.0)


def summarize(rows: list[dict]) -> dict:
    speeds = [r["touchdown_speed_percent_of_limit"] for r in rows
              if r["touchdown_speed_percent_of_limit"] is not None]
    return {
        "cases": len(rows),
        "success_percent": 100.0 * sum(r["success"] for r in rows) / len(rows),
        "mean_peak_height_error_percent": sum(
            r["peak_height_error_percent"] for r in rows) / len(rows),
        "maximum_peak_height_error_percent": max(
            r["peak_height_error_percent"] for r in rows),
        "touchdowns": len(speeds),
        "mean_touchdown_speed_percent_of_5_mps_limit": (
            sum(speeds) / len(speeds) if speeds else None),
        "maximum_touchdown_speed_percent_of_5_mps_limit": (
            max(speeds) if speeds else None),
        "minimum_fuel_remaining_kg": min(r["fuel_remaining_kg"] for r in rows),
    }


def main() -> None:
    torch.set_num_threads(1)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    nozzle = NozzleMap(DEFAULT_MAP)
    network = load_policy(SOURCE_CHECKPOINT, nozzle)
    policies = {
        "dqn_only": lambda: lambda env, obs: greedy_action(network, obs),
        "dqn_ascent_guided_descent": lambda: GuidedLandingPolicy(network),
        "timed_ascent_guided_descent": lambda: reference_policy,
    }
    masses = [round(99.0 + 0.1 * i, 1) for i in range(21)]
    rows = []
    for name, factory in policies.items():
        for mass in masses:
            metrics, trace = run_episode(
                LandingEnv(nozzle), factory(), 500.0, mass,
                record=(mass == 100.0),
            )
            rows.append({"policy": name, **metrics})
            if trace:
                save_csv(FINAL_DIR / f"{name}_nominal_trace.csv", trace)
    save_csv(FINAL_DIR / "validation_cases.csv", rows)
    summary = {name: summarize([r for r in rows if r["policy"] == name])
               for name in policies}
    summary["evaluation_scope"] = {
        "target_height_m": 500.0,
        "height_tolerance_percent": 10.0,
        "touchdown_speed_limit_m_s": 5.0,
        "minimum_fuel_remaining_kg": 5.0,
        "initial_mass_range_kg": [99.0, 101.0],
        "initial_mass_increment_kg": 0.1,
        "note": "Deterministic validation sweep; not unseen CFD or flight-test data.",
    }
    (FINAL_DIR / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    shutil.copy2(SOURCE_CHECKPOINT, FINAL_DIR / "selected_dqn.pt")
    metadata = {
        "checkpoint": "selected_dqn.pt",
        "source_checkpoint": SOURCE_CHECKPOINT.relative_to(PROJECT_ROOT).as_posix(),
        "nozzle_map": DEFAULT_MAP.relative_to(PROJECT_ROOT).as_posix(),
        "nozzle_map_sha256": nozzle.sha256,
        "training": "120,000-step DQN, one seed, reference replay seeding "
                    "and behavior-cloning auxiliary loss",
        "recommended_policy": "DQN ascent + guided descent",
        "descent_rule": "900 kPa stopping-distance rule; below 20 m, track "
                        "1+sqrt(4h) m/s downward-speed envelope",
        "actuator_time_constant_s": LandingConfig().actuator_time_constant_s,
        "control_interval_s": LandingConfig().control_interval_s,
        "integration_step_s": LandingConfig().integration_step_s,
        "limitation": "DQN alone is evaluated separately and does not meet the "
                      "landing criteria. CFD nozzle map and point-mass flight "
                      "model are educational approximations.",
    }
    (FINAL_DIR / "policy_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
