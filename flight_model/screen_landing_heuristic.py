"""Screen target-aware reference-controller offsets across landing scenarios."""

from __future__ import annotations

import csv
from pathlib import Path

from landing_env import LandingEnv, landing_heuristic
from train_landing_dqn import run_episode


def main() -> None:
    rows = []
    for offset in (-100.0, -50.0, 0.0, 25.0, 50.0, 75.0, 100.0):
        for target in (450.0, 500.0, 550.0):
            for mass in (97.0, 100.0, 103.0):
                env = LandingEnv()
                metrics, _ = run_episode(
                    env, lambda e, obs, offset=offset: landing_heuristic(e, offset),
                    target, mass,
                )
                rows.append({"cutoff_offset_m": offset, **metrics})
    path = Path(__file__).resolve().parent / "results/landing/reference_screen.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for offset in sorted({r["cutoff_offset_m"] for r in rows}):
        subset = [r for r in rows if r["cutoff_offset_m"] == offset]
        print(offset, "success", sum(r["success"] for r in subset), "/", len(subset),
              "mean height error %", round(sum(r["peak_height_error_percent"] for r in subset) / len(subset), 2),
              "mean touchdown % limit", round(sum(r["touchdown_speed_percent_of_limit"] or 0 for r in subset) / len(subset), 2))
    print(path)


if __name__ == "__main__":
    main()
