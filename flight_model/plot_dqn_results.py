"""Plot final controller comparison and selected DQN pressure-command trace."""

from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt

from train_dqn import RESULTS


def main() -> None:
    report = json.loads((RESULTS / "final_evaluation.json").read_text(encoding="utf-8"))
    names = ["DQN seed 0", "DQN seed 1", "DQN seed 2", "Fixed 800", "Cutoff rule"]
    keys = ["dqn_seed0", "dqn_seed1", "dqn_seed2", "fixed_800", "ballistic_cutoff_1000"]
    metrics = [report["policies"][key] for key in keys]
    with (RESULTS / "dqn_demo.csv").open(newline="", encoding="utf-8") as stream:
        trace = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = ["#6b9cc8", "#6b9cc8", "#1667a8", "#8a8a8a", "#bf741c"]
    axes[0, 0].bar(names, [m["mean_absolute_apogee_error_m"] for m in metrics], color=colors)
    axes[0, 0].set_ylabel("Mean absolute apogee error (m)")
    axes[0, 1].bar(names, [m["mean_fuel_used_kg"] for m in metrics], color=colors)
    axes[0, 1].set_ylabel("Mean fuel used (kg)")
    for ax in axes[0]:
        ax.tick_params(axis="x", labelrotation=22)
        ax.grid(axis="y", alpha=0.25)
    t = [float(row["time_s"]) for row in trace]
    axes[1, 0].plot(t, [float(row["altitude_m"]) for row in trace], color="#1667a8")
    axes[1, 0].axhline(float(trace[0]["target_apogee_m"]), ls="--", color="#bf741c", label="Target")
    axes[1, 0].set_ylabel("Altitude (m)")
    axes[1, 0].legend()
    axes[1, 1].step(t, [float(row["command_kpa"]) if row["command_kpa"] else 0.0
                         for row in trace], where="post", color="#1667a8")
    axes[1, 1].set_ylabel("Inlet pressure command (kPa)")
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)
    fig.suptitle("Provisional PINN-map DQN pressure control")
    output = RESULTS / "dqn_results.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
