"""Plot the trajectory and engine history from a saved flight CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "results/example_flight.csv")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results/example_flight.png")
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Flight CSV has no rows")
    values = lambda key: [float(row[key]) for row in rows]
    t = values("time_s")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Time (s)")
    axes[0, 0].plot(t, values("altitude_m"), color="#1465ab")
    axes[0, 0].set_ylabel("Altitude (m)")
    axes[0, 1].plot(t, values("radial_velocity_m_s"), label="Vertical")
    axes[0, 1].plot(t, values("tangential_velocity_m_s"), label="Downrange")
    axes[0, 1].set_ylabel("Velocity (m/s)")
    axes[0, 1].legend()
    axes[1, 0].plot(t, values("mass_kg"), color="#386a39")
    axes[1, 0].set_ylabel("Vehicle mass (kg)")
    axes[1, 1].plot(t, values("thrust_n"), color="#a85116")
    axes[1, 1].set_ylabel("Thrust (N)")
    fig.suptitle("Provisional PINN-based flight demonstration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
