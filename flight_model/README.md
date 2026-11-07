# Flight model and 500 m landing demonstrator

This Python point-mass model links the frozen PINN-derived nozzle performance
table to a two-dimensional trajectory. It tracks radius/altitude, downrange
angle, radial and tangential speed, and vehicle mass. Forces are gravity,
thrust, and drag. The controller-facing input is an inlet-pressure command
between 600 and 1000 kPa, or `None` for engine off. A separate pitch schedule
sets thrust direction; the future DQN will not control attitude in this version.

Run from the `Intern` folder:

```powershell
python flight_model/run_flight.py
python flight_model/plot_flight.py
python -m unittest discover -s flight_model -p 'test_*.py' -v
```

The runner writes `results/example_flight.csv` and a JSON summary. The plotter
writes `results/example_flight.png`. You can
change `--pressure-kpa`, `--wet-mass-kg`, `--dry-mass-kg`, `--dt-s`, and other
parameters. `--map` accepts a replacement table with the same columns after
Fluent outlet thrust has been checked. The JSON records the input map's SHA-256.

The map contains vacuum thrust and mass flow at nine pressure settings. The
model linearly interpolates within the map and calculates
`F = F_vacuum - p_ambient * A_exit` at each step. This assumes ambient pressure
does not alter internal nozzle flow. It may fail for separated or shocked
regimes. No extrapolation below 600 or above 1000 kPa is permitted; engine off
is a separate command.

The atmosphere is an isothermal exponential approximation, and Earth is
nonrotating. The example uses a 100 kg initial mass, 50 kg dry mass, 0.01 m²
frontal area, drag coefficient 0.5, and an illustrative pitch programme. These
vehicle values are explicit **demonstration assumptions**, not values of a
real rocket. The model supports radial and tangential orbital dynamics, but
the current 300 K air nozzle and example vehicle do not achieve orbit. The
flight trajectory is provisional because the PINN thrust map has not yet been
compared with Fluent outlet-face thrust results.

With the example inputs and a 0.1 s step, the current map gives a peak altitude
of about 2.94 km and a ground return at about 58.26 s. A 0.05 s step gives a
peak altitude within 0.001 m. These are numerical checks of the demonstrator,
not validation against real flight data.

The first integrated apogee DQN controller is implemented in `dqn_env.py` and
`train_dqn.py`. Its selected checkpoint, full results, controls, and limits
are in `DQN_RESULTS.md`. That historical apogee controller has instantaneous
pressure commands. The newer 500 m landing environment in `landing_env.py`
includes a simplified first-order actuator response. The Fluent outlet audit
is in `CFD_OUTLET_AUDIT.md`.

## Reproduce the takeoff-to-touchdown result

Run from `flight_model`:

```powershell
python finalize_landing.py
python -m unittest test_landing.py test_flight_model.py test_dqn_env.py
```

This run used Python 3.13.15, NumPy 2.4.4 and PyTorch 2.6.0. Package
requirements are in `requirements.txt`.

`finalize_landing.py` compares DQN alone, DQN ascent with guided descent, and
timed ascent with the same guided descent for a 500 m vertical hop. It uses
the frozen checkpoint and writes full case metrics and animation-ready
nominal traces to `results/landing/final/`. The completed result, percentages,
and limits are in `PROJECT_STATUS.md`. The selected DQN alone does **not** meet
the landing target; the recommended policy is a hybrid. This difference should
remain visible in the report and animation.
