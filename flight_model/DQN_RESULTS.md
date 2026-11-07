# DQN pressure controller — first integrated result

## What is connected

At every 0.5 s decision point, DQN chooses **engine off** or an inlet total
pressure of **600, 700, 800, 900, or 1000 kPa**. The frozen PINN-derived
nozzle table supplies mass flow and vacuum thrust. The flight model subtracts
ambient-pressure force (`p_ambient * exit_area`), advances vehicle position,
speed, and mass in 0.1 s integration steps, then returns the next observation
and reward. Pitch is prescribed separately; DQN does not steer the vehicle.

The first mission is to reach a requested **apogee** with low altitude error
and modest fuel use. Episodes end at apogee, ground impact, or 90 s. The
network observes altitude, radial and tangential speed, remaining fuel,
requested apogee, a drag-free ballistic-apogee estimate, ambient pressure, and
elapsed-time fraction. Time is included because the prescribed pitch changes
with time. The training target is randomized from 1600–2800 m and initial mass
from 95–105 kg; the dry mass is 50 kg.

Reward uses potential-difference shaping for predicted apogee, a penalty of
`0.015 * fuel_used_kg` per control step, and a terminal altitude score
`8 exp(-abs(error_m)/300) - 0.003 abs(error_m)`. Failure to leave the ground
adds a further `-3`. These are design choices, not physical laws.

## Training

`train_dqn.py` implements **vanilla DQN** in PyTorch: two 64-unit ReLU hidden
layers, epsilon-greedy exploration, 30,000-transition replay memory, 128-sample
minibatches, Adam at `3e-4`, Huber loss, discount `0.99`, and a target-network
copy every 500 steps. Each of seeds **0, 1, and 2** trained for **100,000
environment steps**. The nozzle PINN checkpoint and its thrust table were not
trained or modified during DQN training. The current map SHA-256 is recorded
in every DQN checkpoint.

The selected model is `results/final/selected_dqn.pt`, a byte-identical copy
of `results/time_obs_seed2/dqn_seed2.pt`. Earlier
30,000-step and no-time-observation experiments remain in `results/` and are
not the current controller.

## Measured controller comparison

The nine development scenarios used targets 1800, 2200 and 2600 m and masses
95, 100 and 105 kg. **Seed 2** was selected using these scenarios. The separate
six-scenario check used targets 1700, 2300 and 2700 m and masses 97 and 103 kg.
The table below reports results on those six scenarios.

| Controller | Mean absolute apogee error | Mean absolute percentage error | Mean fuel used | Within 100 m |
|---|---:|---:|---:|---:|
| DQN seed 0 | 348.7 m | 15.90% | 48.8 kg | 0/6 |
| DQN seed 1 | 422.6 m | 21.97% | 50.0 kg | 2/6 |
| **DQN seed 2, selected** | **117.8 m** | **5.26%** | **47.6 kg** | **3/6** |
| Fixed 800 kPa | 693.5 m | 35.94% | 50.0 kg | 1/6 |
| Ballistic-apogee cutoff at 1000 kPa | 114.7 m | 5.33% | 41.1 kg | 3/6 |

Thus the selected DQN tracks altitude much better than fixed pressure. The
simple cutoff rule is slightly better in absolute metres (114.7 versus
117.8 m), while DQN is marginally better by mean percentage error (5.26%
versus 5.33%). The cutoff rule also uses about **6.5 kg less fuel** on average.
Seed variability is substantial. These results do not support a claim that
DQN is the best controller.

The selected DQN's mean fuel use is **95.18% of the 50 kg nominal available
propellant** in these six scenarios. This policy cannot simply be reused for
a powered landing. Its environment also accepts target heights only from
1000–4000 m, trains at 1600–2800 m, and terminates at apogee.

In the saved 2300 m demonstration, the selected DQN reaches 2405.9 m, an
error of 105.9 m, and uses 50 kg of fuel. The command trace shows that it can
switch the engine off and back on, but it does not consistently conserve fuel.

## Reproduce

Run from the `Intern` workspace folder:

```powershell
python flight_model/train_dqn.py --seed 2 --steps 100000 --output-dir flight_model/results/reproduction_seed2
python flight_model/evaluate_dqn.py
python flight_model/run_dqn_demo.py
python flight_model/plot_dqn_results.py
python -m unittest discover -s flight_model -p 'test_*.py' -v
```

The full evaluation is in `results/final_evaluation.csv` and
`results/final_evaluation.json`; training logs and all checkpoints are under
`results/`. `results/dqn_demo.csv` records commands and flight states, and
`results/dqn_results.png` shows the comparison and example trajectory.

## Limits and next checks

- The nozzle map is PINN-derived and provisional. The supplied Fluent mass
  reports have 1.07–3.70% inlet/outlet imbalance, and the 750 kPa outlet
  profile lacks flow fields. See `CFD_OUTLET_AUDIT.md`. None of these CFD
  exports were silently substituted into the flight map.
- The current map assumes ambient pressure does not alter internal flow. The
  atmosphere is isothermal; vehicle mass and drag values are illustrative;
  pressure changes and shutdown are instantaneous; attitude follows a fixed
  pitch schedule. A real pressure actuator and sensor noise are not modelled.
- This is a suborbital apogee-control demonstration. The 300 K air nozzle and
  example vehicle cannot represent an orbital launch system. Orbital insertion
  would require a physically appropriate propulsion model and new mission
  design, not a relabelled target altitude.
- Before any performance claim, compare nozzle thrust against Fluent outlet
  face integrations, improve CFD conservation, and evaluate robustness to
  thrust-map error, mass uncertainty, and actuator lag. The cutoff rule is a
  necessary benchmark for further DQN improvement.

Algorithm reference: [official PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning).
