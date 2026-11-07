# What the staged nozzle model does

## Data and split

`nozzle_pinn.py` reads the Fluent comma-separated exports by **column name**,
because Fluent wrote the columns in different orders. Every case has 30,892
matching cell centres. The network uses seven pressure cases for training and
650 kPa for development validation. The 650 kPa field never enters the training
loss or normalization statistics. A future independent case is still required
for a final unbiased test.

## Network inputs and outputs

The three inputs are axial position, radial position, and inlet total pressure.
Positions and pressure are divided by fixed scales before entering the network.
The four learned outputs are axial velocity, radial velocity, log static pressure,
and log static temperature. The log representation keeps pressure and temperature
positive after exponentiation. Density is then calculated as `p/(R*T)`, with
`R = 287.04 J/(kg*K)` for the air used in the CFD runs.

## Physics loss

PyTorch automatic differentiation computes spatial derivatives of the network
outputs. The physics loss penalizes the steady axisymmetric continuity residual

`d(rho*ux)/dx + d(rho*ur)/dr + rho*ur/r`.

The data loss measures differences from CFD fields. Training uses the sum of
data loss and a weighted continuity loss. This is a **partial** PINN: momentum,
energy, and SST transport equations are not yet enforced. It should not be
reported as a complete turbulent nozzle-flow solver.

## Training phases

1. **Activation screen:** train a Tanh network and a SiLU network using Adam
   and only the CFD data loss. Compare both on 650 kPa.
2. **Joint training:** continue the better activation with Adam and the
   continuity loss added.
3. **Optimizer refinement:** continue with LBFGS on a fixed batch. LBFGS
   reevaluates a closure, so its batch must stay fixed during each step.

The script saves a checkpoint for each phase. `best.pt` is the phase with the
lowest **combined normalized field MSE** at 650 kPa. This selection criterion
does not establish the best exit thrust prediction.

## First measured result

The 650 kPa validation field is very smooth between the 600 and 700 kPa CFD
fields. Direct linear interpolation gives roughly 1.30 m/s axial-velocity RMSE,
131 Pa pressure RMSE, and 0.28 K temperature RMSE. The best staged network by
combined field MSE gives roughly 40.2 m/s, 18.2 kPa, and 5.59 K respectively.
Therefore, the current neural surrogate is **not accepted for DQN training**.
The checkpoint is an experiment, not a finished propulsion model.

## What comes next

We should improve the neural architecture or physics formulation and compare it
again against interpolation. DQN can use the validated interpolation map while
the PINN remains a separate research experiment. Before a PINN drives DQN,
validate its exit-integrated mass flow, momentum, pressure, and thrust against
Fluent surface reports; cell-centre exports alone do not provide exact exit
face integration data.
