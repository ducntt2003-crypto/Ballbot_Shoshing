# Reinforcement learning: actual released lineage

## Architecture and observations

[PPO environment](../_vendor/ballbot_sloshing_research/large_figure8_rl/env.py), [base frame](../_vendor/ballbot_sloshing_research/phase2/env.py), [frozen checkpoint](../pretrained/ppo_500k.zip).

Independent actor/critic MLP branches: Linear(31,64), Tanh, Linear(64,64), Tanh; actor outputs three Gaussian means and learned log std, critic one value. No CNN. Inputs float32, clipped to [-10,10], no VecNormalize in this path:

| Features | Count | Scaling |
|---|---:|---|
| Body gravity direction | 3 | XY divided by .05 |
| Body angular velocity | 3 | /.3 |
| XY position error | 2 | /.02 |
| XY velocity error | 2 | /.05 |
| Wheel speed | 3 | /2 |
| Previous applied normalized action | 3 | /.1 |
| Reference XY velocity | 2 | /.05 |
| Base action | 3 | /.1 |
| sin/cos measured yaw | 2 | unit |
| Liquid displacement | 2 | /.0084 |
| Liquid velocity | 2 | /(omega*.0084) |
| Reference acceleration | 2 | /[.5*w²,w²], w=2pi/30 |
| sin/cos reference phase | 2 | unit |

The 31-vector includes privileged modal information. The exact acceleration scaling is inherited and is not replaced with amplitude-normalized alternatives.

## Action and reward

For raw action a, delta=.25*clip(a,-1,1), composed=clip(V4+delta,-1,1), torque=-10*clip(base+.03*composed,-1,1) Nm. Actuator/native guards remain unchanged.

[Authoritative reward configuration](../_vendor/ballbot_sloshing_research/results/large_figure8_rl_20261111_v1/reward_config.json): C=sum(w_i c_i), where position/velocity/balance/effort/variation are normalized quadratics and slosh=E/E0. Weights are 1,.1,.2,.05,.01,.001 for position, velocity, balance, slosh, effort, variation. Scales are .1 m, .1480960979386122 m/s, .12 rad, .0002728148812576366 J, 10 Nm and .25 residual. Reward per policy interval sums .002*exp(-C) across native samples. Invalid termination returns -5.

$$\delta_t^V=r_t+\gamma V(s_{t+1})-V(s_t),\quad \hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}^V.$$

PPO clips likelihood ratio to [1-.2,1+.2]. Bootstrap/reset masks follow the trainer/rollout implementation; truncation and invalid termination are not interchangeable. gamma=.99, lambda=.95, learning rate3e-4, value coefficient.5, entropy0, gradient norm.5, target KL.03, minibatch64, up to512 intervals/worker and5epochs.

## Parallelism, selection and limitations

Eight independent MuJoCo worker processes produce data for **one policy** seed2611001, actual worker seeds2611001..8. Four curriculum levels were used. 500,000 executed transitions; 497,600 optimized,124 buffers,612 epoch counters; internal checkpoint timestep499,992. An I/O interruption discarded2400 pre-update samples, then workers reset. This is not exact continuation. Checkpoint selection used two development contexts; three fresh held-out contexts were not selection data.

[Detailed provenance](PPO_PROVENANCE.md), [original progress](../research/training_500k/training/logs/progress.csv), [recovered progress](../research/training_500k/training_recovery1/logs/progress.csv), [recovered transition diagnostics](../research/training_500k/training_recovery1/transitions.csv.gz). Progress contains sparse rows and logger restarts; blank cells are not interpolated as measurements. Learning plots show observed data and the recovery boundary, not multiple-seed confidence bands. No training was launched for this README revision.

Historical mixture-policy evidence is distinct from this Standard PPO lineage; see [algorithms](ALGORITHMS.md).
