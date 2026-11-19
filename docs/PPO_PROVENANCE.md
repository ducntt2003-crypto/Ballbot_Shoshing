# Frozen PPO provenance

Checkpoint: `pretrained/ppo_500k.zip`.
SHA-256: `651989a98dd3ab04294adddd4a0bc0fb63e09b96749e9a980c5041e51bbd5db7`.

Origin: `end_to_end_parallel_rl_20261116_v1/training_recovery1/checkpoint_500000.zip`, selected on two development contexts before three fresh held-out contexts. Internal timestep is 499,992; filename counts 500,000 executed budget transitions. Eight workers collect for ONE shared policy and ONE independent policy seed (2611001), not eight replications.

Actor and critic: separate 64/64 Tanh MLPs. Observation: 31 privileged features. Action: 3 diagonal-Gaussian residual components. Residual mapping: delta=0.25*clip(a,-1,1), compose with V4, then existing 0.03 normalized residual and -10 motor sign/scaling; no authority retuning. Registered PPO: learning rate 3e-4, gamma .99, GAE .95, clip .2, value coefficient .5, entropy coefficient 0, gradient cap .5, target KL .03, minibatch64, up to512 intervals/worker, up to5epochs. Four curriculum levels were used.

The inherited exponential reward uses position/velocity/balance/modal-energy/effort/action-variation costs; authoritative coefficients remain in `_vendor/ballbot_sloshing_research/results/large_figure8_rl_20261111_v1/reward_config.json`.

497,600 optimized transitions, 124 buffers, 612 recorded epoch counters. Previous audit independently reconstructed GAE (gap8.94e-8) and behavior log probabilities (gap1.43e-6). Actor/critic weights changed, but full-task outcome remained15/16. A pre-update I/O failure discarded2400 executed transitions; recovery reset workers and was not exact continuation. SB3 overwrote planned initial worker seeds; actual seeds2611001..8 are disclosed. These deviations limit strict preregistration claims.

This release reuses checkpoints/traces and does not train. No new policy was selected using release outcomes.
