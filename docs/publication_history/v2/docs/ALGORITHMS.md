# Controller contracts

| Method | Public evaluation | Information and command contract |
|---|---|---|
| PID | NOT EVALUATED | Existing attitude PID has no equivalent XY tracking or yaw-rate command interface; not silently extended. |
| V2 | Three native enlarged trials | Original position outer loop; robot state; no velocity/acceleration FF. |
| V3 | Three native enlarged trials | V2 cascade with inherited reduced inner rate gain; robot state. |
| V2_FFv1 | Three native enlarged trials | Existing analytical p/v/a adapter, original V2 gains. |
| V3_FFv1 | Three native enlarged trials | Existing analytical p/v/a adapter, original V3 gains. |
| V4 | Three reused native enlarged trials | Frozen centroid/modal residual on V2_FFv1; privileged native liquid/controller state. |
| Standard PPO | Three reused native enlarged trials | Frozen 500k policy, 31 features -> three residual actions, privileged V4 backbone. |
| Teacher variants | NOT EVALUATED | Historical variants exist, but no equivalent frozen enlarged-task release contract was verified. |
| Gaussian Mixture PPO | NOT EVALUATED | Historical mixture-policy research is not a verified 30s enlarged-task release checkpoint. It is not relabeled Standard PPO. |
| Teleop_FF_Heading | Standard + extended command replay | New declared adapter of existing FF and heading P6, V3 inner controller; ideal robot state. |
| Teleop_FF_Heading_Modal | Gated diagnostic | Same adapter plus unchanged frozen centroid/modal feedback; privileged, experimental backbone transfer; one trial failed. |

All comparisons use native robot/liquid dynamics and original motor limits. Different information and reference interfaces prevent an algorithm-only causal interpretation. Early-terminated traces cannot be compared by total energy as if they survived the full task. Teleoperation does not substitute a command profile for the global figure-eight reference of Experiment 1.
