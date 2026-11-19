# Algorithms, implementation and comparability

PID, cascade control, analytic feedforward, linear pole placement, receding-horizon optimization, PPO and Gaussian mixtures are inherited scientific methods. The project contributions are their BallBot–liquid integration, controller-state/contact audits, variants, guards and evaluation workflow.

## PID and cascades

PID is inherited attitude-only control and has no frozen matching XY/yaw interface here. [V2](../_vendor/ballbot_fast_rl/robust_control/v2/variant.py) constrains desired tilt/slew and uses conditional integral updates. [V3](../_vendor/ballbot_fast_rl/robust_control/v3/variant.py) changes the inner bandwidth/rate gain. [FeedforwardMixin](../_vendor/ballbot_sloshing_research/track8/controller.py) adds a_ref/kp to the inspected outer-loop proportional error and v_ref to the derivative reference, while integrating real position error. This preserves the original gains and limits.

## V4

[CentroidModalFeedback](../_vendor/ballbot_sloshing_research/track8_c/centroid_controller.py) uses a frozen identified augmented model, nominal reference state and modal feedback matrix. Only selected growing fluid-dominant modes were targeted; a motor headroom factor bounds the correction. It retains V2_FFv1, native liquid dynamics and physical damping. This is local model-based design with privileged liquid/controller states, not a global nonlinear stability proof.

## Teachers and PPO

Historical teacher studies explore preview/receding-horizon tracking/energy constraints. A verified equivalent 30 s enlarged-task release teacher was not available, so no teacher performance number is inserted in the main table. Standard PPO uses the released V4 residual contract; see [RL details](REINFORCEMENT_LEARNING.md).

Historical TRACK8-Y implements a stateless seven-component Gaussian mixture: weights [.7,.05,.05,.05,.05,.05,.05], central mean and ±.5 offsets along each of three action axes, marginalized log-probability. This changes variance and multimodality together. The matched pilot uses two training seeds, 16,384 transitions per arm/seed; these short historical DEV evaluations are **not** enlarged-task full-horizon results. The recorded decision is STRUCTURED_PPO_INCONCLUSIVE, not a general mixture improvement. [Archived pilot evidence](../research/mixture_history/engineering_decision.json), [distribution implementation](../research/mixture_history/distribution.py), [configuration](../research/mixture_history/exploration_config.json), [historical summary](../research/mixture_history/mixture.csv).

## Comparison contract

V2, V3, V2_FFv1, V3_FFv1, V4 and Standard PPO have three matched enlarged-task runs each. Only V3_FFv1/V4/PPO survive all full horizons; V3_FFv1 fails yaw safety. V4/PPO have privileged information; others do not. Early termination, differing interfaces and local modeling prevent algorithm-only causal claims. Teleop_FF_Heading is a distinct bounded-command adapter and not a substitute figure-eight reference. Historical methods/checkpoints are never relabeled as new task results.
