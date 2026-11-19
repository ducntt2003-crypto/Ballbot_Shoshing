# Reproducibility and execution boundaries

Exactly two main experiments are exposed. Registered protocols are in configs/protocol.json plus two versioned addenda. Scripts were frozen before their corresponding simulations. Command replays execute closed-loop control; they never apply archived motor torques. Default simulations refuse existing output folders. Rendering replays saved native states without integrating new physics.

Experiment1 reuses six full-horizon V4/PPO trajectories from end_to_end_parallel_rl_20261116_v1. Twelve new V2/V3/FF trials use the same reset and terrain seeds. Compare initial qpos/qvel, hfield and model physical arrays across conditions with scripts/verify_release.py. Full task always uses original metrics/16 windows, not an easier short-duration task flag. Baseline short failures remain failures.

Experiment2 starts with3s flat/benchmark/playground checks. The standard16s script and separate26s traversal use independent listed reset seeds, fixed terrain and identical high-level commands. The modal-feedback addendum preserves baseline adapters, makes one declared control addition and gates off remaining trials after a physical failure. No new gain/physics tuning is applied. Recorded commands are quantized to the next10ms policy boundary; inputs include requested time and actual application simulation time.

GLFW logs press/release (including focus-neutral behavior), wall elapsed time, simulation time and reset episode; native telemetry is flushed to CSV. Reset creates a new episode index rather than pretending continuation. The current standard replay format accepts the frozen event script. Convert an actual session with scripts/keyboard_log_to_script.py, selecting one episode explicitly; pause intervals are simulation-time events, not torque replay.

The original wrapper records rows after valid observed steps. Guarded native failures can occur2ms after the last stored row. Original failed trials are preserved and flagged in stability/comparison tables; their q peak is an observed lower bound. The diagnostic revision adds terminal.npz without changing equations. Never reinterpret a last-valid q value below8.4mm as proving no ROM guard.

Tested: existing dependency environment, editable install without dependency downloads, doctor, GUI smoke, native tests and registered evaluation commands. Fresh installation is a documented recipe, not a tested clean-OS claim. The C plugins rebuild per local .cache and link only MuJoCo3.3.2. Shared-object files/caches are excluded from the clean archive. Runtime dependencies are not vendored binaries.

To reproduce a trial, use a new output path and the explicit seeds in configs. Avoid rerunning the manager merely to regenerate plots. To verify delivered files, run scripts/verify_release.py; this is read-only and invokes no physics. Essential tests execute a small counted native-step budget. Historical tests requiring excluded original large datasets are retained under tests/historical but only self-contained subsets are run in the clean release.

No GitHub upload/push, hardware command, Chrono/CUDA modification, PPO retraining or unauthorized campaign is performed.
