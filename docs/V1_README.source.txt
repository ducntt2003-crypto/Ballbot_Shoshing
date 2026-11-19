# SloshBot-RL

**Learning to Balance, Track Trajectories, and Carry Liquids on Rough Terrain**

![Native PPO figure-eight replay](assets/gifs/experiment_1_figure8.gif)

Two reproducible MuJoCo experiments investigate a liquid-carrying BallBot: autonomous figure-eight controller comparison and keyboard-driven navigation. The liquid is a **two-coordinate modal ROM**, not resolved water. This is simulation research, with no hardware or SPH/CFD validation.

**Honest status:** no evaluated controller passes the complete enlarged figure-eight task. Frozen V4 and its residual PPO complete 30 seconds but reach only 15/16 ordered windows. Keyboard stability depends on the command sequence; both successful traversals and ROM-triggered stops are retained.

## Quick start

Tested on Linux, Python 3.10.12 and MuJoCo 3.3.2. A C compiler (`cc`) is required for two process-local collision/observer plugins. They use the public MuJoCo ABI; they do not patch the engine. Keyboard mode requires a graphical desktop/OpenGL; offscreen recording was tested with EGL.

```bash
# In the extracted SloshBot-RL release directory:
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -B scripts/doctor.py
python -B teleop_keyboard.py
```

Editable installation, doctor, evaluation, tests and the graphical application were executed using the existing environment. The fresh-environment dependency download above was **not** executed; the pinned versions match the measured environment. No driver, CUDA or Chrono installation is required for these two ROM experiments. The tested distribution is a source-tree/editable release; a standalone wheel is not certified. Keep `_vendor/`, assets, configs and pretrained files beside the application.

## Experiment 1 — Autonomous figure-eight tracking

The desired path is `x=0.75 sin(2πt/30)`, `y=0.375 sin(4πt/30)`: 1.50 × 0.75 m over 30 s. Native RK4 uses 2 ms, the benchmark roughness bound is 2 mm, motor limits are ±10 Nm, workspace radius is 1.00 m, and the ROM displacement limit is 8.4 mm.

```bash
# Each output must be a NEW directory. No training is launched.
python -B demo_figure8.py --controller V4 --output runs/v4_demo --duration 30
python -B demo_rl.py --output runs/ppo_demo --duration 30
python -B -m sloshbot.experiment --controller V2_FFv1 --output runs/v2ff_demo
```

The existing V4/PPO results were reused byte-for-byte for traces; four additional compatible controllers were evaluated on the same three reset/terrain pairs. Observation/controller privilege is reported rather than treating this as an information-matched algorithm contest.

| Controller | Full 30 s | Full task | Observed limitation |
|---|---:|---:|---|
| Cascade V2 | 0/3 | 0/3 | Tracking guard near 1.3–1.4 s |
| Cascade V3 | 0/3 | 0/3 | Tracking guard near 1.3–1.4 s |
| V2_FFv1 | 0/3 | 0/3 | ROM guard near 20–24 s |
| V3_FFv1 | 3/3 | 0/3 | Large yaw drift, 15/16 windows |
| Slosh-aware V4, privileged | 3/3 | 0/3 | 15/16 windows |
| Standard residual PPO + V4, privileged | 3/3 | 0/3 | 15/16 windows |

PID balance-only, teacher transfer, and Gaussian Mixture PPO are **NOT EVALUATED** here: no verified equivalent enlarged-task release contract was established. Existing code or short historical results are not a 30-second comparison.

![Controller paths](assets/plots/experiment_1/trajectory.png)
![Ordered windows](assets/plots/experiment_1/windows.png)

See [per-trial comparison](experiments/experiment_1_figure8/comparison.csv), [evaluation protocol](configs/protocol.json) and [native videos](assets/videos/). Failed-prefix energies are not ranked against full trajectories. Full-task acceptance retains all sixteen synchronous windows, RMSE/max-error, tilt, roll/pitch, yaw, saturation, contact, ROM and path-length guards.

## Experiment 2 — Keyboard navigation and stability

![Keyboard command replay, including ROM failure](assets/gifs/experiment_2_keyboard.gif)

```bash
python -B teleop_keyboard.py
# Reproducible HIGH-LEVEL commands, closed-loop controller, new output:
python -B scripts/replay_commands.py --script configs/keyboard_script.json   --duration 16 --terrain playground --seed 2618001 --output runs/standard_replay
python -B scripts/replay_commands.py --script configs/terrain_traverse.json   --duration 26 --terrain playground --seed 2618001 --output runs/rough_replay
```

| Key | Action |
|---|---|
| Up / W | Forward |
| Down / S | Backward |
| Left / A | Turn left |
| Right / D | Turn right |
| Space | Controlled stop request |
| R | Reset |
| P | Pause/resume |
| Esc | Exit |
| 1 / 2 / 3 / 4 | Chase / isometric / top / side camera |

Key release and focus loss return the request to neutral. Commands pass through speed/yaw-rate bounds, slew limits, a world reference adapter and a stabilizing controller. Keys never directly specify motor torque. Speed is bounded to 0.04 m/s, yaw rate to 0.08 rad/s. The adapter combines inherited feedforward cascade control and heading P6; this is a separately named application controller, not a replacement for frozen V4.

The 4 × 4 m collision playground contains a flat centre, mild roughness, smooth uneven terrain, two small smooth raised features and open space. Three 26-second extended replays passed the registered stability checks and traversed flat/transition/mild/smooth regions, reaching approximately 0.72 m radius. **Raised features and arbitrary human trajectories remain unvalidated.**

The separate standard 16-second sequence passed on 2/3 contexts; the remaining run stopped at the ROM guard around 11.986 s. One controlled addition of frozen V4 modal feedback passed the first context but failed the second; its remaining trials were gated off. That variant is experimental and is not promoted as a superior controller.

![Actual closed-loop rough-terrain traversal](assets/gifs/rough_terrain.gif)
![Terrain paths](assets/plots/experiment_2/terrain_path.png)
![Matched balance traces, including failure](assets/plots/experiment_2/paired_balance.png)

Input timestamps, desired commands, native state, torque, modal coordinates, contact status and termination are saved. Recovery requires speed error ≤0.015 m/s, yaw-rate error ≤0.04 rad/s and tilt ≤0.06 rad held for 0.5 s before the next command; unsettled intervals are censored. The GUI smoke test measured about 25.5 rendered FPS and near-unity simulation/wall ratio over **only 2 seconds**, without human input. This is not a sustained real-time or user-usability claim.

See [stability trials](experiments/experiment_2_teleoperation/stability.csv), [recovery intervals](experiments/experiment_2_teleoperation/recovery.csv) and [command scripts](configs/).

## Control algorithms and reinforcement learning

Public entry points support V2, V3, their declared feedforward variants, frozen privileged V4 and the existing Standard residual PPO. Teleoperation exposes one robot-state FF/heading adapter plus a separately labeled privileged modal-feedback transfer diagnostic. Historical PID/teacher/GMM implementations are audited as context, not falsely advertised as evaluated release controllers.

PPO uses separate 64–64 Tanh actor/critic networks, 31 privileged observation features and three Gaussian residual actions. A shared learner collected experience from eight process-isolated environments, using the inherited exponential cost reward and V4 residual torque mapping. The 500,000 executed-transition campaign used one training seed; 497,600 transitions entered completed optimization buffers. Framework receipt was 499,992 because eight executed steps were lost at an I/O return boundary. Recovery reset environments and was not exact continuation; an SB3 worker-seed override is disclosed. This release performs **zero new training**.

![Real training returns](assets/plots/experiment_1/training_04_reward.png)

The selected checkpoint was frozen using development cases before held-out evaluation. The tiny nominal held-out tracking differences do not establish PPO superiority; energy changes are mixed. See [algorithm contracts](docs/ALGORITHMS.md) and [checkpoint provenance](docs/PPO_PROVENANCE.md).

## Liquid model and physical scope

The existing two-mode ROM uses one moving modal body with orthogonal passive slide coordinates and a fixed liquid component. Mass, native coupling, gravity and damping are retained. The visible modal marker is a generalized-coordinate illustration, **not a fluid surface**. `SPH_BLOCKED` remains in force. No hardware, CFD, full-fluid or universal-contact validity claim is made.

## Repository structure

```text
SloshBot-RL/
├── teleop_keyboard.py, demo_figure8.py, demo_rl.py
├── sloshbot/                 # applications, adapters, replay, presentation
├── _vendor/                  # unchanged historical runtime closure + model assets
├── configs/                  # frozen scripts and registered gates
├── experiments/
│   ├── experiment_1_figure8/  # native traces + controller comparison
│   └── experiment_2_teleoperation/ # commands, states, failures + stability
├── assets/                   # real GIFs, native videos, scientific plots
├── pretrained/ppo_500k.zip
├── scripts/, tests/, docs/, paper/
└── pyproject.toml, requirements.txt, LICENSE
```

## Reproducibility

```bash
python -B scripts/test_essential.py
python -B scripts/verify_release.py
```

Tests include retained native mass/COM/gravity/passive-joint checks, contact frames and virtual work, command semantics, exact reset replay, checkpoint integrity and controller-physics equality. SHA manifests protect historical source/results. [Reproducibility](docs/REPRODUCIBILITY.md) distinguishes reused data, new evaluation and diagnostic changes. Commands refuse to overwrite experiment directories. The registered manager skips completed outputs; it does not restart PPO.

## Research paper

[Versioned English manuscript](paper/main.tex), [new two-experiment results](paper/sections/two_experiments.tex) and [claim/evidence map](docs/CLAIM_EVIDENCE.md). No TeX engine was available, so no compiled manuscript PDF or checked final layout is supplied. Author/affiliation/venue metadata remains pending.

## Limitations

- Enlarged figure-eight full-task success remains 0/3 for every evaluated method.
- One trained PPO lineage, three evaluation contexts; no robust multi-training-seed superiority.
- Privileged V4/PPO and robot-state controllers use different information.
- Keyboard stability is command- and initial-state-dependent; ROM failures are retained.
- Some early observer-guard failures have only the last valid recorded frame, 2 ms before the native stop; later diagnostics also store the terminal native state.
- No human driving study, hardware deployment, SPH/CFD validation, or complete playground validation.
- The release is prepared locally and **has not been published to GitHub**.

## Citation and license

The upstream repository is [OpenBallBot-RL](https://github.com/salehiac/OpenBallBot-RL). The existing [Apache License 2.0](LICENSE) is retained. This release does not invent a DOI, author list or published-paper citation. Cite a verified repository revision once a public release exists. Dependency licenses remain their respective authors' licenses; see [NOTICE](NOTICE).

The final playground code normalizes all height data to [0,1] while retaining physical heights within float32 quantization. Existing16/26s results use the earlier representation; the corrected code has a new3s pilot and1s GUI smoke, not a claimed repeat of full-route validation. See [heightfield fix validation](docs/heightfield_fix_validation.json).
