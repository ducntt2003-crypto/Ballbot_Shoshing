# SloshBot-RL

**Learning to Balance, Track Trajectories, and Carry Liquids on Rough Terrain**

![Actual terrain traversal and green trail](upgrade/media/keyboard_live_trail.gif)

This version adds registered low-terrain crossings, a live native trajectory trail and a liquid surface reconstructed from the actual two-mode ROM. Independent Chrono GPU screening remains numerically incomplete: **SPH_BLOCKED**. A visible water surface is not evidence of validated CFD or accurate liquid torques.

The project retains exactly two primary experiments. All V1 models, controller gains, checkpoints and experiment outputs are preserved; this is a separate versioned source export. No PPO training or GitHub publication is performed.

## Quick start — Mode A: interactive ROM

```bash
# In the extracted source tree, with the tested dependencies already installed:
python -B scripts/doctor.py
python -B teleop_keyboard.py --level MILD
python -B teleop_keyboard.py --level BUMPS
```

The V1 pinned requirements and editable source-tree setup remain in [requirements.txt](requirements.txt) and [reproducibility notes](docs/REPRODUCIBILITY.md). Existing Python3.10/MuJoCo3.3.2 dependencies were reused. No new large library, CUDA or driver was installed. A compiler builds the process-local contact/observer plugins into .cache. A graphical desktop is needed for keyboard mode; EGL was used for offline MP4/GIF rendering. Fresh dependency installation is not claimed tested.

| Keys | Command |
|---|---|
| Arrows / WASD | Forward, reverse, left/right turn |
| SPACE | Slew-limited controlled stop |
| R / P / ESC | Reset / pause-resume / exit |
| 1 / 2 / 3 / 4 / 5 | Chase / fixed isometric / top / side / liquid close-up |

Requests are bounded velocity/yaw targets, not motor torques. Forward follows the commanded robot-heading reference; measured robot heading closes the P6 loop, so actual and desired headings can differ. Speed0.04m/s, yaw rate0.08rad/s and inherited acceleration limits are retained. Jerk limiting is not added or claimed. Press/release and focus loss return requests to neutral. WORLD XYZ, measured tank motion, native state, modal state, torque, contact, keyboard timestamps and reset episode are logged.

Green trails use actual position samples and the true heightfield surface plus2.5mm display offset. Landmarks and heading arrows are visualization-only. Short enhanced GUI smokes measured about26FPS but simulated/wall ratios only0.59–0.77 (under varying CPU load); this version is **interactive, not certified real-time**. No human driving study was performed.

## Experiment 1 — Autonomous Figure-Eight Tracking and Controller Comparison

The inherited1.50×0.75m,30s experiment is unchanged: V2, V3, V2_FFv1, V3_FFv1, privileged V4 and privileged Standard residual PPO, three matched contexts each. All methods have0/3 full-task passes. V4/PPO complete30s but reach15/16 ordered windows. V3_FFv1 has excessive yaw drift. Failed prefixes, privilege distinctions and all original gates remain intact.

See [original comparison CSV](experiments/experiment_1_figure8/comparison.csv), [PPO provenance](docs/PPO_PROVENANCE.md) and [actual training plot](assets/plots/experiment_1/training_04_reward.png). The existing500k campaign is reused, not rerun. PID, teacher and Gaussian-mixture variants remain NOT EVALUATED under this enlarged-task release contract.

## Experiment 2 — Keyboard-Controlled Rough-Terrain Navigation and Balance Stability

All six new collision heightfields span4×4m. Dimensions are derived from the measured90mm sphere radius and a conservative initial non-ball clearance near89.9mm. New bump/plateau height is at most2% of sphere radius; maximum facet slope is screened below0.03rad, one quarter of the inherited tilt guard. This margin is a screening rule, not a stability proof. Mild terrain retains the inherited2mm generator with a flat initialization region.

| Condition | Nominal height | Max facet slope | Full route passes |
|---|---:|---:|---:|
|FLAT|0.00mm|0.00°|3/3|
|MILD|2.00mm|1.24°|3/3|
|RANDOM|1.50mm|1.10°|3/3|
|BUMPS|1.79mm|1.05°|3/3|
|SLOPES|1.80mm|1.11°|3/3|
|COURSE|1.20mm|1.45°|2/3|

Each route uses the fixed26s high-level keyboard script, three reset contexts,2ms native RK4, original motor/ROM/contact limits and unchanged Teleop_FF_Heading controller. Success requires ordered approach, corridor crossing, progress beyond the registered feature, at least0.5m XY displacement, physical validity and full route completion. A pilot or a spinning sphere is not counted as traversal. COURSE seed2618001 stops at a ROM guard around19.57s; its failure is preserved.

![Actual mild roughness](upgrade/media/mild_rough.gif)
![Actual smooth bump crossing](upgrade/media/smooth_bumps.gif)
![Measured base height](upgrade/figures/height.png)

**Contact limitation:** offline native queries find about5.4–5.6mm compliant penetration, larger than the1.2–1.8mm features. Sphere-centre height nevertheless rises about1.80–1.83mm at bump/plateau and falls after them. These are genuine movements in the inherited compliant-contact simulation; they do not establish hard-obstacle or hardware capability. No contact parameter is retuned. See [contact audit](docs/v2/contact_traversal_audit.json).

```bash
# Mode B: research evaluation; always choose a NEW output directory.
python -B -m sloshbot.upgrade_evaluate --level BUMPS --seed 2618001   --duration 26 --output runs/new_bump_trial
python -B -m sloshbot.upgrade_evaluate --level SLOPES --seed 2618002   --duration 26 --output runs/new_slope_trial
```

See [registered protocol](upgrade/configs/protocol.json), [all traversal metrics](upgrade/results/traversal.csv), [geometry normals/slopes](upgrade/results/geometry.json), [native videos](upgrade/media/) and [actual container motion packet](research/high_fidelity/recorded_tank_motion.csv). Contact, torque and ROM failures remain per-trial outcomes, not deleted samples.

## Dynamic liquid: physics versus visualization

![Actual ROM states during acceleration and braking](upgrade/media/liquid_acceleration_braking.gif)
![Actual turning with liquid](upgrade/media/turning_with_liquid.gif)

The existing liquid model has0.499kg split into one fixed component and ONE moving body with two passive slide coordinates. Modal frequency is2.1287Hz; damping ratio0.02 is assumed. Native MuJoCo supplies two-way multibody reactions; the diagnostic wrench is not applied a second time. Modal mass is not duplicated per axis.

The new surface uses a cylindrical first-mode Bessel basis, normalized so its mean height is constant and its first volume moment equals the ROM liquid-COM shift. It is driven only by actual q states, transformed with the measured tank pose. This is a **derived linear visualization approximation**, not a new liquid solver, validated surface field, nonlinear slosh model or SPH particles. Transparent walls and all surface triangles exist only in MjvScene. Tests verify that rendering does not mutate physics state.

[ROM/surface/coupling audit](docs/v2/LIQUID_MODEL_VERIFICATION.md) documents SI units, assumptions and the baseline modal-height approximation that limits continuum torque fidelity.

## Mode C — Independent high-fidelity liquid candidate

Chrono10.0.0 was run on the available RTX4090Laptop GPU in an isolated static container:8mm/50µs,6mm/50µs and6mm/25µs. The new total is1600 native SPH steps. The coarse20ms screen passes; both6mm screens fail pressure/weight-force gates. Actual liquid masses differ between resolutions, so this is not a consistent spatial-convergence study. Acceleration, braking, turning, tilt and recorded-terrain fluid replays remain gated off.

![Independent GPU static screening](research/high_fidelity/static_screen.png)

Read the complete [LIQUID_ROM_VS_HIGH_FIDELITY_REPORT](LIQUID_ROM_VS_HIGH_FIDELITY_REPORT.md). Dynamic surface/frequency/damping/phase/force errors are NOT MEASURED. No ROM-versus-SPH demonstration GIF is fabricated. The measured6DoF tank packet and tested interpolator are available, but are not falsely advertised as a completed SPH coupling. The external existing Chrono build is required for rerunning Mode C; it is not installed or bundled automatically.

## Tests, reproducibility and paper

```bash
python -B scripts/test_essential.py
python -B scripts/verify_release.py
```

The test suite retains native mass/COM/gravity/contact checks and adds translation, ordered obstacle accounting, normalized geometry, Bessel volume/moment consistency, visualization separation, native SPH-output checks, motion replay and camera transforms. A versioned English [manuscript](paper/main.tex) adds [terrain and multi-fidelity sections](paper/sections/terrain_liquid_upgrade.tex). No TeX compiler is available, so no compiled paper PDF is claimed. [Vietnamese handoff](REPORT_VI.md) answers all requested questions.

## Structure and limits

`sloshbot/` contains the adapters/renderers; `upgrade/` holds the new protocols/traces/plots/media; `research/high_fidelity/` holds independent GPU screening and motion packets. `_vendor/`, `pretrained/`, original Experiment1 and all V1 datasets remain preserved. New results are write-once; old results are not relabeled.

No hardware, SPH/CFD accuracy, nonlinear fluid validation, broad terrain robustness, PPO superiority or sustained real-time claim is supported. The liquid status remains **SPH_BLOCKED**. Arbitrary human commands and larger bumps/slopes remain unvalidated. Complete figure-eight acceptance remains unresolved.

## Attribution and license

Based on [OpenBallBot-RL](https://github.com/salehiac/OpenBallBot-RL). The existing [Apache2.0 license](LICENSE) and [NOTICE](NOTICE) are retained; the [Chrono BSD license](research/high_fidelity/CHRONO_LICENSE) accompanies its reference description. No author, DOI, hardware result or public GitHub release is invented. This package is prepared locally and has not been published.

### Motion quality and actuator headroom

Geometric route completion is not identical to the earlier velocity/yaw-quality gate. Applying the unchanged V1 thresholds descriptively gives FLAT3/3, MILD3/3, RANDOM3/3, BUMPS2/3, SLOPES3/3 and COURSE1/3. These secondary numbers do not rewrite the registered traversal definition. Motors reach±10Nm; some trials spend up to39.4% of samples at or above9.9Nm on at least one motor. This limited headroom is not hidden behind successful route crossing. The original Experiment1 saturation guard remains unchanged. See [motion quality CSV](upgrade/results/motion_quality.csv) and [saturation audit](docs/v2/torque_saturation.json).
