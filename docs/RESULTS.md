# Results and interpretation

## Original enlarged task

| Method | n | Mean observed duration (s) | XY RMSE (mm) | Ordered windows | Full task |
|---|---:|---:|---:|---|---:|
| V2 | 3 | 1.39 | 158.84 | 0,0,0 | 0/3 |
| V3 | 3 | 1.39 | 157.27 | 0,0,0 | 0/3 |
| V2_FFv1 | 3 | 21.88 | 68.66 | 11,10,12 | 0/3 |
| V3_FFv1 | 3 | 30.00 | 67.50 | 15,15,15 | 0/3 |
| V4 | 3 | 30.00 | 68.08 | 15,15,15 | 0/3 |
| PPO | 3 | 30.00 | 68.04 | 15,15,15 | 0/3 |

Means over three contexts; early-terminated metrics describe observed prefixes only. The baseline table is generated from the preserved [raw CSV](../experiments/experiment_1_figure8/comparison.csv).

## Height study

| Terrain | V4 RMSE (mm) | PPO RMSE (mm) | Ordered windows, each controller | Full task |
|---|---:|---:|---|---:|
| 5 mm | 71.80 | 71.77 | 15, 15, 15/16 | 0/6 |
| 10 mm | 81.35 | 81.32 | 14, 14, 14/16 | 0/6 |
| 20 mm | 113.87 | 113.85 | 10, 11, 11/16 | 0/6 |
| Mixed 5–20 mm | 100.07 | 100.03 | 12, 12, 12/16 | 0/6 |

Means over three paired reset contexts, not three trained policies. [Raw rows](../research/terrain_heights/comparison.csv) retain all 24 runs, zero physical-duration failures and zero full-task successes. [Full Vietnamese height report](../research/terrain_heights/REPORT_VI.md).

PPO minus V4 RMSE is only about−0.025to−0.038mm. This is not sufficient superiority evidence. H20 mean RMSE exceeds100mm; all other levels still miss ordered windows. Maximum observed tilt is about 0.04960 rad, below 0.12 rad; no height-study ROM or actuator guard failures were recorded. Height-study modal energy does not increase monotonically with height.

## Navigation outcomes

17/18 geometric routes pass; COURSE seed 2618001 hits the ROM guard at approximately 19.57 s. Motion quality is15/18. [Quality audit](../upgrade/results/motion_quality.csv), [torque audit](v2/torque_saturation.json). Up to 39.4% of samples are near 9.9 Nm on at least one motor in some navigation runs. Physical guards, geometric traversal and tracking quality are different endpoints.

## Figures and uncertainty

[Trajectory](../assets/plots/experiment_1/trajectory.png), [tracking](../assets/plots/experiment_1/tracking_error.png), [energy](../assets/plots/experiment_1/modal_energy.png), [balance](../assets/plots/experiment_1/balance.png), [torque](../assets/plots/experiment_1/torque.png), [height curves](../research/terrain_heights/figures/03_performance_by_height.png). Raw seed-level rows are preferred to unsupported significance claims. This update does not estimate training-seed confidence intervals from reset episodes.

The liquid is a linear ROM with assumed damping and approximate moment lever; contact compliance and timestep uncertainty remain. Independent Chrono screening is SPH_BLOCKED; no hardware validation.
