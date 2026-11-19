# Liquid ROM versus high-fidelity reference

**HIGH_FIDELITY_REFERENCE_IMPLEMENTED — SPH_BLOCKED.** Independent GPU data exist, but dynamic ROM validation is not completed.

## Solver and numerical setup

Project Chrono10.0.0, pinned commit9faf13dd8f1128dd75ed233a9627027b0422c3f7, existing RTX4090Laptop GPU build. Binary/source hashes are in protocol.json. The existing activation script was used; cuInit and device count succeeded. No driver, CUDA Toolkit, Chrono core or MuJoCo modification occurred. Chrono is an independent numerical model, not experimental truth. [Chrono10 FSI documentation](https://api.projectchrono.org/10.0.0/manual_fsi.html) describes its SPH/rigid coupling architecture; exact settings here come from the pinned local application source. The upstream BSD license is retained beside the copied application source.

Cylinder R70mm, wall height160mm, nominal fill0.5L, rho998kg/m³, viscosity0.001002Pa·s at assumed20°C, g9.81m/s². GPU WCSPH: Tait gamma7, sound speed10m/s, cubic kernel h/dx1.2, Adami/cartesian-complement BCE with3layers, RK2, delta-SPH0.1, no shifting/artificial viscosity. These are inherited exploratory settings, not tuned to match ROM. Static container is prescribed by a constrained Chrono fixture; its rotational inertia is explicitly uncalibrated.

| Case | N | Actual mass kg | Peak pressure RMSE Pa | Peak RMS velocity m/s | Max weight-force defect | Screen |
|---|---:|---:|---:|---:|---:|---|
|dx8_dt50|964|0.49218242|6.39146|0.0011772|1.250%|PASS short screen|
|dx6_dt50|2145|0.46201933|36.98332|0.0020394|9.926%|FAIL|
|dx6_dt25|2145|0.46201933|36.97890|0.0020393|9.927%|FAIL|

Each run covers20ms;400+400+800=1600 new native GPU steps. Runtime per run was approximately4.90,5.92 and11.94s; use execution.json/run_status for exact timings. Particle mass sum is constant and no escaped particle is reported, but this alone does not validate hydrostatics. Estimated volumeΣm/rho is reported separately; WCSPH is weakly compressible, so exact constant volume is not assumed.

## Gates and resolution limitations

Preregistered short static thresholds: peak pressure RMSE≤31.8Pa, RMS relative velocity≤0.005m/s, density error≤1%, COM drift≤0.1dx, wall force/weight defect≤5%, mass drift≤1e−12 and no escapes. The8mm case passes this SHORT screen. Both6mm cases fail pressure and wall-force criteria. This prevents opening the dynamic comparison.

The two spacings have actual masses0.49218 and0.46202kg, versus nominal ROM mass0.499kg. The Cartesian fill uses floor(h/dx) layers and a circular inclusion mask. Therefore this is a **two-spacing numerical screening with geometry/mass error, NOT a consistent spatial convergence study**. Finer spacing does not monotonically improve pressure. Changing density or rescaling mass to disguise this mismatch was not attempted.

At6mm,25/50µs gives closely matched20ms histories; this supports only a limited temporal check on that discretization, not long-term accuracy. The window is much shorter than the approximately0.470s first-mode period. Frequency, damping and phase lag cannot be identified from these static20ms data.

## Comparable observables

- Static COM change is compared relative to each initial COM, not between different fill heights as though that were ROM error.
- Fluid-on-tank force is in WORLD coordinates; torque is shifted from the dry fixture COM to the tank-floor origin by τ_T=τ_C+(r_C−r_T)×F. Expected static Fz=−mg uses each solver run's ACTUAL mass. Absolute force differences versus0.499kg ROM are not called force prediction errors.
- Pressure RMSE uses the native application's linear hydrostatic reference at nominal fill. The initial Tait/linear-reference discrepancy is small but explicitly present.
- ROM modal energy and particle total/relative kinetic energies are different quantities; no incompatible energy-error percentage is reported.
- Surface displacement, modal frequency/damping, phase lag and dynamic reaction-load errors: **NOT MEASURED**, because dynamic cases remain gated.

## Six prescribed cases

| Case | Status |
|---|---|
| Static container | Three new GPU runs; two6mm numerical screens fail |
| Horizontal acceleration | NOT RUN; static/mass-match gate incomplete |
| Braking | NOT RUN; same gate |
| Turning/lateral acceleration | NOT RUN; same gate, existing worker supports1D translation only |
| Container tilt | NOT RUN;6DoF prescribed-motion worker not implemented |
| Recorded rough-terrain motion | Actual6DoF packet exported and interpolation tested; NOT fed into SPH |

Measured tank body-origin positions, rotations, WORLD linear/angular velocities are stored in recorded_tank_motion.csv/npz. A C1 Hermite position path and C2 rotation spline provide internally differentiated kinematics without extrapolation. Spline angular rates differ slightly from measured native rates; that interpolation uncertainty is reported. No interface is silently advertised as a completed6DoF SPH coupling.

## Engineering decision

Use unchanged ROM for the bounded real-time application and a clearly marked ROM-based renderer. Retain Chrono as the offline independent candidate because it is installed, GPU-executable and provides free-surface/BCE functionality. A new engine installation is not justified by these data alone. First resolve fixed-geometry/fill/mass consistency and static pressure/force screens, then extend duration/resolution and only then excite the six motions. No full fluid–BallBot co-simulation, real-fluid accuracy claim, or large training is opened.
