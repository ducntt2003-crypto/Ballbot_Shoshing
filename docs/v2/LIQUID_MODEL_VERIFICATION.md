# Liquid ROM, surface reconstruction and coupling audit

## Frozen physical baseline

The native model is unchanged from V1. Tank internal radius0.070m, internal height0.160m, nominal volume0.0005m³, fluid density998kg/m³, depth0.032480600631m and liquid mass0.499kg. The factory shell has2mm walls/bottom and an assumed polymer density1200kg/m³. Geometry is an assumed prototype, not measured hardware.

The first antisymmetric cylindrical eigenvalue is α=1.8411837813406593, k=α/R, ω²=gk tanh(kh). The inherited moving mass is m₁=2mR tanh(αh/R)/[hα(α²−1)]=0.338884966314kg; fixed mass0.160115033686kg. One moving body has TWO orthogonal slide coordinates, not two copies of moving mass. ω=13.375241112rad/s, f=2.128735738Hz. Passive spring k_s=m₁ω²=60.6255291684N/m and damper c=2ζωm₁=0.181306725347N·s/m, with **assumed**, unmeasured ζ=0.02. The two q coordinates are metres in tank axes, qdot in m/s.

The [NASA-hosted Orr paper](https://ntrs.nasa.gov/api/citations/20210000785/downloads/Orr_AAS_2021_rotary_paper.pdf), equations1–4, supports the linear cylinder frequency and equivalent-mass formulas. This does not validate this project's complete implementation, damping, contact or nonlinear fluid response.

## Native reaction and mass accounting

`_vendor/ballbot_sloshing_research/coupling/model.py` creates a fixed fluid body and one movable modal body at h/2, passive slide stiffness/damping and tiny positive modal inertia. Native MuJoCo integrates the coupled multibody mass matrix, gravity, gyroscopic effects and constraints. Tank translation/rotation, acceleration, projected gravity and inertial terms excite the modes through those equations; this is not an independently applied acceleration filter.

The fluid-on-tank diagnostic returns WORLD [F,τ] at the requested WORLD reference point using rigid-body momentum balance. Gravity is included once in native dynamics; `reaction_wrench` is a diagnostic and is NOT reapplied. No extra fluid mass or reaction load is added by V2. Retained tests check rigid/liquid mass and COM consistency, passive spring/damper forces, gravity once, action signs and static wrench. Code inspection rules out duplicate application in the published path; it does not prove equivalence to all continuum pressure moments.

**Moment limitation:** the baseline puts modal mass at h/2. The linear continuum equivalent lever from Orr Eq4 would place it at h−R tanh(αh/R)/α, a different height. The baseline itself labels this COM-height approximation, not full pressure-moment equivalence. This release deliberately does not change that historical model or silently claim its reaction torque is physically validated.

## ROM-driven surface: derived visualization, not new dynamics

For r=√(x²+y²), define

η(x,y) = C J₁(kr) [qₓx+qᵧy]/r,
C = (m₁/m) k h / J₂(α).

The removable r=0 limit is k/2. The displayed surface is z=h+η. The Bessel radial derivative vanishes at R. The continuous mean η is zero, so cylinder volume is constant. Since ∫₀ᴿr²J₁(kr)dr=R²J₂(α)/k, its first volume moments reproduce ΔCOM_xy=(m₁/m)q exactly in this linear reconstruction. Those identities are tested by quadrature. No oscillator, random wave or amplitude multiplier is added to q. Default native rendering uses a four-ring,32-angle triangle mesh; this is a visual tessellation, not SPH resolution.

The mode basis is linear potential-flow theory; mapping the equivalent mechanical state to this surface through its first moment is a **derived approximate reconstruction**. No current SPH/experimental surface comparison establishes its accuracy. It omits higher modes, breaking, splashing, wetting, contact-angle and capillary effects. Native modal energy½m₁|qdot|²+½k_s|q|² is not asserted equal to continuum fluid total kinetic-plus-potential energy.

`surface.py` changes only MjvScene triangles/colors. Model mass, inertia, q/qdot, controls, passive forces and integration are never written by the renderer. The world transformation uses the actual tank body frame. Transparent walls and the hidden equivalent-mass marker are scene copies only. Tests compare physics arrays before/after decoration. Green trails use actual body XYZ projected onto the exact collision heightfield facets with a2.5mm visual offset. Decorative grid/arrow objects are not collision obstacles.

Status: ROM preserved; renderer tested as read-only; **SPH_BLOCKED**, no physical liquid validation.
