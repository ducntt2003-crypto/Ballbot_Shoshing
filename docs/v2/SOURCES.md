# Primary references checked for V2

- [OpenBallBot-RL](https://github.com/salehiac/OpenBallBot-RL): requested reference repository; local frozen model/controller sources remain authoritative for this release, rather than applying upstream changes.
- [Chrono10 FSI manual](https://api.projectchrono.org/10.0.0/manual_fsi.html): independent FSI/SPH architecture. Exact numerical switches are taken from copied pinned local application source, not development-version API assumptions.
- [NASA-hosted Orr AAS21-433 paper](https://ntrs.nasa.gov/api/citations/20210000785/downloads/Orr_AAS_2021_rotary_paper.pdf): cylindrical eigenvalue, frequency, slosh mass and lever formulas. Bessel surface-to-ROM first-moment mapping is derived in this release and explicitly not independently validated.
- Installed MuJoCo3.3.2 headers and local engine visualization source: rendering-only triangle representation; no engine modification. Triangles are scene decorations.

Online sources were read on16 November2026. Physical observations and all performance/traversal numbers are from the actual local runs, not web claims. No experimental fluid data were collected.
