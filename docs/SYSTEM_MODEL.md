# System model and source mapping

The research demo integrates the original BallBot geometry with an assumed cylindrical prototype tank. No hardware dimensions or damping measurements are claimed. SI units are used; world XY defines the reference, tank axes define modal displacement, and MuJoCo uses native quaternion ordering wxyz.

## Assembled dynamics

$$M(q)\ddot q+b(q,\dot q)=B\tau+J^Tf_c+f_p.$$

This summarizes native integration, not a separate plant. [Model construction](../_vendor/ballbot_sloshing_research/coupling/model.py) supplies one fixed fluid mass and one moving mass with two orthogonal passive slide joints. [Robot model builder](../_vendor/chrono_fsi/integration/geometry.py) retains the upstream robot/actuators. The sphere radius is 0.090 m and motor limits ±10 Nm.

## Liquid parameters and reduced equation

[ModalParameters](../_vendor/ballbot_sloshing_research/fluid/reduced_order.py) defines R=0.070 m, V=0.0005 m³, rho=998 kg/m³, h=V/(pi R²)=0.0324806 m, zeta=0.02 (assumed). With alpha=1.8411837813406593:

$$\omega^2=g(\alpha/R)\tanh(\alpha h/R),\quad m_1=\frac{2mR\tanh(\alpha h/R)}{h\alpha(\alpha^2-1)},\quad k_s=m_1\omega^2,\quad c=2\zeta\omega m_1.$$

m=0.499 kg; m1=0.338884966314 kg; fixed=0.160115033686 kg; omega=13.375241112 rad/s; k_s=60.6255291684 N/m; c=0.181306725347 N s/m. One mass moves in two directions; its mass is not duplicated.

The standalone translation limit is ddot(xi)+2*zeta*omega*dot(xi)+omega²*xi=g_parallel-a_parallel. In the actual native path MuJoCo integrates coupled inertial/rotating effects, including slide reactions, rather than calling a second oscillator. [Full native stepping](../_vendor/ballbot_sloshing_research/envs/ballbot_sloshing_env.py).

$$E=\tfrac12m_1\|\dot\xi\|^2+\tfrac12 k_s\|\xi\|^2.$$

This is modal mechanical energy, not established continuum total fluid energy.

## Reference and coupling

[Analytic reference](../_vendor/ballbot_sloshing_research/track8/reference.py): p=p0+[A sin(wt), B sin(2wt)], A=.75 m, B=.375 m, w=2pi/30 s. Velocity/acceleration are exact analytic derivatives. Nonzero initial reference velocity and stationary robot make startup part of the task.

[reaction_wrench](../_vendor/ballbot_sloshing_research/coupling/model.py) calculates world fluid-on-tank force and torque at a requested world origin using rigid-body momentum balance. Gravity is included once; the diagnostic wrench is not re-applied. Native modal mass is placed at h/2: an acknowledged COM-height approximation, not the continuum equivalent pressure-moment lever.

[Detailed ROM/surface audit](v2/LIQUID_MODEL_VERIFICATION.md) includes the Bessel-based, volume/moment-consistent linear visualization. Rendering changes only MjvScene. [Heightfield geometry](../research/terrain_heights/geometry.json) gives actual elevations/slopes; it is collision geometry. Compliance audit finds ~5.4–5.6 mm penetration; no contact tuning is hidden.
