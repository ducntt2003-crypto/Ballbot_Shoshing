# Navigation commands and keyboard interface

Keyboard input supplies bounded speed/turn requests to a closed-loop controller, not direct torque. This interface is an implemented usability feature, not the main research contribution.

```bash
python -B teleop_keyboard.py --level BUMPS
```

Desktop/GLFW is required. Available levels are the choices printed by `python -B teleop_keyboard.py --help`. Arrow keys/WASD request forward/reverse/turn; SPACE requests a slew-limited stop (not an instantaneous safety brake); R starts a new reset episode; P pauses; ESC exits. Cameras 1–5 select chase/isometric/top/side/liquid views. Focus loss returns input to neutral.

Speed limit: 0.04 m/s; yaw rate: 0.08 rad/s; linear acceleration: 0.04 m/s²; yaw acceleration: 0.08 rad/s². Position and heading requests may differ from measured state. Green trails are actual positions with a small scene-only offset. Visible liquid is a ROM-based reconstruction. Sustained real-time execution is not certified.

```bash
# Bounded GUI launch check, not an evaluation episode:
python -B teleop_keyboard.py --level BUMPS --smoke-seconds 1 --output runs/my_new_gui_smoke
```

Logs include keyboard press/release, wall/simulation time and reset episode; native telemetry contains states, motor torques and guards. Outputs must be new directories. [Command conversion tool](../scripts/keyboard_log_to_script.py) accepts one explicit reset episode. Replaying requests closes the loop again; it does not apply saved torques. The registered route script is distinct from an arbitrary human session. [Reproducibility](REPRODUCIBILITY.md).
