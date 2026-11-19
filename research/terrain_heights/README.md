# Higher terrain comparison — 19 November 2026

New true collision mounds: nominal5/10/20mm and a mixed-height playground. Frozen V4/PPO, three paired contexts,30s enlarged figure-eight, unchanged robot/liquid/contact/actuators.24physically complete full-duration runs and4pilots; no full-task pass. No training.

Open REPORT_VI.md, media/mixed_5_10_20mm.gif, media/height_comparison.gif and figures/01_height_maps.png. The grid shown in native videos follows actual terrain geometry; no vertical exaggeration. ROM surface rendering is not CFD. Source runtime dependency: public_releases/SloshBot-RL_20261116_v2.

replay_viewer.py provides camera-interactive playback of recorded native states, not new simulation. experiment.py computes a new native run only into a new output directory. Raw traces, terminal states, models, normalized heightfields and all per-case metrics remain under runs/. protocols and registration are frozen in this result folder.
