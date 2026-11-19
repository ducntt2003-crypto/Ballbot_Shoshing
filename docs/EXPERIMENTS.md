# Experimental protocols

## Experiment 1: enlarged autonomous figure-eight

[Registered protocol](../configs/protocol.json), [metrics source](../_vendor/ballbot_sloshing_research/track8/evaluate.py). A=.75 m, B=.375 m, T=30 s, 2ms native RK4,10ms policy. Three contexts: reset2616001..3, paired terrain2617001..3. Six methods produce18 trials. V4/PPO traces were reused from frozen completed experiments, not regenerated.

Full completion requires TIME_LIMIT and30s; all16 temporal windows |t-T*j/16|<=T/32 contain a sample within .1m of reference; XY RMSE<=.1m; max error<=.25m; actual length>=.7 reference; max tilt<=.12rad; roll/pitch RMS<=5degrees; yaw drift<=10degrees; fraction of samples with any torque>=9.9Nm<=.05; modal norm<=.0084m; native validity. A physical-valid flag does not replace these task/safety checks.

## Experiment 2: rough-terrain recorded navigation

[Route protocol](../upgrade/configs/protocol.json), [route table](../upgrade/results/traversal.csv). Six4x4m heightfields, reset2618001..3, fixed26s high-level velocity/yaw commands. Limits .04m/s,.08rad/s,.04m/s²,.08rad/s². Ordered approach/corridor/post-feature progress and>=.5m displacement define geometric traversal plus physical validity/full route. Separately, velocityRMSE<=.04m/s and yawrateRMSE<=.08rad/s define motion quality. Keyboard input is one command source; no human study is claimed.

## Height extension of Experiment 1

[Protocol](../research/terrain_heights/protocol.json), [registration](../research/terrain_heights/registration.json), [geometry](../research/terrain_heights/geometry.json). Frozen V4/PPO, reset2611801..3, fixed terrain seed2611811,30s each,4conditions=24full runs, plus4V4 3s pilots. 366,000native steps; aggregate worker wall493.316s, registered cap850s/400,000steps. Each full run capped90s. New training0.

129x129collision grids span4x4m; compact C2 positive bumps have kernel max(0,1-r²/R²)^3. H05/H10/H20 use the same .60m footprint, different nominal height; MIXED has separate5/10/20mm hills. Maximum sampled slope<=.06rad, initial flat disc>=.12m. No gains, contact material, masses, damping, ROM limits or checkpoints were retuned. New model XML recompiles height bounds to match native collision AABBs. This is not a hard vertical-step benchmark.

Height progression used a geometry screen,3sV4pilot, then paired full runs; higher levels opened only if both controllers had>=2/3physical completions. Original task gates remain unchanged. Saved trace rows are native2ms samples; render replay introduces no physics. Baseline2mmterrain has a different footprint, so comparisons to it are not amplitude-only causal ablations.
