# Reproducibility and preservation

The publication is a separate export; historical source/results are not modified. [Original V2 README](publication_history/v2/README.md), [original reproducibility notes](publication_history/v2/docs/REPRODUCIBILITY.md) and original manifests are retained. Statements there about publication status describe that historical snapshot.

## Environment

Python3.10, pinned [requirements](../requirements.txt); editable source installation without dependency changes was tested in the existing environment. New clean-OS installation remains untested. MuJoCo3.3.2 is required by the native observer ABI; local C compiler and graphical desktop for GUI. CPU inference uses the frozen checkpoint; no training/download is performed by demos.

```bash
python -B scripts/doctor.py
python -B scripts/test_essential.py
python -B scripts/verify_release.py
```

The essential suite executes a small counted native-step budget. The verifier checks publication hashes, protected V2 hashes, all18baseline and24height rows, aggregate means, checkpoint identity, local Markdown links and manuscript sources. It does not train or simulate. Use the new publication manifest; the old exact-file-set manifest is historical and intentionally predates added docs/assets.

## Run only missing/new experiments

```bash
python -B demo_figure8.py --controller V4 --duration 30 --output runs/new_v4
python -B demo_rl.py --duration 30 --output runs/new_ppo
python -B -m sloshbot.upgrade_evaluate --level BUMPS --seed 2618001 --duration 26 --output runs/new_navigation
```

All output paths must be new. Short smokes verify launch only, not 30 s acceptance. Do not rerun coordinators to regenerate plots. [Protocols](EXPERIMENTS.md), [raw results](RESULTS.md), [PPO logs](REINFORCEMENT_LEARNING.md).

## Height artifacts and source dependency

[Height experiment.py](../research/terrain_heights/experiment.py), manager, plotting scripts and original protocol are preserved byte-for-byte. They record their original workspace paths and depend on the frozen V2 export at public_releases/SloshBot-RL_20261116_v2. Those original coordinators are archival entrypoints, not portable top-level launchers. Recreate that path from the929-entry manifest with historical changed documents restored from docs/publication_history/v2 before using them; never overwrite current publication/results. Raw traces/models/heights/terminalstates and figures are included, so verification needs no rerun. Assets in savedmodelXMLuse __ASSET_ROOT__, resolved to _vendor/ballbotgym/ballbotgym/assets, not a user's absolute path. [Height verification receipt](../research/terrain_heights/verification.json).

## Frozen evidence

The 500k checkpoint SHA-256 is 651989a98dd3ab04294adddd4a0bc0fb63e09b96749e9a980c5041e51bbd5db7. Actual execution: 500,000; optimization: 497,600; one policy seed. Eight workers are not independent policy replications. [Original PPO provenance](PPO_PROVENANCE.md).

Rendering uses recorded native qpos/qvel and introduces zero physics steps. GIFs represent actual runs; the surface is a ROM visualization. Statistical inference cannot treat 2 ms samples as independent episodes. No sensor parity, CFD accuracy, hardware performance or full-task success is implied.
