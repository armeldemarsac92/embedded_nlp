# Embedded NLP for a Teensy-Based Keyboard Implant

This repository contains a lightweight French text classification pipeline built for a constrained embedded setting. The current codebase trains a compact hashed-feature MLP in Python, then exports headers that can be embedded in a Teensy 4.1 deployment.

The project grew in two phases:

- `src/` contains the current modular pipeline.
- `legacy/` preserves the earlier one-file trainers, generators, and verification scripts used during the initial research iterations.

## Repository layout

```text
.
├── data/      Historical and current datasets
├── docs/      Research writeups and article drafts
├── legacy/    Earlier scripts kept for reference
├── src/       Current training, export, and feature-extraction pipeline
└── README.md
```

## Current entrypoint

The main maintained entrypoint is:

```bash
python src/main.py
```

By default it:

1. Loads `data/DataSetTeensyv9_ULTRA_CLEAN.csv`
2. Verifies hashing compatibility with sklearn
3. Runs Optuna-based training
4. Exports generated artifacts under `artifacts/current/`

Generated files are intentionally not tracked.

## Important paths

- Dataset: `data/DataSetTeensyv9_ULTRA_CLEAN.csv`
- Current outputs: `artifacts/current/`
- Legacy outputs: `artifacts/legacy/`

## Legacy material

The `legacy/` folder contains the original single-file experimentation workflow:

- dataset generators and cleaner
- the first Optuna trainer
- batch and interactive model testers
- Python/C++ verification helpers

These files are kept for historical reference and comparison with the newer modular pipeline in `src/`.

## Notes

- The repository now separates source code from generated artifacts and local editor/runtime files.
- If you regenerate models or reports, the outputs will stay under ignored paths instead of cluttering the project root.
