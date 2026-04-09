# Legacy Scripts

This folder preserves the earlier single-file workflow used before the modular `src/` pipeline existed.

Use it if you want to inspect the original experimentation process:

- dataset generation and cleaning
- the initial Optuna trainer
- interactive and batch testing helpers
- Python/C++ verification utilities

These scripts now read datasets from `../data/` and write generated artifacts under `../artifacts/legacy/`.
