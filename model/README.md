# Model code

Implement a compact, direct multi-horizon forecasting model only after the external environment is selected and baseline evaluation is reproducible. Keep these responsibilities separate:

- Architecture and construction.
- Schema validation, chronology-preserving preprocessing, splitting, and loaders.
- Training, validation, checkpoints, early stopping, tracking, and latest-run logging.
- Evaluation without parameter updates.
- Artifact loading and 12-week inference with preprocessing parity.
- Serialization and restoration.
