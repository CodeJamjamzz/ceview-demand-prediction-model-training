# Model code

Implement a compact, direct multi-horizon forecasting model only after the external environment is selected and baseline evaluation is reproducible. Keep these responsibilities separate:

- Architecture and construction.
- Schema validation, chronology-preserving preprocessing, splitting, and loaders.
- Training, validation, checkpoints, early stopping, tracking, and latest-run logging.
- Evaluation without parameter updates.
- Artifact loading and 12-week inference with preprocessing parity.
- Serialization and restoration.

The initial shared layers now live in [components.py](components.py), raw-index metrics in [metrics.py](metrics.py), and protocol/file verification in [experiment.py](experiment.py). They use [forecasting-v1](../experiments/forecasting-v1.json). A/B/C models are assembled in [architectures.py](architectures.py), and the external runner lives in [training/](../training/README.md); see the [protocol documentation](../docs/forecasting_protocol.md).
