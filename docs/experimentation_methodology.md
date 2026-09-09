# Experimentation methodology

Define the dataset provenance and version, leakage controls, split policy, selection metric, candidate comparison method, checkpoint rule, early-stopping policy, and keep or reject criteria before the first run.

Compare the last-value, seven-week moving-average, and 52-week seasonal-naive baselines before complex candidates. Compare BiLSTM-only, Transformer-only, and BiLSTM + Transformer candidates on the same rolling-origin validation windows with seeds 42, 123, and 2026, following [forecasting experiments](forecasting_experiments.md). Other candidates are deferred. Use validation data to tune candidates, then evaluate the selected champion once on a held-out test partition. Preserve each completed run in `experiments/registry.json` and update `experiments/latest.json` atomically.

Record `run_id`, data version, feature set, model type, lookback, horizon, validation MAE and sMAPE, and training timestamp. Record test MAE and sMAPE only for the selected artifact after its final evaluation. Leave alert precision and recall unmeasured until category integration semantics are defined. Use the agreed JSON/CSV tracking and Google Drive persistence described in the [training contract](training.md#selected-experiment-stack). Save each candidate's best validation checkpoint with the scaler, feature list, hyperparameters, data date range, and schema version.

Follow the frozen validation-MAE early-stopping, checkpoint, metric, and selection rules in [forecasting-v1](forecasting_protocol.md). Document optimizer and scheduler rationale, observability, environment parity, and reproducibility limits. Fixed seeds improve reproducibility but do not guarantee identical outcomes across hardware or nondeterministic operations.
