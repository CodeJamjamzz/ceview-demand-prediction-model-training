# Experimentation methodology

Define the dataset provenance and version, leakage controls, split policy, selection metric, candidate comparison method, checkpoint rule, early-stopping policy, and keep or reject criteria before the first run.

Compare the last-value, seven-week moving-average, and 52-week seasonal-naive baselines before complex candidates. Evaluate XGBoost and compact global GRU or LSTM candidates on the same rolling-origin windows. Use validation data to tune candidates, then evaluate the selected champion once on a held-out test partition. Preserve each completed run in `experiments/registry.json` and update `experiments/latest.json` atomically.

Record `run_id`, data version, feature set, model type, lookback, horizon, validation MAE and sMAPE, test MAE and sMAPE, alert precision, alert recall, and training timestamp. Save each candidate's best validation checkpoint with the scaler, feature list, hyperparameters, data date range, and schema version.

Use early stopping on validation MAE when the chosen framework supports it. Document optimizer and scheduler rationale, observability, environment parity, and reproducibility limits. Fixed seeds improve reproducibility but do not guarantee identical outcomes across hardware or nondeterministic operations.
