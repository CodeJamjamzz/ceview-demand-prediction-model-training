# Today

- [x] Define the Module 2 forecasting purpose, target, inputs, outputs, metrics, integration boundary, and safety constraints.
- [x] Implement weekly category preprocessing, pandemic exclusion, diagnostic review gate, economic joins, and shared historical transformations.
- [x] Review dataset assumptions and record outstanding provenance limitations in docs/data.md; this does not approve the data.
- [x] Supply verified collection metadata for the three category CSVs and approve the fixed 2023 coastal-island diagnostic before an approved dataset build; the existing export remains provisional.
- [x] Confirm data permissions, Google Trends collection consistency, privacy constraints, retention, and external artifact storage.
- [ ] Implement and evaluate last-value, seven-week moving-average, and seasonal-naive baselines using chronological rolling windows.
- [ ] Evaluate BiLSTM-only, Transformer-only, and BiLSTM + Transformer against the baselines using the initial three-seed plan.
- [x] Select PyTorch, Google Colab GPU, Weights & Biases tracking, and Google Drive artifact storage with the user.
- [x] Freeze the available provisional dataset bytes, metric definitions, and initial experiment configuration in experiments/forecasting-v1.json.
- [x] Implement shared identity/position embeddings, prediction head, and pre-normalization Transformer block; 18 synthetic checks passed.
- [ ] Configure the external runtime and exact Drive location and verify original artifact access and persistence.
- [ ] Select the production deployment path.
- [ ] Integrate a model only if it meets the documented unseen-data and alert-quality acceptance rule.

- [x] Assemble A/B/C architectures and implement the Colab notebook, seeded runner, baselines, W&B logging, and one-time test reporting; validate locally with synthetic data.
- [ ] Add the W&B key to local .env and verify online logging and Drive persistence in the approved Colab run.

- [x] Run one selected experiment per Colab session, defaulting to A; retain prior results and defer final comparison/testing until A/B/C are complete.
