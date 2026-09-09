# Frozen initial forecasting protocol

Original version: forecasting-v1, frozen September 10, 2026. The Colab workflow uses [forecasting-v2-wandb](../experiments/forecasting-v2-wandb.json), a tracker revision requested the same day. Dataset bytes and numerical settings remain unchanged; the original JSON is preserved.

The machine-readable contract is [experiments/forecasting-v1.json](../experiments/forecasting-v1.json). It records the exact available dataset bytes, metric definitions, A/B/C architecture settings, three seeds, training rules, and validation-only selection policy. The JSON is the source of configuration for the [shared components](../model/components.py) and [metrics](../model/metrics.py).

## Dataset lock and limits

The lock covers all 13 files in `dataset/demand-v2-provisional/`, the three original source CSVs, and the archived provisional exporter. Every file has a SHA-256 checksum. It preserves the existing chronological partitions, 52-week history, 12-week target, ordered features, identity mapping, scaling, calendar assumptions, and pandemic exclusion.

The original exporter matches the checksum recorded in the dataset status. All training and validation market/category IDs were cross-checked against the window manifest and the exporter's vocabulary ordering. The original saved metadata directory still denies access and has not been verified. Its files are not included in this lock. The frozen vocabulary reflects the independently checked training/validation mapping; do not claim this verifies the inaccessible originals.

The held-out test NPZ was hashed as opaque bytes, not opened or evaluated. Freezing preserves a provisional package. It does not approve provenance, resolve collection assumptions, or authorize training. Existing governance and diagnostic blockers remain in force.

Run the read-only integrity check from the repository root:

```powershell
python -m model.experiment
```

The verifier rejects changed or missing files and paths outside the project. The protocol itself has a checksum over canonical JSON excluding its own checksum field. This detects accidental edits; it is not an authentication mechanism. Move locations or change file contents only through an explicit protocol revision. Keep the old contract for prior runs, create a new version and checksum, and rerun affected comparisons. Do not silently update this lock or overwrite the existing dataset.

## Frozen metrics

`forecast_metrics(actual_scaled, predicted_scaled)` accepts nonempty matching arrays of shape `[examples, 12]`. Targets must be in [0, 1]; predictions may extend outside that range. It rejects nonfinite values and never silently discards observations. The function receives arrays and does not load any dataset or choose an evaluation partition.

Multiply targets and predictions by 100 before scoring. Primary metrics use raw predictions without clipping. The serving contract may clamp displayed values separately.

| Metric | Definition |
| --- | --- |
| MAE | Mean absolute error in index points |
| RMSE | Square root of mean squared error in index points |
| sMAPE | Mean of `200 * abs(predicted - actual) / (abs(actual) + abs(predicted))`; a zero/zero pair contributes zero |
| MAPE | Mean of `100 * abs(predicted - actual) / abs(actual)` for targets with absolute value at least 1 index point |
| MAPE coverage | Included count divided by all values; report the included count and total as well |
| No eligible MAPE targets | Return JSON-compatible null, with zero coverage; never return a fabricated zero error |

The one-index-point MAPE threshold is a prespecified experiment convention, not a source-data verification claim or optimized cutoff.

Reports include:

- `overall`: Pool all example-horizon errors with equal weight per value.
- `per_horizon`: Twelve chronological metric records, one for each future week.
- `mean_of_horizon_metrics`: Arithmetic mean of the twelve MAE, RMSE, and sMAPE values. Mean horizon RMSE differs from pooled RMSE; keep these labels distinct. Overall MAPE uses all eligible values rather than averaging horizon MAPEs with unequal coverage.
- `demand_mean_4w` and `demand_mean_12w`: Average each example's targets and predictions over those weeks first, then score the averages. Apply the same one-point MAPE threshold to averaged targets.
- `by_market`, `by_category`, and `by_series`: The same reports for supplied integer identities. Groups include only represented identities. Overall results do not average group scores.

Summarize all three seeds using the arithmetic mean and sample standard deviation (`ddof=1`). Seed summarization and validation-only selection are implemented in training/engine.py. Do not treat overlapping windows as independent observations for significance claims. Directional, alert, and ranking metrics remain deferred until their semantics are defined.

## Training and selection rules

Use the architecture and initial training settings in the [experiment specification](forecasting_experiments.md). The frozen JSON makes the remaining implementation choices explicit:

- Use seeds 42, 123, and 2026 and one configuration per architecture. No initial hyperparameter search.
- Shuffle complete training examples with the run seed, preserving order inside every 52-week sequence and preserving split membership. Do not shuffle validation or test. Use zero loader workers and retain the final partial batch.
- Aggregate epoch MAE by total absolute error divided by the number of example-horizon values, not by an unweighted mean of batch losses.
- Use AdamW with learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.0001, no AMSGrad, and no decay on biases or LayerNorm parameters.
- Use float32 without mixed precision, constant learning rate, batch size 32, at most 100 epochs, and global gradient norm clipping at 1.0.
- Save every strictly lower validation-MAE checkpoint, keeping the earlier checkpoint on an exact tie. A small improvement still saves a checkpoint even when it does not reset early stopping.
- Reset the patience counter only when validation MAE improves by more than 0.0001 relative to the last significant best. Stop after 12 consecutive epochs without that improvement. Restore the lowest validation-MAE checkpoint.
- Seed Python, NumPy, PyTorch, and the loader. Require deterministic algorithms, disable cuDNN benchmarking and TF32, and set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. Fail rather than silently changing reproducibility settings if unsupported.
- Compare architectures by mean validation MAE across all three completed seeds. Break exact ties by mean sMAPE, then parameter count, then A/B/C order.
- A neural candidate qualifies only if its mean validation MAE strictly beats the best required baseline. Otherwise record no neural champion.
- For the selected architecture, use the prespecified seed 42 and its best validation checkpoint for the final test. Do not choose a seed using test results. Record the selection before the one-time test evaluation.

These frozen requirements are implemented in training/engine.py and orchestrated by the [Colab notebook](../training/forecasting_colab.ipynb). Real-data training has not run. PyTorch documents the attention-weight dropout behavior in [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention) and the limits and controls of [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Shared components and validation

Implemented in `model/components.py`:

| Component | Behavior |
| --- | --- |
| `SharedEmbeddings` | Append 4-dimensional country and category embeddings to each of 52 historical rows, producing 11 features |
| `LearnedPositions` | Add 52 learned 64-dimensional position embeddings and apply dropout 0.10 |
| `PredictionHead` | Mean over time, Dense 32, ReLU, dropout 0.10, linear Dense 12 |
| `PreNormTransformerBlock` | LayerNorm before two-head historical attention and before the 64/128/64 GELU feed-forward network, with dropout and residual additions |

The encoder block does not include the architecture-level final LayerNorm. Add that after the encoder stack when assembling B and C. The components do not accept target tensors, apply a causal mask, clamp predictions, or carry recurrent state. Each model must instantiate its own components; A/B/C share definitions, not trained parameter objects.

The 18 synthetic checks cover identity order and repetition, invalid inputs, position embeddings, mean pooling, unclamped output, residual identity, pre-normalization, batch independence, finite gradients, one small optimizer step, serialization, metric edge cases, and lock tampering/path validation. The actual lock check verifies 17 file checksums. No baseline or neural dataset evaluation ran.

Install the experiment dependencies alongside the existing development requirements when preparing an environment:

```powershell
python -m pip install -r requirements.txt -r training/requirements.txt
python -m pytest -q tests/test_forecasting.py -p no:cacheprovider
```

Local verification used Python 3.13.6 and PyTorch 2.11.0+cpu. The pinned experiment package versions are in [training/requirements.txt](../training/requirements.txt). Colab GPU provisioning, exact Python/CUDA/cuDNN/hardware capture, and Drive access/persistence checks remain pending. Use the same recorded external environment for every comparison run; CPU synthetic checks do not establish GPU reproducibility.
