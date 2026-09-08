# CeView forecasting experiments

Planned work: September 9, 2026.

Status: architecture specifications and experiment notes only. No forecasting models have been implemented or trained yet.

## Research objective

Test whether the proposed BiLSTM + Transformer hybrid predicts 12 future weeks of tourism search interest more accurately than either component alone. BiLSTM-only and Transformer-only are component ablations, not replacements chosen in advance. GRU and TCN are optional future comparisons, outside this first experiment set.

All layers and embeddings will train from scratch. A more complex model is not automatically better; results must establish whether the hybrid adds value.

## Dataset for the first experiments

Use [demand-v2-provisional](../data/processed/demand-v2-provisional/README.md).

| Item | Current value |
| --- | --- |
| Countries | Japan, South Korea, United States |
| Tourism categories | Seven, producing 21 country-category series |
| Clean weekly rows | 8,778 |
| Training examples | 4,368 |
| Validation examples | 651 |
| Held-out test examples | 651 |
| History length | 52 consecutive weeks |
| Forecast horizon | Following 12 weeks |
| Numerical features | `trend_scaled`, `week_sin`, `week_cos` |
| Target | Google Trends index divided by 100 |
| Excluded period | All weekly intervals overlapping 2020-2022 |

The examples overlap and are not independent observations. No history or target crosses the pandemic gap. Preserve the existing chronological partitions and keep the diagnostic year, 2023, in training.

The dataset remains provisional: exact collection settings, category aggregation, weekly boundaries, and cross-country normalization compatibility are not fully verified. Dataset generation did not approve the diagnostic. Resolve and document these limitations before treating results as final research evidence.

The current dataset contains no IMF, exchange-rate, or OurAirports features. Do not describe these first experiments as using all the sources listed in the research abstract. Evaluate additional sources in a separate feature experiment.

## Shared inputs and prediction head

| Component | Configuration | Purpose |
| --- | --- | --- |
| Numerical input | `[batch, 52, 3]`, float32 | Historical trend and annual calendar features |
| Country identity | Embedding with 3 entries, 4 dimensions | Learn country-specific behavior |
| Category identity | Embedding with 7 entries, 4 dimensions | Learn category-specific behavior |
| Concatenation | Repeat identity embeddings over 52 positions; concatenate with numerical features | Produces `[batch, 52, 11]` |
| Sequence summary | Global average pooling over time | Summarizes 52 encoded positions into 64 values |
| Hidden prediction layer | Dense 32, ReLU | Converts the sequence summary into forecasting features |
| Head dropout | 0.10 | Regularization |
| Output | Dense 12, linear | Direct forecast of all 12 future weeks |

Use the saved identity vocabulary order from `artifacts/demand-v2-provisional/identity_vocabularies.json`. Country/category IDs are separate NPZ arrays, not continuous numerical measurements.

Multiply outputs by 100 to report index points. Preserve raw predictions; clamp displayed predictions to 0-100 according to the integration contract. Use the same prediction treatment for every model and state whether each reported metric uses raw or clamped predictions. Use raw inverse-scaled predictions for the primary research comparison.

## Experiment A: BiLSTM only

Research question: How well does the recurrent component forecast without self-attention?

```text
Input: 52 weeks x 3 numerical features
    -> Country and category embeddings, concatenated at each week
    -> BiLSTM: 1 layer, 32 hidden units per direction, return sequences
    -> LayerNorm: 64 features, epsilon 1e-5
    -> Dropout: 0.10
    -> Global average pooling over 52 time positions
    -> Dense: 32 neurons, ReLU
    -> Dropout: 0.10
    -> Dense: 12 neurons, linear
    -> 12 weekly predictions
```

The two LSTM directions concatenate into 64 output features per week. Use standard sigmoid gates and tanh state updates. Start with zero recurrent dropout and use the explicit output dropout above. Do not carry hidden state between examples or mix series within one sequence.

## Experiment B: Transformer only

Research question: Can attention capture the useful historical relationships without recurrent processing?

```text
Input: 52 weeks x 3 numerical features
    -> Country and category embeddings, concatenated at each week
    -> Linear projection: 11 features to 64 dimensions
    -> Add learned position embeddings: 52 positions x 64 dimensions
    -> Dropout: 0.10
    -> Transformer encoder: 1 block, 2 attention heads
    -> LayerNorm: 64 features, epsilon 1e-5
    -> Global average pooling over 52 time positions
    -> Dense: 32 neurons, ReLU
    -> Dropout: 0.10
    -> Dense: 12 neurons, linear
    -> 12 weekly predictions
```

Position embeddings identify positions within the 52-week window. They serve a different purpose from the existing annual sine/cosine features, which identify calendar season.

## Experiment C: BiLSTM + Transformer

Research question: Does self-attention applied to recurrent representations improve forecasts over either component alone?

```text
Input: 52 weeks x 3 numerical features
    -> Country and category embeddings, concatenated at each week
    -> BiLSTM: 1 layer, 32 hidden units per direction, return sequences
    -> Add learned position embeddings: 52 positions x 64 dimensions
    -> Dropout: 0.10
    -> Transformer encoder: 1 block, 2 attention heads
    -> LayerNorm: 64 features, epsilon 1e-5
    -> Global average pooling over 52 time positions
    -> Dense: 32 neurons, ReLU
    -> Dropout: 0.10
    -> Dense: 12 neurons, linear
    -> 12 weekly predictions
```

No extra projection is needed between BiLSTM and Transformer because the bidirectional output already has 64 features. The starting hybrid has one recurrent layer, one Transformer block, and one hidden prediction layer. The Transformer block also contains its own feed-forward subnetwork.

The BiLSTM and encoder attention may inspect the entire historical window. No causal attention mask is required for this encoder because all 52 input weeks precede the forecast horizon. The 12 target weeks must never enter the encoder. No autoregressive decoder or teacher forcing is used.

## Exact Transformer block for B and C

Use an identical pre-normalization encoder block in both models. Let `x` be its input sequence.

```text
x
    -> LayerNorm(64, epsilon=1e-5)
    -> Multi-head self-attention: 2 heads, 32 dimensions per head
    -> Output dropout: 0.10
    -> Add x, producing u

u
    -> LayerNorm(64, epsilon=1e-5)
    -> Dense 128, GELU
    -> Dropout: 0.10
    -> Dense 64, linear
    -> Output dropout: 0.10
    -> Add u, producing the block output
```

Set attention-weight dropout to 0.10 as well. Each head uses 32-dimensional queries, keys, and values; total model width is 64. The feed-forward layers act separately at each weekly position and share weights across positions.

LayerNorm operates over the final feature dimension at each time position, not over the time or batch dimensions. Use no BatchNorm in these initial models. LayerNorm avoids dependence on batch statistics; pre-normalization has research support for improved gradient behavior at initialization. [Layer Normalization](https://arxiv.org/abs/1607.06450), [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).

Residual additions preserve a direct path through each sublayer. The attention and feed-forward design follows the Transformer building blocks, with the specified pre-normalization arrangement. [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Shared initial training settings

These settings are proposed defaults, not validated optimum values.

| Setting | Value |
| --- | --- |
| Initialization | Random, no pretrained weights |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001; exclude biases and LayerNorm parameters |
| Loss | MAE averaged across examples and all 12 horizons, using scaled targets |
| Batch size | 32 |
| Maximum epochs | 100 |
| Early-stopping monitor | Validation MAE, minimize |
| Patience | 12 epochs without improvement |
| Minimum improvement | 0.0001 on scaled validation MAE |
| Checkpoint | Save the lowest validation-MAE checkpoint and restore it |
| Gradient clipping | Global gradient norm 1.0 |
| Seeds | 42, 123, 2026 |
| Initial learning-rate schedule | Constant; do not add a scheduler only for one model |

AdamW decouples weight decay from the adaptive update. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).

Use the same data loader policy, features, stopping rules, and tuning budget for all models. Seed language, numerical, framework, and loader randomness where supported. Record software versions and hardware; seeds do not guarantee identical behavior across environments.

Run heavy training in the selected external environment, not in the IDE. Framework, external environment, tracker, and artifact storage still need selection. Notebooks should orchestrate repository functions rather than duplicate model or preprocessing logic.

## Baselines, evaluation, and model selection

- Run last-value, seven-week moving-average, and 52-week seasonal-naive baselines on the same eligible forecast origins before interpreting neural model results.
- The initial neural comparison consists of three architectures x three seeds, or nine runs.
- Report validation MAE, RMSE, and sMAPE per horizon, country, and category, plus the overall 12-horizon mean.
- Also report error in the mean predicted demand over weeks 1-4 and weeks 1-12. Error of an average is different from the average of weekly errors; label them separately.
- Document zero and near-zero target handling before calculating MAPE. Report the fraction of values included and never silently remove difficult observations. MAPE <= 15% is a research target, not a guaranteed result.
- Compare mean and standard deviation across seeds. Use mean validation MAE as the primary architecture comparison, with sMAPE and per-series results to assess tradeoffs.
- Report parameter count, best epoch, training duration, and inference time under the same hardware conditions. The hybrid has more capacity; stronger accuracy alone does not isolate the effect of combining components from the effect of additional parameters.
- Do not treat overlapping windows as independent samples in statistical significance claims. If formal significance testing is required, choose a time-dependence-aware method before using it.
- Choose the final configuration and checkpoint policy using validation only. Evaluate the selected artifact once on the held-out test set; never select a model or seed by test performance.
- Defer category-based alert and market-ranking evaluation until their integration semantics are defined. The forecasting experiment does not implement the abstract's downstream XGBoost ranking model.

## Controlled follow-up experiments

Complete the initial comparison first. Change one factor at a time from Experiment C's starting configuration; use the same three seeds and validation policy. These are optional follow-ups, not nine-run prerequisites.

| Experiment | Starting setting | Alternative | Question |
| --- | --- | --- | --- |
| Encoder depth | 1 Transformer block | 2 blocks | Does additional attention depth improve validation forecasts? |
| Dropout | 0.10 throughout | 0.20 throughout | Does stronger regularization reduce overfitting? |
| Sequence summary | Global average pooling | Learned attention pooling | Does weighting historical positions improve prediction? |

For an attention-pooling follow-up, specify the scoring layer before implementation and keep its output width at 64. A small candidate is a shared Dense 1 score per time position, softmax over the 52 positions, and a weighted sum of the encoded states. Do not interpret pooling weights as a validated causal explanation.

The previously discussed width changes, 32 versus 64 BiLSTM units per direction and two versus four attention heads, are later options. They are not part of the first comparison. Increasing BiLSTM width changes its output dimension; explicitly align the Transformer width or add a projection rather than silently changing several settings.

## Tomorrow's checklist

- [ ] Review dataset assumptions and record outstanding provenance limitations.
- [ ] Select the framework, external compute environment, tracker, and artifact destination.
- [ ] Freeze the dataset version, metric definitions, and experiment configuration.
- [ ] Implement the shared embeddings, prediction head, and pre-normalization Transformer block.
- [ ] Implement Experiments A, B, and C without changing their shared inputs and outputs.
- [ ] Check input/output shapes, finite gradients, and a small synthetic training step locally.
- [ ] Verify that targets never enter encoder inputs and that training/inference feature order matches.
- [ ] Run the three simple forecasting baselines.
- [ ] Execute the nine initial neural runs externally, saving best validation checkpoints.
- [ ] Compare validation results and per-country/category errors before optional follow-ups.
- [ ] Record the selected configuration and rationale before opening held-out test results.
- [ ] Keep weights, datasets, and tracker exports outside Git; record lightweight results in `experiments/`.

## Experiment notes and result register

No measurements have been collected. All results below remain pending.

| Experiment | Seeds | Status | Validation MAE | Validation sMAPE | Parameter count | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| A: BiLSTM only | 42, 123, 2026 | Planned | Not measured | Not measured | Not measured | Pending |
| B: Transformer only | 42, 123, 2026 | Planned | Not measured | Not measured | Not measured | Pending |
| C: BiLSTM + Transformer | 42, 123, 2026 | Planned | Not measured | Not measured | Not measured | Pending |

For each completed run, record the run ID, dataset fingerprint, architecture settings, seed, hardware/software versions, best epoch, parameter count, training time, validation metrics, artifact location, and interpretation. Record overfitting, unstable training, or weak country/category performance even when the overall average improves.

Current decisions: keep the hybrid as the proposed research model; use A and B as component ablations; start with small networks; defer optional layers and external features until the initial comparison is complete. No architecture has been selected as the empirical winner.
