# Ceview Transformer

Ceview Transformer provides the forecasting component for Module 2, Market Radar & Notifications. It predicts the next 12 weekly Cebu travel-demand signals for each tracked source market: South Korea, Japan, and the United States.

The initial target is the weekly Google Trends `trend_index` for each of seven Cebu tourism categories in each market, producing 21 series. It is a normalized 0 to 100 search-interest proxy, not an actual visitor count. Preprocessing excludes weeks overlapping 2020 through 2022. The normal build requires a reviewed coastal-island diagnostic for 2023; the existing user-requested provisional export does not constitute diagnostic approval.

## Layout

- `docs/`: model, data, integration, training, and experimentation contracts.
- `model/`: preprocessing, shared components, A/B/C architectures, metrics, and protocol verification.
- `training/`: Colab notebook, training engine, W&B logging, and experiment orchestration.
- `data/`: local raw sources and preprocessing diagnostics, ignored by Git.
- `dataset/`: prepared versioned datasets, including cleaned tables and training/validation/test arrays; dataset files remain ignored by Git. See the [dataset information](dataset/README.md).
- `experiments/`: lightweight experiment status and append-only history.
- `tasks/`: active scoped work.
- `notes/`: temporary session context.

## Setup and training

The stack is PyTorch, Google Colab GPU, Weights & Biases tracking, and Google Drive checkpoint/result storage. The [Colab notebook](training/forecasting_colab.ipynb) implements the full experiment workflow; see [training setup](training/README.md) and add your W&B key to the local .env. Runtime access, dataset review, and external persistence verification remain pending. See the [training stack and readiness requirements](docs/training.md#selected-experiment-stack). Heavy training runs externally, never in the IDE.

The first comparison uses last-value, seven-week moving-average, and 52-week seasonal-naive baselines, followed by BiLSTM-only, Transformer-only, and BiLSTM + Transformer models across three seeds. Compare candidates on validation data; evaluate the selected artifact once on held-out test data.

## Dataset readiness

The existing `demand-v2-provisional` export contains 8,778 weekly rows across 21 series and 4,368/651/651 training/validation/test examples. Each example uses 52 weeks and three numerical features to predict 12 weeks. Export records report successful structural and preprocessing checks; no models have been trained.

The [dataset review and evidence checklist](docs/data.md#dataset-assumptions-and-provenance-review) records exact queries and aggregation, category versioning, collection dates and historical availability, weekly boundaries, normalization compatibility, and governance as unresolved. Sunday-start weeks, UTC, and zero publication lag remain assumptions. Diagnostic approval is pending. Documenting these limitations completes the review objective, not source verification or training readiness.

## Documentation

- [Model contract](docs/model.md)
- [Data contract](docs/data.md)
- [Preprocessing commands and metadata contract](docs/preprocessing.md)
- [Forecasting experiment architectures and checklist](docs/forecasting_experiments.md)
- [Frozen dataset, metrics, configuration, and shared components](docs/forecasting_protocol.md)
- [Integration contract](docs/integration.md)
- [Training contract](docs/training.md)
- [Experimentation methodology](docs/experimentation_methodology.md)
