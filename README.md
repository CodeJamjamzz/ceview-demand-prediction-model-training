# Ceview Transformer

Ceview Transformer provides the forecasting component for Module 2, Market Radar & Notifications. It predicts the next 12 weekly Cebu travel-demand signals for each tracked source market: South Korea, Japan, and the United States.

The initial target is the weekly Google Trends `trend_index` for each of seven Cebu tourism categories in each market, producing 21 series. It is a normalized 0 to 100 search-interest proxy, not an actual visitor count. Preprocessing excludes weeks overlapping 2020 through 2022 and requires a reviewed coastal-island diagnostic for 2023 before dataset generation.

## Layout

- `docs/`: model, data, integration, training, and experimentation contracts.
- `model/`: future framework-specific model, preprocessing, training, evaluation, inference, and serialization code.
- `training/`: external-training notebooks that orchestrate repository code without duplicating it.
- `data/`: local raw data, ignored by Git.
- `dataset/`: optional curated data, ignored by default until its policy is approved.
- `experiments/`: lightweight experiment status and append-only history.
- `tasks/`: active scoped work.
- `notes/`: temporary session context.

## Setup and training

No training framework or external execution environment has been selected. Heavy training must run in the approved external environment. The first implementation milestone is a reproducible export and evaluation workflow that compares seasonal-naive, moving-average, XGBoost, and compact GRU or LSTM candidates on identical chronological test periods.

## Documentation

- [Model contract](docs/model.md)
- [Data contract](docs/data.md)
- [Preprocessing commands and metadata contract](docs/preprocessing.md)
- [Forecasting experiment architectures and checklist](docs/forecasting_experiments.md)
- [Integration contract](docs/integration.md)
- [Training contract](docs/training.md)
- [Experimentation methodology](docs/experimentation_methodology.md)
