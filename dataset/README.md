# Prepared forecasting dataset

Prepared datasets live here; raw source CSVs and preprocessing diagnostics remain in `data/`.

## Current version

[demand-v2-provisional](demand-v2-provisional/README.md) contains 8,778 cleaned weekly rows across Japan, South Korea, and the United States, with seven tourism categories per country (21 series).

| File | Contents |
| --- | --- |
| [weekly_series.csv](demand-v2-provisional/weekly_series.csv) | Cleaned weekly table for inspection; not separate model windows |
| [weekly_series.parquet](demand-v2-provisional/weekly_series.parquet) | Columnar version of the cleaned table |
| [core/train.npz](demand-v2-provisional/core/train.npz) | 4,368 training examples |
| [core/validation.npz](demand-v2-provisional/core/validation.npz) | 651 validation examples for development and model selection |
| [core/test.npz](demand-v2-provisional/core/test.npz) | 651 held-out examples for the selected model's final evaluation |
| [core/window_manifest.csv](demand-v2-provisional/core/window_manifest.csv) | Window dates, identities, partition, and array-row mapping |
| [dataset_status.json](demand-v2-provisional/dataset_status.json) | Dataset fingerprint, assumptions, counts, boundaries, and recorded verification |
| [audit.json](demand-v2-provisional/audit.json) and [quality_report.csv](demand-v2-provisional/quality_report.csv) | Source-quality and exclusion information |
| `core/excluded_windows.csv`, `core/excluded_counts.csv`, `core/retained_counts.csv` | Excluded windows and count summaries |

The train, validation, and test splits are NumPy NPZ files, not separate CSV files.

## Inputs and targets

- Each example uses 52 consecutive historical weeks to predict the following 12 weeks.
- Numerical feature order: `trend_scaled`, `week_sin`, `week_cos`.
- Historical trends and targets are divided by 100; multiply predictions by 100 to recover index points.
- Country and category IDs are separate arrays. Use the saved vocabulary order from `artifacts/demand-v2-provisional/identity_vocabularies.json`; verify access before training.
- No economic or flight features are included.
- All weekly intervals overlapping 2020-2022 are excluded. Windows never bridge that gap.
- Windows overlap and are not independent observations.

| Partition | Target partition start | Target partition end |
| --- | --- | --- |
| Training | 2014-12-28 | 2024-05-19 |
| Validation | 2024-05-26 | 2025-03-09 |
| Held-out test | 2025-03-16 | 2025-12-28 |

Dates bound calendar partitions, not the first and last retained forecast origins. Do not shuffle observations across partitions or use test results to choose models.

## Status and provenance

The dataset files are available, but the version remains provisional. Sunday-start labels, UTC, and zero publication lag are assumptions. Exact queries and aggregation, category definition version, collection dates and historical availability, weekly boundaries, normalization compatibility, and data governance still need evidence. Diagnostic approval remains pending.

Existing export records report successful shape, finite-value, chronology, split, pandemic-exclusion, and transformation-parity checks. See the [dataset review](../docs/data.md#dataset-assumptions-and-provenance-review) for evidence requirements and the [preprocessing contract](../docs/preprocessing.md) for build rules.

On September 10, 2026, the complete version directory moved from `data/processed/demand-v2-provisional/` to `dataset/demand-v2-provisional/`. SHA-256 checks confirmed all 13 moved files remained byte-for-byte unchanged, including the original version README and status records. This was a relocation, not regeneration, approval, or a new dataset version.

The archived one-off exporter at `artifacts/build_provisional_dataset.py` retains its original path and checksum history. Do not rerun it to relocate or overwrite this version. The normal approved build now writes to `dataset/<dataset_version>/`; preprocessing metadata remains in `artifacts/<dataset_version>/`.

Only this directory-level README is eligible for Git tracking. Generated tables, arrays, reports, and version-level files remain ignored.

## Frozen experiment reference

The existing package is now locked by [forecasting-v1](../experiments/forecasting-v1.json). Run `python -m model.experiment` from the project root to verify its file checksums without loading test arrays. See the [frozen protocol](../docs/forecasting_protocol.md) for metrics, settings, and the distinction between a frozen provisional dataset and approved provenance.
