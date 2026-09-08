# Weekly preprocessing workflow

The preprocessing package handles 21 series: JP, KR, and US across seven categories. It does not train a model or change production rankings. The target is observed search interest, not visitor counts.

## Commands

Run from the repository root using Python 3.11 or later. The local `.venv` includes PyArrow; install `requirements.txt` in a fresh environment when moving the project.

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m model.preprocessing audit
python -m model.preprocessing analyze
python -m pytest -q --basetemp=.pytest_cache/local-tests
```

`audit` prints validation and coverage. `analyze` saves the fixed coastal-island, 2023 diagnostic in `data/analysis/<dataset_version>/`. Neither command creates model arrays. The supplied config assumes Sunday start labels and UTC only for provisional inspection. Those assumptions require matching source evidence before approval.

After a person reviews a complete, comparable diagnostic, record their identity and decision with the `approve` command's `--reviewer` and `--note` arguments. Stable leadership additionally requires `--acknowledge-stable-leader`. Missing provenance, invalid coverage, and unverified comparability cannot be overridden by this flag. Run `python -m model.preprocessing build` only after approval. Review records are local attestations, not authentication or access-control mechanisms.

Sources, metadata, configuration, implementation checksums, and the diagnostic report bind the approval. Any change requires a new dataset version and analysis. Existing analysis, dataset, and artifact directories cannot be overwritten. The sole permitted analysis update is recording the pending review's approval.

## Collection metadata contract

Create `data/collection_metadata.json` from actual collection records. Do not invent missing provenance. It contains:

- `calendar`: exactly the configuration's `date_label` (`start` or inclusive `end`), label `weekday` (Monday=0), IANA `timezone`, and nonnegative `availability_lag_days` after the underlying seven-day interval completes.
- `sources`: object keyed by each source path in the config. Each source contains `source`, timezone-aware `collected_at`, `query_geography`, `requested_time_range`, `category_definition_version`, `normalization`, and `availability_evidence`.
- Each source also contains `category_definitions` and `query_definitions`, objects with all seven configured category identifiers, plus `observed_data_confirmed: true` and `within_series_compatible: true` only when verified.
- `cross_market_comparison`: `verified` boolean and `evidence` explaining the common scale or supported calibration. Independent within-country Trends downloads do not establish cross-country demand volume.

The current adapter accepts one stable category-definition version and compatible snapshots per dataset. Changed definitions and overlapping conflicting snapshots require an explicit source selection or a separate version. Any calibration must happen through a separately reviewed source preparation, not an automatic transformation here.

`as_of` is a fixed, timezone-aware completion cutoff in the config, ensuring reruns do not depend on today's clock. A historical row's information cutoff is interval completion plus the documented publication lag. This adapter assumes the source contains observed, finalized weekly records with that documented availability policy; it does not reconstruct historical Google Trends vintages.

## Pandemic and missingness rules

Exclude intervals overlapping `[2020-01-01, 2023-01-01)`. Keep raw files unchanged. A Sunday-start calendar also excludes the week starting 2019-12-29. Reindex eligible calendar positions while preserving the gap. No window or fitted statistic can bridge or include that gap.

Malformed dates and conflicting duplicates stop processing. Invalid numerical trend observations become unavailable and count in quality reports. Valid zeros remain observed. No imputation runs. A missing history or label excludes the window. Window rejection reasons use first-failure precedence: pandemic gap, split crossing, missing history, missing targets, missing economics.

## Diagnostic interpretation

Analyze only `coastal_island` in 2023, across all three countries, requiring every weekly start position in that year. Assign complete weeks to their underlying start month without prorating. Do not switch the category or year after seeing results.

The report includes weekly series, trailing four-week means, deviations from each country's annual mean, monthly averages and changes, and variability. These are retrospective descriptions, never model features or proof of structural change. Missing weeks remain missing. Partial-month averages may be shown descriptively, but leadership metrics are withheld if any required weekly observation is absent.

Comparable complete data also supports monthly leader sets, ties (absolute tolerance 1e-8 index points), changes in leader sets, inclusive leading-month shares, and longest leading runs. Tied countries each receive credit, so shares can sum above one. Leading or tying every month triggers the stable-leader review. Unknown comparability blocks approval and suppresses ranking metrics.

## Optional economic records

Set `economics` to the CSV path in the config. A null value builds only the core variant. Required columns are `market`, `reference_period`, `available_at`, `indicator`, `value`, `unit`, `source`, and `collected_at`. Timestamps require explicit timezone offsets.

- GDP uses `indicator=gdp_growth`, `unit=percent`, and an additional consistent `growth_definition` across markets. Reference periods support `YYYY`, `YYYYQn`, or daily `YYYY-MM-DD`. Preserve distinct releases and revisions.
- Exchange rates use `indicator=exchange_rate` and daily `YYYY-MM-DD` periods. Supported units are `PHP/JPY`, `PHP/KRW`, `PHP/USD`, the inverse forms, and `PHP/100 JPY`, `PHP/100 KRW`, or `PHP/100 USD`.
- Exclude reference periods overlapping the pandemic interval. Post-pandemic rows cannot carry pre-pandemic GDP forward.
- At each historical cutoff, select the latest eligible GDP reference period and then its latest known revision. Preserve release time and age. Earlier unavailable GDP stays missing.
- For exchange rates, use the latest available observation per day within the week, requiring at least three distinct days, then average normalized rates. Do not use later revisions.
- Missing economics affect only the economic variant. If a feature has no training observations, skip that variant with a recorded reason. Empty economic partitions are explicitly marked not training-ready. Malformed supplied economic records fail validation rather than being silently ignored.

## Splits, features, and outputs

Assign the first floor(80% of eligible common weeks) to training, then weeks through floor(90%) to validation, and the remainder to test. Diagnostic weeks must stay in training. Full 12-week target spans must fit their partitions; inputs can use preceding history without crossing the pandemic gap. Test data remains reserved for final evaluation after model selection.

Core ordered features are trend/100, annual sine, and annual cosine. Calendar encodings use ISO year and that year's 52 or 53 weeks. Economic features append standardized GDP growth, exchange rate, and GDP age in days. Fit their population mean and standard deviation on unique market-week training rows, without category or overlapping-window duplication. Use scale 1 for constants.

`transform_history` accepts exactly 52 validated, ordered rows for a supported market/category and optional saved economic scalers. The same function builds training histories. At inference, run the same source/calendar validation and point-in-time joins first, then select the final 52 available rows. The function does not need targets. Inverse target scaling is multiplication by 100; display clamping belongs to serving.

Generated files include the clean Parquet table, quality and window-exclusion reports, per-variant manifests and NPZ arrays, and versioned configuration/schema/scaler/vocabulary/source metadata. All remain ignored by Git. No model weights, experiment scores, or champion decisions are generated.
