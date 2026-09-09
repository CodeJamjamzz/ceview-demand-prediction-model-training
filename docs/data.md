# Data contract

## Training unit and target

Build one sample for each valid `(market, category, forecast_origin_week)` pair, using one stable category-definition version. A sample contains 52 consecutive historical weekly observations and the next 12 observed weekly values as labels. Exclude intervals overlapping 2020 through 2022 without joining the histories across the gap. See [preprocessing](preprocessing.md) for the implemented contract.

```text
market: korea
category: coastal_island
origin_week: 2025-06-08
history_trend: [52 values ending on 2025-06-08]
known_numeric_features: [trend_scaled, week_sin, week_cos]
calendar_features: [week_sin, week_cos]
target: [trend at +1, trend at +2, ..., trend at +12]
```

## Sources and limitations

- Google Trends supplies the initial demand proxy. Its normalized index is not an arrival count, and separately downloaded queries may not be directly comparable.
- GDP and exchange-rate data supply contextual features. Record source, retrieval date, coverage, revision policy, and publication date where available.
- The implemented source adapter reads the three wide category CSVs in `data/`. Collection metadata must establish provenance, calendar boundaries, and normalization compatibility. `tbl_market_signal_record` is a possible later adapter, not a required input to this implementation.
- Arrivals or bookings may become the future production target when reliable market-level history is approved. Google Trends can then remain an input feature.

The current export contains 21 series with a pandemic gap and overlapping windows. Use the recorded coverage below rather than treating the sample count as independent observations. Compact models and strong baselines are required.

## Leakage and versioning rules

Use only values that were available at each forecast-origin date. Never supply future exchange rates, revised GDP values not public at the origin, or features derived from future observations. Create windows only after the chronological split.

Keep raw data out of Git. Store a version identifier, source metadata, extraction timestamp, schema version, and date coverage with each experiment. Do not commit data without confirmed license, privacy, and size approval.

## Dataset assumptions and provenance review

Reviewed on September 9, 2026. This review records available evidence and remaining limitations; it does not approve collection provenance or the diagnostic.

The existing export is named `demand-v2-provisional`. Its local `dataset_status.json` records user confirmation of Google Trends, Web Search, tourism destinations, and use of the same category filter. These confirmations do not establish the exact query settings, aggregation method, or shared normalization.

| Item | Recorded value |
| --- | --- |
| Source files | `data/trends_weekly_japan_by_category.csv`, `data/trends_weekly_korea_by_category.csv`, `data/USA_trends_weekly_by_category.csv` |
| Coverage | Japan, South Korea, United States; seven categories; 21 series |
| Clean weekly rows | 8,778 |
| History and target | 52 consecutive historical weeks; next 12 weekly observations |
| Ordered numerical features | `trend_scaled`, `week_sin`, `week_cos` |
| Scaling | Historical trends and targets divided by 100 |
| Identities | Country and category IDs in separate arrays, using saved vocabulary order |
| Additional sources | No IMF, exchange-rate, or OurAirports features in this export |
| Calendar assumptions | Sunday-start labels, UTC, zero publication lag |
| Exclusion | Every weekly interval overlapping 2020-2022; no window bridges the gap |
| Dataset fingerprint reported by export | `297c228ae6e97a2a8a832345776ac98a435b2f1864830d9ab642e507e08727a8` |

| Partition | Examples | Target partition start | Target partition end |
| --- | ---: | --- | --- |
| Training | 4,368 | 2014-12-28 | 2024-05-19 |
| Validation | 651 | 2024-05-26 | 2025-03-09 |
| Held-out test | 651 | 2025-03-16 | 2025-12-28 |

These dates bound calendar partitions, not the first and last retained forecast origins. The partitions preserve chronological order and keep 2023 in training. Overlapping windows are not independent observations. Keep held-out test data sealed for the selected artifact's final evaluation.

The export status records successful shape, finite-value, chronology, split, pandemic-exclusion, and transformation-parity checks. The audit records 3,297 pandemic category rows excluded, zero exact duplicates removed, zero incomplete category rows excluded, and zero invalid category values. This documentation review read existing reports; it did not rerun those checks or evaluate test targets.

### Outstanding evidence

| Unverified item | Why it matters | Evidence needed to close it |
| --- | --- | --- |
| Exact queries and aggregation | Query wording, topic IDs, and aggregation can change what a category measures. | Original terms or topic IDs, geographic and category filters, requested date ranges, search settings, and aggregation procedure for each export. |
| Category definition version | Definition changes can create artificial changes over time. | A dated mapping of all seven categories to their queries, with evidence of consistent use across countries and dates. |
| Collection dates and historical availability | Current exports may not represent information available at past forecast origins. | Timezone-aware collection timestamps, release lag, revision or snapshot policy, and evidence that rows are observed data rather than forecasts. |
| Weekly boundaries | Start/end semantics, timezone, and lag affect cutoffs and exclusion boundaries. | Source documentation confirming label semantics, weekday, timezone, and availability after each weekly interval. |
| Within-series and cross-country scale compatibility | Independently normalized series do not establish comparable demand levels. | Collection-window and normalization records; evidence of within-series consistency and a supported common scale or separately reviewed calibration across countries. |
| Data governance | Technical checks do not establish permission to train, upload, retain, or share data. | Data-owner confirmation of source permissions, collection consistency, privacy constraints, retention, and dataset versioning. |

The data collector or owner must supply these records in `data/collection_metadata.json` using the [collection metadata contract](preprocessing.md#collection-metadata-contract). Do not mark unknown fields verified or infer calibration from the observed rankings.

### Diagnostic status and completion boundary

The fixed coastal-island diagnostic for 2023 reports all 53 expected weekly positions for each country under the provisional calendar. Its report marks cross-market comparability false and withholds leadership metrics. `data/analysis/demand-v1/review.json` remains pending. Stable leadership, if established after comparability verification, requires an explicit review decision.

The September 8 provisional export was assembled at the user's request despite unresolved collection assumptions. That export did not approve the diagnostic or replace the normal approval gate. Before an approved build, supply verified metadata, rerun analysis under the documented versioning rules, and obtain the required recorded review. Resolve provenance and governance blockers before training; do not present provisional results as final research evidence.

Objective 1 is complete as an assumptions-and-limitations review. Source verification and diagnostic approval remain open. The saved vocabulary and schema under `artifacts/demand-v2-provisional/` also require an access check before external execution: the documentation inspection encountered an access-denied error and could not verify their contents.

Evidence records, stored locally and ignored by Git: `dataset/demand-v2-provisional/README.md`, `dataset_status.json`, and `audit.json` in that directory; `data/analysis/demand-v1/report.json` and `review.json`. This tracked review preserves their reported status for readers without the local files.
