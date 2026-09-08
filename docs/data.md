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

With three markets and roughly five years of weekly data, the expected sample volume is small. A compact model and strong baselines are required.

## Leakage and versioning rules

Use only values that were available at each forecast-origin date. Never supply future exchange rates, revised GDP values not public at the origin, or features derived from future observations. Create windows only after the chronological split.

Keep raw data out of Git. Store a version identifier, source metadata, extraction timestamp, schema version, and date coverage with each experiment. Do not commit data without confirmed license, privacy, and size approval.
