# Integration contract

## Scope

The forecasting service replaces the current prompt-based 12-week forecast only. It does not replace economic viability, ranking, notification delivery, or persistence.

Economic viability remains a separate score. The existing market score remains `0.40 × demand + 0.35 × seasonality + 0.25 × economic viability`. A demand alert occurs when four-week predicted demand exceeds `1.2 × rolling_7d_avg`.

## Response shape

Return exactly 12 chronological weekly values, followed by the derived means and immutable model identity.

```json
{
  "weekly_forecasts": [73.1, 72.4, 71.0, 69.5, 68.2, 67.5, 67.1, 66.8, 66.5, 66.3, 66.1, 65.9],
  "predicted_demand_4w": 71.5,
  "predicted_demand_12w": 68.4,
  "source": "nn",
  "model_version": "gru-v1"
}
```

The values in this example are illustrative only. `predicted_demand_4w` is the mean of weeks 1 through 4, and `predicted_demand_12w` is the mean of all 12 weeks.

## Compatibility and operations

Preprocessing now produces category-specific histories and targets for 21 market-category series. The response above remains a market-level integration contract, not an implemented category aggregation. A later integration decision must define category selection or aggregation before connecting these datasets to production rankings. This preprocessing change does not modify scores or recommendation logic.

The service must verify the required 52-week history, compatible feature-schema version, scaler, and model artifact before inference. Training and inference must share the same validated preprocessing and postprocessing implementation. Load model artifacts through the serving lifecycle, not per request, subject to the selected deployment environment and device constraints.

On missing history, missing required features, incompatible schema, or unavailable artifact, return a clear service error and no fabricated values. Consumers validate their inputs and handle service errors through the application's availability behavior. Store model version, feature-schema version, origin date, input freshness, missing features, raw prediction, displayed prediction, and actual outcome when available.
