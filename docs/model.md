# Model contract

## Identity and purpose

Module 2, Market Radar & Notifications, forecasts Cebu travel-demand interest for South Korea, Japan, and the United States. At the end of forecast-origin week `t`, it predicts 12 weekly values for `t+1` through `t+12`.

The first model target is `trend_index`, the weekly Google Trends index for travel interest in Cebu. It ranges from 0 to 100 and measures relative search interest. It does not measure actual visitor arrivals. The model supports planning and surge detection, not certain visitor-count predictions.

The model replaces only the current prompt-based 12-week forecast. Economic viability, market ranking, persistence, notifications, and alert delivery remain outside the model.

## Inputs and preprocessing

Each sample represents one `(market, category, forecast_origin_week)` pair across three markets and seven categories. The implemented preprocessing uses a 52-week history and 12-week target, excluding intervals overlapping 2020 through 2022. Other window lengths require a separate experiment and preprocessing version.

- Historical target sequence: weekly `trend_index` values ending at the origin week.
- Core numeric features: `trend_scaled`, `week_sin`, and `week_cos`.
- Optional economic variant: standardized `gdp_growth`, `exchange_rate`, and `gdp_age_days`, using values available at each historical cutoff.
- Calendar features: week-of-year sine and cosine encodings.
- Market and category identity: saved integer vocabularies for future embedding or one-hot use.
- Optional historical features: holidays, school breaks, tourism events, flight seats, frequency, fares, and direct-flight availability, only when values were known at the origin date and are dependable.

Sort each market-category series by date, remove exact duplicates, and preserve missing dates. Exclude incomplete windows without imputation. Scale trends and targets by dividing by 100; fit economic scalers on unique training-period market-week rows only. GDP may be carried forward only from an eligible value publicly known at that week. See [preprocessing](preprocessing.md).

## Output and fallback behavior

The inference result contains 12 chronological, inverse-scaled weekly predictions. Clamp displayed `trend_index` predictions to the inclusive 0 to 100 range. Derive four-week and 12-week demand from the corresponding means.

If the required historical window, feature schema, or compatible artifact is unavailable, return a clear service error. Do not fabricate a forecast. The application then applies its established availability behavior.

## Candidate architecture and non-goals

Start with one compact global GRU or LSTM across all three markets: one recurrent layer with 32 to 64 hidden units, a small dense head, and direct 12-value multi-horizon output. The architecture remains a candidate, not a chosen champion, until it beats the required baselines on unseen rolling test periods.

Do not train a large network per market. Do not use historical Groq or Gemini forecasts as labels. Do not replace the separate economic-viability score, which uses GDP, forex, flights, and distance.

## Score interpretation

The 12 output values are predicted Google Trends index points. They are not probabilities, calibrated confidence, actual visitor counts, or proof that a surge will occur. Forecast quality is assessed with error and alert metrics after actual values become available.
