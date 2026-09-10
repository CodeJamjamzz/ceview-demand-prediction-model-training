# Tourism search-demand dataset

Status: provisional, diagnostic approval remains pending.

Created at the user's request using the current Sunday-start UTC calendar and no publication lag.
Collection and normalization details remain unverified. No approval or metadata was fabricated.

Clean weekly rows: 8,778; series: 21 (three countries and seven categories).

Split | Examples | Target partition start | Target partition end
--- | ---: | --- | ---
train | 4,368 | 2014-12-28 | 2024-05-19
validation | 651 | 2024-05-26 | 2025-03-09
test | 651 | 2025-03-16 | 2025-12-28

Open weekly_series.csv to inspect the cleaned table. Training arrays are in core/.
Each X has 52 weeks and three numerical features; each y has 12 subsequent trend values.
Country/category IDs are separate arrays. Trends and targets are divided by 100.
All intervals overlapping 2020-2022 are excluded. Windows never bridge that gap.
Raw sources remain unchanged. GDP and exchange rates are not included.
Test data is reserved for final evaluation after model selection.
See dataset_status.json for assumptions and verification results.
