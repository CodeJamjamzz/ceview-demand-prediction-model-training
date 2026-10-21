# Philippines regional dataset review

Reviewed on October 22, 2026, for `philippines-regions-v1` and protocol revision `forecasting-regions-v2-approved`.

## Collection process

The data owner confirmed that the 16 regional CSVs use the same Google Trends collection process as the approved international dataset in `forecasting-v3-approved`; the geographic location is the only collection change. The regional dataset retains the seven category identifiers and the approved weekly preprocessing, pandemic exclusion, 52-week history, and 12-week horizon. This review inherits the prior approval for Google Trends source use, Colab training, and experiment tracking.

The original regional export does not contain the exact query strings, collection timestamps, or historical revision records. The collection settings remain documented as inherited from the approved protocol, not independently reconstructed. Google Trends index levels can be normalized separately by region, so this experiment forecasts each region's own relative-interest series and does not compare or rank regional index levels.

## Fixed 2023 diagnostic

The diagnostic uses `coastal_island` for calendar year 2023 across the 16 regions selected for the frozen dataset. Both conflicting NCR/Metro Manila exports remain excluded as recorded in `dataset_status.json`.

- Each selected region has all 53 Sunday-labeled weekly positions from January 1 through December 31, 2023.
- No selected region has duplicate week labels or invalid, non-numeric, or out-of-range coastal-island values.
- The regional vocabulary has 16 unique regions and 7 unique categories.
- All 16 selected source CSV checksums match the values recorded in `dataset_status.json`.
- Train and validation region/category identity arrays match `window_manifest.csv`.
- The frozen dataset's five files pass SHA-256 verification. The held-out test archive was checksum-verified only; its arrays were not opened for this review.

Decision: approve the fixed diagnostic for within-region forecasting. This decision does not establish a common search-volume scale across regions and does not support cross-region demand ranking.
