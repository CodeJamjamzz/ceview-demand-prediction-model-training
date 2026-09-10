"""One-off provisional export requested by the data owner; records no approval."""

from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.preprocessing.common import (
    CATEGORIES, MARKETS, eligible, fingerprint, load_config, load_weekly,
    new_output, write_json,
)
from model.preprocessing.dataset import CORE_FEATURES, make_windows, split_calendar, transform_history


def main():
    config = load_config(ROOT / "model/preprocessing/config.json")
    config["dataset_version"] = "demand-v2-provisional"
    version = config["dataset_version"]
    destination = ROOT / "data/processed" / version
    artifact = ROOT / "artifacts" / version
    if destination.exists() or artifact.exists():
        raise ValueError("Export version already exists")
    digest, provenance = fingerprint(config, ROOT)
    weekly, audit, coverage = load_weekly(config, ROOT)
    weekly["split"] = weekly.week_start.map(split_calendar(weekly))
    weekly["category_definition_version"] = pd.NA
    arrays, manifest, excluded = make_windows(weekly)
    manifest["dataset_version"] = version
    manifest["dataset_status"] = "provisional"

    assert eligible(weekly.week_start).all()
    assert weekly.trend_observed.all()
    assert not weekly.duplicated(["market", "category", "week_start"]).any()
    assert (manifest.target_end - manifest.history_start).dt.days.eq(63 * 7).all()
    assert not ((manifest.history_start < pd.Timestamp("2020-01-01"))
                & (manifest.target_end >= pd.Timestamp("2023-01-01"))).any()
    counts = {}
    for split, bucket in arrays.items():
        counts[split] = len(bucket["X"])
        assert counts[split] > 0
        assert bucket["X"].shape == (counts[split], 52, 3)
        assert bucket["y"].shape == (counts[split], 12)
        assert np.isfinite(bucket["X"]).all() and np.isfinite(bucket["y"]).all()
        assert ((bucket["y"] >= 0) & (bucket["y"] <= 1)).all()
        rows = manifest[manifest.split.eq(split)]
        dates = weekly.loc[weekly.split.eq(split), "week_start"]
        assert rows.target_start.ge(dates.min()).all() and rows.target_end.le(dates.max()).all()
        for row in rows.iloc[[0, -1]].itertuples(index=False):
            history = weekly[weekly.market.eq(row.market) & weekly.category.eq(row.category)
                             & weekly.week_start.between(row.history_start, row.origin_week)]
            np.testing.assert_array_equal(transform_history(history)[0], bucket["X"][row.array_index])

    boundaries = {split: {"start": str(weekly.loc[weekly.split.eq(split), "week_start"].min().date()),
                          "end": str(weekly.loc[weekly.split.eq(split), "week_start"].max().date())}
                  for split in arrays}
    status = {
        "dataset_version": version, "status": "provisional", "diagnostic_approval": "pending",
        "purpose": "User-requested dataset assembly with documented collection assumptions",
        "confirmed_by_user": {"source": "Google Trends", "search_type": "Web Search",
                              "subject": "tourism destinations", "same_category_filter": True},
        "unverified": ["exact queries and category aggregation", "category definition version",
                       "collection dates and historical availability", "weekly boundary semantics",
                       "cross-country normalization compatibility"],
        "calendar_assumptions": config["calendar"], "economics_included": False,
        "fingerprint": digest, "counts": counts, "weekly_rows": len(weekly),
        "splits": boundaries, "validation": "Passed shape, finite-value, chronology, exclusion, split, and transformation-parity checks",
        "export_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with new_output(destination) as output, new_output(artifact) as metadata:
        weekly.to_csv(output / "weekly_series.csv", index=False)
        weekly.to_parquet(output / "weekly_series.parquet", index=False)
        coverage.to_csv(output / "quality_report.csv", index=False)
        write_json(output / "audit.json", audit)
        write_json(output / "dataset_status.json", status)
        core = output / "core"
        core.mkdir()
        manifest.to_csv(core / "window_manifest.csv", index=False)
        excluded.to_csv(core / "excluded_windows.csv", index=False)
        manifest.groupby(["market", "category", "split"]).size().rename("retained").to_csv(core / "retained_counts.csv")
        excluded.groupby(["market", "category", "split", "reason"]).size().rename("excluded").to_csv(core / "excluded_counts.csv")
        for split, bucket in arrays.items():
            np.savez_compressed(core / f"{split}.npz", **bucket)
        write_json(metadata / "preprocessing_config.json", {**config, "lookback": 52, "horizon": 12,
                   "excluded_interval": ["2020-01-01", "2023-01-01"], "splits": boundaries})
        write_json(metadata / "feature_schema.json", {"core": CORE_FEATURES, "dtype": "float32",
                   "target_scale": "index / 100", "target_inverse": "prediction * 100",
                   "calendar_encoding": "sin/cos of 2*pi*(ISO_week-1)/weeks_in_ISO_year"})
        write_json(metadata / "identity_vocabularies.json", {"market": MARKETS, "category": CATEGORIES})
        write_json(metadata / "scaler_parameters.json", {})
        write_json(metadata / "dataset_metadata.json", {**status, "provenance": provenance})
        text = ["# Tourism search-demand dataset", "", "Status: provisional, diagnostic approval remains pending.", "",
                "Created at the user's request using the current Sunday-start UTC calendar and no publication lag.",
                "Collection and normalization details remain unverified. No approval or metadata was fabricated.", "",
                f"Clean weekly rows: {len(weekly):,}; series: 21 (three countries and seven categories).", "",
                "Split | Examples | Target partition start | Target partition end", "--- | ---: | --- | ---"]
        text += [f"{s} | {counts[s]:,} | {boundaries[s]['start']} | {boundaries[s]['end']}" for s in arrays]
        text += ["", "Open weekly_series.csv to inspect the cleaned table. Training arrays are in core/.",
                 "Each X has 52 weeks and three numerical features; each y has 12 subsequent trend values.",
                 "Country/category IDs are separate arrays. Trends and targets are divided by 100.",
                 "All intervals overlapping 2020-2022 are excluded. Windows never bridge that gap.",
                 "Raw sources remain unchanged. GDP and exchange rates are not included.",
                 "Test data is reserved for final evaluation after model selection.",
                 "See dataset_status.json for assumptions and verification results.", ""]
        (output / "README.md").write_text("\n".join(text), encoding="utf-8")
        # Verify saved arrays and the Parquet roundtrip before publishing the export.
        assert len(pd.read_parquet(output / "weekly_series.parquet")) == len(weekly)
        for split, bucket in arrays.items():
            with np.load(core / f"{split}.npz", allow_pickle=False) as saved:
                for key in bucket:
                    np.testing.assert_array_equal(saved[key], bucket[key])
    assert fingerprint(config, ROOT)[0] == digest, "Source files changed during export"
    print(destination)
    print(status)


if __name__ == "__main__":
    main()
