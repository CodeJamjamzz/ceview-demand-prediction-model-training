"""Build provisional 52-to-12 weekly windows from regional Google Trends CSVs."""

from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import CATEGORIES, eligible, write_json

ROOT = Path(__file__).resolve().parents[2]
VERSION = "philippines-regions-v1"
LOOKBACK, HORIZON = 52, 12
EXCLUDED_FILES = {
    "trends_weekly_metro_manila_by_category.csv",
    "trends_weekly_ncr_by_category.csv",
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def region_name(path):
    stem = re.sub(r"\s*\(\d+\)$", "", path.stem)
    return stem.removeprefix("trends_weekly_").removesuffix("_by_category").lower().replace("-", "_")


def read_sources(source_dir):
    frames, checksums, quality = {}, {}, {}
    for path in sorted(source_dir.glob("*.csv")):
        if path.name in EXCLUDED_FILES:
            continue
        frame = pd.read_csv(path)
        required = {"date", "market", *CATEGORIES}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing required columns in {path.name}")
        frame = frame[["date", "market", *CATEGORIES]].copy()
        frame["week_start"] = pd.to_datetime(frame.pop("date"), format="%Y-%m-%d", errors="raise")
        if (frame.week_start.dt.dayofweek != 6).any():
            raise ValueError(f"Dates are not Sunday labels in {path.name}")
        for category in CATEGORIES:
            values = pd.to_numeric(frame[category], errors="coerce")
            invalid = values.notna() & (~np.isfinite(values) | ~values.between(0, 100))
            if invalid.any():
                raise ValueError(f"Invalid trend index in {path.name}:{category}")
            frame[category] = values
        if frame.week_start.duplicated().any():
            raise ValueError(f"Duplicate weekly dates in {path.name}")
        region = region_name(path)
        if region in frames:
            raise ValueError(f"Duplicate region identity: {region}")
        frames[region] = frame.sort_values("week_start").reset_index(drop=True)
        checksums[path.name] = digest(path)
        quality[region] = {
            "source_file": path.name,
            "source_market_value": str(frame.market.iloc[0]),
            "rows": len(frame),
            "first_week": str(frame.week_start.min().date()),
            "last_week": str(frame.week_start.max().date()),
            "missing_category_values": int(frame[CATEGORIES].isna().sum().sum()),
        }
    if not frames:
        raise ValueError(f"No regional CSVs found in {source_dir}")
    return frames, checksums, quality


def make_splits(frames):
    dates = pd.DatetimeIndex(sorted(set().union(*(set(f.week_start) for f in frames.values()))))
    dates = dates[eligible(dates)]
    first, second = int(len(dates) * 0.8), int(len(dates) * 0.9)
    if min(first, second - first, len(dates) - second) < HORIZON:
        raise ValueError("Insufficient eligible weeks for chronological 80/10/10 partitions")
    parts = {date: ("train" if i < first else "validation" if i < second else "test")
             for i, date in enumerate(dates)}
    if not any(date.year == 2023 and parts[date] == "train" for date in dates):
        raise ValueError("The 2023 diagnostic year must remain in training")
    return dates, parts


def build_arrays(frames, dates, partitions):
    regions, categories = sorted(frames), CATEGORIES
    lookup = {region: frame.set_index("week_start") for region, frame in frames.items()}
    arrays = {name: {key: [] for key in ("X", "y", "market_id", "category_id")}
              for name in ("train", "validation", "test")}
    manifest = []
    excluded = {"pandemic_gap": 0, "split_crossing": 0, "missing_history": 0, "missing_target": 0}
    for region_id, region in enumerate(regions):
        frame = lookup[region]
        for category_id, category in enumerate(categories):
            values = frame[category].reindex(dates).to_numpy(dtype=np.float64)
            for start in range(max(0, len(dates) - LOOKBACK - HORIZON + 1)):
                window_dates = dates[start:start + LOOKBACK + HORIZON]
                history = values[start:start + LOOKBACK]
                target = values[start + LOOKBACK:start + LOOKBACK + HORIZON]
                split_names = {partitions[date] for date in window_dates[LOOKBACK:]}
                if not (window_dates[1:] - window_dates[:-1] == pd.Timedelta(days=7)).all():
                    excluded["pandemic_gap"] += 1
                    continue
                if len(split_names) != 1:
                    excluded["split_crossing"] += 1
                    continue
                if not np.isfinite(history).all():
                    excluded["missing_history"] += 1
                    continue
                if not np.isfinite(target).all():
                    excluded["missing_target"] += 1
                    continue
                split = split_names.pop()
                iso = window_dates[:LOOKBACK].isocalendar()
                weeks_in_year = np.array([
                    pd.Timestamp(year=int(year), month=12, day=28).isocalendar().week
                    for year in iso.year
                ])
                phase = 2 * np.pi * (iso.week.to_numpy(dtype=float) - 1) / weeks_in_year
                features = np.column_stack((history / 100, np.sin(phase), np.cos(phase))).astype(np.float32)
                bucket = arrays[split]
                index = len(bucket["X"])
                bucket["X"].append(features)
                bucket["y"].append((target / 100).astype(np.float32))
                bucket["market_id"].append(region_id)
                bucket["category_id"].append(category_id)
                manifest.append({
                    "region": region, "category": category, "split": split, "array_index": index,
                    "history_start": str(window_dates[0].date()), "origin_week": str(window_dates[LOOKBACK - 1].date()),
                    "target_start": str(window_dates[LOOKBACK].date()), "target_end": str(window_dates[-1].date()),
                })
    for split, bucket in arrays.items():
        bucket["X"] = np.asarray(bucket["X"], dtype=np.float32).reshape(-1, LOOKBACK, 3)
        bucket["y"] = np.asarray(bucket["y"], dtype=np.float32).reshape(-1, HORIZON)
        for key in ("market_id", "category_id"):
            bucket[key] = np.asarray(bucket[key], dtype=np.int64)
        if not len(bucket["y"]):
            raise ValueError(f"No complete windows in {split} partition")
    return arrays, pd.DataFrame(manifest), excluded, regions


def main():
    source_dir = ROOT / "data_regions"
    destination = ROOT / "dataset" / VERSION
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    frames, checksums, quality = read_sources(source_dir)
    dates, partitions = make_splits(frames)
    arrays, manifest, excluded, regions = build_arrays(frames, dates, partitions)
    destination.mkdir(parents=True)
    core = destination / "core"
    core.mkdir()
    for split, values in arrays.items():
        np.savez_compressed(core / f"{split}.npz", **values)
    manifest.to_csv(core / "window_manifest.csv", index=False)
    write_json(destination / "dataset_status.json", {
        "version": VERSION,
        "status": "provisional, not approved for training or external upload",
        "source_zip_sha256": digest(ROOT / "data_regions.zip"),
        "source_csv_sha256": checksums,
        "source_files_excluded": {
            name: "Conflicting alternative NCR/Metro Manila export; excluded pending source selection."
            for name in sorted(EXCLUDED_FILES)
        },
        "regions": regions,
        "categories": CATEGORIES,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "features": ["trend_scaled", "week_sin", "week_cos"],
        "target_scale": "trend index / 100",
        "calendar_assumptions": {"date_label": "Sunday-start week", "timezone": "unverified", "availability_lag": "unverified"},
        "excluded_interval": ["2020-01-01", "2023-01-01"],
        "split_boundaries": {
            name: {"start": str(min(d for d in dates if partitions[d] == name).date()),
                  "end": str(max(d for d in dates if partitions[d] == name).date())}
            for name in ("train", "validation", "test")
        },
        "examples": {name: len(values["y"]) for name, values in arrays.items()},
        "excluded_window_counts": excluded,
        "source_quality": quality,
        "unverified": [
            "Google Trends query and collection provenance",
            "category definitions and consistency",
            "historical availability and weekly timezone/boundaries",
            "within-series normalization consistency",
            "data-owner permissions, privacy, and retention",
        ],
    })
    print(json.dumps({"dataset": str(destination), "examples": {k: len(v['y']) for k, v in arrays.items()},
                      "regions": len(regions), "categories": len(CATEGORIES),
                      "excluded_windows": excluded}, indent=2))


if __name__ == "__main__":
    main()
