"""Chronological windows and reusable inference feature transformations."""

import numpy as np
import pandas as pd

from .analysis import require_approval
from .common import (CATEGORIES, DIAGNOSTIC_YEAR, HORIZON, LOOKBACK, MARKETS, eligible,
                     fingerprint, metadata_status, new_output, write_json)
from .economics import join_economics, load_economics

CORE_FEATURES = ["trend_scaled", "week_sin", "week_cos"]
ECONOMIC_FEATURES = ["gdp_growth", "exchange_rate", "gdp_age_days"]
SPLITS = ["train", "validation", "test"]


def split_calendar(weekly):
    dates = pd.DatetimeIndex(sorted(weekly.week_start.unique()))
    first, second = int(len(dates) * .8), int(len(dates) * .9)
    if min(first, second - first, len(dates) - second) < HORIZON:
        raise ValueError("Insufficient weeks for chronological 80/10/10 split")
    mapping = {date: SPLITS[0 if i < first else 1 if i < second else 2]
               for i, date in enumerate(dates)}
    diagnostic = dates[dates.year == DIAGNOSTIC_YEAR]
    if not len(diagnostic) or any(mapping[date] != "train" for date in diagnostic):
        raise ValueError("The entire diagnostic year must remain in training")
    return mapping


def fit_scalers(weekly):
    train = weekly[weekly.split.eq("train")].drop_duplicates(["market", "week_start"])
    result = {}
    for feature in ECONOMIC_FEATURES:
        values = train[feature].dropna()
        if not len(values):
            raise ValueError(f"No training observations for {feature}")
        mean, std = float(values.mean()), float(values.std(ddof=0))
        result[feature] = {"mean": mean, "scale": std if std > 0 else 1.0}
    return result


def transform_history(history, scalers=None):
    """Only historical rows are required; used identically by build and inference."""
    if len(history) != LOOKBACK:
        raise ValueError("Exactly 52 history rows required")
    if history.market.nunique() != 1 or history.category.nunique() != 1:
        raise ValueError("A sequence must contain one market and category")
    market, category = history.market.iloc[0], history.category.iloc[0]
    if market not in MARKETS or category not in CATEGORIES:
        raise ValueError("Unsupported market or category")
    dates = pd.DatetimeIndex(history.week_start)
    if not eligible(dates).all() or not (dates[1:] - dates[:-1] == pd.Timedelta(days=7)).all():
        raise ValueError("History must be consecutive and cannot cross the pandemic exclusion")
    if not history.trend_observed.all() or not history.trend_index.between(0, 100).all():
        raise ValueError("History requires observed trend indices from 0 to 100")
    array = numerical_features(history, scalers)
    if not np.isfinite(array).all():
        raise ValueError("Historical features contain missing or nonfinite values")
    return array, MARKETS.index(market), CATEGORIES.index(category)


def numerical_features(history, scalers=None):
    """Row-wise transformation, reusable without recomputing overlapping windows."""
    if "cutoff" in history:
        for feature in ("gdp_available_at", "exchange_rate_available_at"):
            if feature in history and (pd.to_datetime(history[feature], utc=True)
                                       > pd.to_datetime(history.cutoff, utc=True)).any():
                raise ValueError("Economic release occurs after historical cutoff")
    iso = pd.DatetimeIndex(history.week_start).isocalendar()
    weeks_in_year = np.array([pd.Timestamp(year=int(year), month=12, day=28).isocalendar().week
                              for year in iso.year])
    phase = 2 * np.pi * (iso.week.to_numpy(dtype=float) - 1) / weeks_in_year
    columns = [history.trend_index.to_numpy(dtype=float) / 100, np.sin(phase), np.cos(phase)]
    if scalers is not None:
        for feature in ECONOMIC_FEATURES:
            params = scalers[feature]
            columns.append((history[feature].to_numpy(dtype=float) - params["mean"]) / params["scale"])
    return np.column_stack(columns).astype(np.float32)


def make_windows(weekly, scalers=None):
    result = {split: {"X": [], "y": [], "market_id": [], "category_id": []} for split in SPLITS}
    manifest, exclusions = [], []
    for (market, category), series in weekly.groupby(["market", "category"], sort=True):
        series = series.sort_values("week_start").reset_index(drop=True)
        features = numerical_features(series, scalers)
        for start in range(max(0, len(series) - LOOKBACK - HORIZON + 1)):
            window = series.iloc[start:start + LOOKBACK + HORIZON]
            history, target = window.iloc[:LOOKBACK], window.iloc[LOOKBACK:]
            split = target.split.iloc[0]
            reason = None
            if not window.week_start.diff().iloc[1:].eq(pd.Timedelta(days=7)).all():
                reason = "pandemic_gap"
            elif target.split.nunique() != 1:
                reason = "target_crosses_split"
            elif not history.trend_observed.all():
                reason = "missing_history_trend"
            elif not target.trend_observed.all():
                reason = "missing_target_trend"
            elif scalers is not None and history[ECONOMIC_FEATURES].isna().any().any():
                reason = "missing_economics"
            if reason:
                exclusions.append({"market": market, "category": category, "split": split,
                                   "reason": reason, "origin_week": history.week_start.iloc[-1]})
                continue
            array = features[start:start + LOOKBACK]
            if not np.isfinite(array).all():
                raise ValueError("Unexpected nonfinite input after missingness filtering")
            market_id, category_id = MARKETS.index(market), CATEGORIES.index(category)
            bucket = result[split]
            manifest.append({"market": market, "category": category, "split": split,
                             "array_index": len(bucket["X"]),
                             "history_start": history.week_start.iloc[0],
                             "origin_week": history.week_start.iloc[-1],
                             "forecast_origin": history.cutoff.iloc[-1],
                             "target_start": target.week_start.iloc[0],
                             "target_end": target.week_start.iloc[-1]})
            bucket["X"].append(array)
            bucket["y"].append(target.trend_index.to_numpy(dtype=np.float32) / 100)
            bucket["market_id"].append(market_id)
            bucket["category_id"].append(category_id)
    feature_count = len(CORE_FEATURES) + (len(ECONOMIC_FEATURES) if scalers is not None else 0)
    for split, bucket in result.items():
        for key in bucket:
            bucket[key] = np.asarray(bucket[key], dtype=np.int64 if key.endswith("id") else np.float32)
        bucket["X"] = bucket["X"].reshape(-1, LOOKBACK, feature_count)
        bucket["y"] = bucket["y"].reshape(-1, HORIZON)
    return (result,
            pd.DataFrame(manifest, columns=["market", "category", "split", "array_index",
                                           "history_start", "origin_week", "forecast_origin",
                                           "target_start", "target_end"]),
            pd.DataFrame(exclusions, columns=["market", "category", "split", "reason", "origin_week"]))


def build(weekly, audit, coverage, config, root):
    require_approval(config, root)
    metadata, problems, _ = metadata_status(config, root)
    if problems:
        raise ValueError("; ".join(problems))
    weekly = weekly.copy()
    weekly["split"] = weekly.week_start.map(split_calendar(weekly))
    weekly["category_definition_version"] = metadata["sources"][config["sources"][0]]["category_definition_version"]
    scalers = None
    if config.get("economics"):
        records, count = load_economics(root / config["economics"])
        audit["pandemic_economic_records_excluded"] = count
        weekly = join_economics(weekly, records)
        if all(weekly.loc[weekly.split.eq("train"), f].notna().any() for f in ECONOMIC_FEATURES):
            scalers = fit_scalers(weekly)
        else:
            audit["economic_status"] = "Skipped: no training observations for one or more economic features"
    variants = {"core": make_windows(weekly)}
    if scalers is not None:
        variants["economic"] = make_windows(weekly, scalers)
    for name, (arrays, _, _) in variants.items():
        if any(not len(arrays[s]["X"]) for s in SPLITS):
            if name == "core":
                raise ValueError("Core has an empty partition after filtering; more eligible history is required")
            audit["economic_status"] = "Insufficient windows; empty partitions are retained and not training-ready"
    version = config["dataset_version"]
    destination, artifacts = root / "dataset" / version, root / "artifacts" / version
    if destination.exists() or artifacts.exists():
        raise ValueError("Dataset or artifact version already exists")
    digest, provenance = fingerprint(config, root)
    with new_output(destination) as output, new_output(artifacts) as artifact:
        weekly.to_parquet(output / "weekly_series.parquet", index=False)
        coverage.to_csv(output / "quality_report.csv", index=False)
        write_json(output / "audit.json", audit)
        for name, (arrays, manifest, exclusions) in variants.items():
            folder = output / name
            folder.mkdir()
            manifest["dataset_version"] = version
            manifest.to_csv(folder / "window_manifest.csv", index=False)
            exclusions.to_csv(folder / "excluded_windows.csv", index=False)
            count_index = pd.MultiIndex.from_product([MARKETS, CATEGORIES, SPLITS],
                                                    names=["market", "category", "split"])
            retained = manifest.groupby(["market", "category", "split"]).size().reindex(count_index, fill_value=0).rename("retained")
            retained.to_csv(folder / "retained_counts.csv")
            if len(exclusions):
                exclusions.groupby(["market", "category", "split", "reason"]).size().rename("excluded").to_csv(folder / "excluded_counts.csv")
            for split, array in arrays.items():
                np.savez_compressed(folder / f"{split}.npz", **array)
        write_json(artifact / "preprocessing_config.json", {
            **config, "lookback": LOOKBACK, "horizon": HORIZON,
            "excluded_interval": ["2020-01-01", "2023-01-01"],
            "splits": {s: {"start": str(weekly.loc[weekly.split.eq(s), "week_start"].min()),
                           "end": str(weekly.loc[weekly.split.eq(s), "week_start"].max())} for s in SPLITS},
        })
        write_json(artifact / "feature_schema.json", {
            "core": CORE_FEATURES, "economic": CORE_FEATURES + [f"{f}_scaled" for f in ECONOMIC_FEATURES],
            "dtype": "float32", "target_scale": "index / 100", "target_inverse": "prediction * 100",
            "definitions": {
                "trend_scaled": "Observed Google Trends index divided by 100",
                "week_sin": "sin(2*pi*(ISO_week-1)/weeks_in_ISO_year)",
                "week_cos": "cos(2*pi*(ISO_week-1)/weeks_in_ISO_year)",
                "gdp_growth_scaled": "Latest eligible released GDP growth, train-standardized",
                "exchange_rate_scaled": "Mean of at least three known daily PHP/source-currency rates, train-standardized",
                "gdp_age_days_scaled": "Days since the GDP release at the historical cutoff, train-standardized",
            },
        })
        write_json(artifact / "scaler_parameters.json", scalers or {})
        write_json(artifact / "identity_vocabularies.json", {"market": MARKETS, "category": CATEGORIES})
        write_json(artifact / "dataset_metadata.json", {"fingerprint": digest, **provenance})
    return destination
