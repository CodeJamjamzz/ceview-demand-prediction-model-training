"""Source validation, provenance, calendar rules, and immutable output helpers."""

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

CATEGORIES = [
    "accommodation_staycation", "adventure_nature", "coastal_island",
    "culinary_gastronomy", "cultural_heritage", "theme_parks_entertainment", "urban_city",
]
MARKETS = ["JP", "KR", "US"]
ALIASES = {"JP": "JP", "JAPAN": "JP", "KR": "KR", "KOREA": "KR",
           "SOUTH KOREA": "KR", "US": "US", "USA": "US", "UNITED STATES": "US"}
EXCLUSION_START = pd.Timestamp("2020-01-01")
EXCLUSION_END = pd.Timestamp("2023-01-01")
LOOKBACK, HORIZON = 52, 12
DIAGNOSTIC_YEAR, DIAGNOSTIC_CATEGORY = 2023, "coastal_island"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path):
    config = read_json(path)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", config["dataset_version"]):
        raise ValueError("dataset_version must be a safe directory name")
    calendar = config["calendar"]
    if calendar["date_label"] not in ("start", "end"):
        raise ValueError("date_label must be start or end (inclusive date)")
    if calendar["weekday"] not in range(7) or calendar["availability_lag_days"] < 0:
        raise ValueError("Invalid calendar weekday or availability lag")
    as_of = pd.Timestamp(config["as_of"])
    if as_of.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    return config


def fingerprint(config, root):
    paths = list(config["sources"]) + [config["metadata"]]
    if config.get("economics"):
        paths.append(config["economics"])
    checksums = {name: sha256((root / name).read_bytes()).hexdigest()
                 if (root / name).is_file() else None for name in paths}
    code = {p.name: sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}
    payload = {"config": config, "sources": checksums, "implementation": code}
    digest = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload


@contextmanager
def new_output(destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Version already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".preparing-", dir=destination.parent))
    try:
        yield temporary
        if destination.exists():
            raise ValueError(f"Version already exists: {destination}")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def eligible(dates):
    """Seven-day intervals are half-open, so boundary weeks are excluded too."""
    return ~((dates < EXCLUSION_END) & (dates + pd.Timedelta(days=7) > EXCLUSION_START))


def cutoff(week, config):
    return (pd.Timestamp(week).tz_localize(config["calendar"]["timezone"])
            + pd.DateOffset(days=7 + config["calendar"]["availability_lag_days"]))


def metadata_status(config, root):
    path = root / config["metadata"]
    if not path.exists():
        return {}, ["Collection metadata missing; calendar and provenance are unverified."], False
    metadata = read_json(path)
    problems = []
    if metadata.get("calendar") != config["calendar"]:
        problems.append("Verified collection calendar must match the configuration.")
    sources = metadata.get("sources", {})
    for name in config["sources"]:
        item = sources.get(name, {})
        required = ["source", "collected_at", "query_geography", "requested_time_range",
                    "category_definition_version", "category_definitions", "query_definitions",
                    "normalization", "availability_evidence"]
        if any(not item.get(key) for key in required):
            problems.append(f"Incomplete provenance: {name}")
        for key in ("category_definitions", "query_definitions"):
            if not isinstance(item.get(key), dict) or set(item[key]) != set(CATEGORIES):
                problems.append(f"All seven {key} required: {name}")
        if item.get("observed_data_confirmed") is not True:
            problems.append(f"Observed, non-generated origin not confirmed: {name}")
        if item.get("within_series_compatible") is not True:
            problems.append(f"Within-series normalization compatibility unverified: {name}")
        try:
            if pd.Timestamp(item["collected_at"]).tzinfo is None:
                raise ValueError("timezone missing")
        except (KeyError, ValueError, TypeError):
            problems.append(f"Invalid collection timestamp: {name}")
    versions = {sources.get(name, {}).get("category_definition_version") for name in config["sources"]}
    if len(versions) != 1 or None in versions:
        problems.append("Category definitions must use one verified compatible version.")
    comparison = metadata.get("cross_market_comparison", {})
    comparable = comparison.get("verified") is True and bool(comparison.get("evidence"))
    if not comparable:
        problems.append("Cross-market scale comparability is unverified.")
    return metadata, problems, comparable


def load_weekly(config, root):
    frames, duplicate_count = [], 0
    for name in config["sources"]:
        frame = pd.read_csv(root / name)
        required = {"date", "market", *CATEGORIES}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing columns in {name}: {required - set(frame.columns)}")
        frame = frame[["date", "market", *CATEGORIES]].copy()
        duplicate_count += int(frame.duplicated().sum())
        frame = frame.drop_duplicates()
        frame["market"] = frame.market.astype(str).str.strip().str.upper().map(ALIASES)
        if frame.market.isna().any():
            raise ValueError(f"Unsupported market in {name}")
        frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
        if (frame.date.dt.dayofweek != config["calendar"]["weekday"]).any():
            raise ValueError(f"Dates disagree with configured weekly labels: {name}")
        if config["calendar"]["date_label"] == "end":
            frame["date"] -= pd.Timedelta(days=6)
        frame["source_file"] = name
        frames.append(frame)
    wide = pd.concat(frames, ignore_index=True)
    values = ["date", "market", *CATEGORIES]
    duplicate_count += int(wide.duplicated(values).sum())
    wide = wide.drop_duplicates(values)
    if wide.duplicated(["date", "market"]).any():
        raise ValueError("Conflicting market-week duplicates; select a source snapshot explicitly")
    if set(wide.market) != set(MARKETS):
        raise ValueError("All three markets are required")
    long = wide.melt(id_vars=["date", "market", "source_file"], value_vars=CATEGORIES,
                     var_name="category", value_name="raw_value").rename(columns={"date": "week_start"})
    numeric = pd.to_numeric(long.raw_value, errors="coerce")
    invalid = long.raw_value.notna() & (~np.isfinite(numeric) | ~numeric.between(0, 100))
    long["trend_index"] = numeric.mask(invalid)
    long["invalid_record"] = invalid
    long["trend_observed"] = long.trend_index.notna()
    long["cutoff"] = long.week_start.map(lambda week: cutoff(week, config))
    excluded = ~eligible(long.week_start)
    incomplete = long.cutoff > pd.Timestamp(config["as_of"])
    audit = {"exact_duplicates_removed": duplicate_count,
             "pandemic_category_rows_excluded": int(excluded.sum()),
             "incomplete_category_rows_excluded": int((~excluded & incomplete).sum()),
             "invalid_category_values": int(invalid.sum())}
    long = long.loc[~excluded & ~incomplete].copy()
    if long.empty:
        raise ValueError("No eligible completed weeks")
    calendar = pd.date_range(long.week_start.min(), long.week_start.max(), freq="7D")
    calendar = calendar[eligible(calendar)]
    grid = pd.MultiIndex.from_product([MARKETS, CATEGORIES, calendar],
                                     names=["market", "category", "week_start"])
    weekly = long.set_index(["market", "category", "week_start"]).reindex(grid).reset_index()
    for flag in ("trend_observed", "invalid_record"):
        weekly[flag] = weekly[flag].eq(True)
    weekly["cutoff"] = weekly.week_start.map(lambda week: cutoff(week, config))
    weekly["segment"] = (weekly.week_start >= EXCLUSION_END).astype(int)
    weekly = weekly.drop(columns="raw_value")
    coverage = weekly.groupby(["market", "category"]).agg(
        first_week=("week_start", "min"), last_week=("week_start", "max"),
        expected_weeks=("week_start", "size"), observed_weeks=("trend_observed", "sum"),
        invalid_values=("invalid_record", "sum"),
        zeros=("trend_index", lambda x: int(x.eq(0).sum())),
    ).reset_index()
    coverage["missing_weeks"] = coverage.expected_weeks - coverage.observed_weeks
    coverage["observed_annual_cycles_equivalent"] = coverage.observed_weeks / 52
    return weekly, audit, coverage
