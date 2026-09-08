from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.preprocessing.analysis import (approve, diagnostic, leader_summary,
                                          require_approval, run_analysis)
from model.preprocessing.common import (CATEGORIES, MARKETS, cutoff, eligible,
                                        load_config, load_weekly, write_json)
from model.preprocessing.dataset import (ECONOMIC_FEATURES, build, fit_scalers,
                                         make_windows, split_calendar, transform_history)
from model.preprocessing.economics import join_economics, load_economics


@pytest.fixture
def source(tmp_path):
    config = load_config(Path(__file__).parents[1] / "model/preprocessing/config.json")
    dates = pd.date_range("2015-01-04", "2025-12-28", freq="7D")
    metadata = {"calendar": config["calendar"], "sources": {},
                "cross_market_comparison": {"verified": True, "evidence": "synthetic comparable fixture"}}
    for name, market in zip(config["sources"], MARKETS):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"date": dates, "market": market})
        for category in CATEGORIES:
            frame[category] = 40 + MARKETS.index(market) * 5 + np.sin(np.arange(len(dates)))
        frame.to_csv(tmp_path / name, index=False)
        metadata["sources"][name] = {
            "source": "synthetic test only", "collected_at": "2026-01-10T00:00:00Z",
            "query_geography": market, "requested_time_range": "2015-2025",
            "category_definition_version": "fixture-v1",
            "category_definitions": dict.fromkeys(CATEGORIES, "fixture category"),
            "query_definitions": dict.fromkeys(CATEGORIES, "fixture topic"),
            "normalization": "shared fixture units", "availability_evidence": "fixture weekly cutoff",
            "observed_data_confirmed": True, "within_series_compatible": True,
        }
    write_json(tmp_path / config["metadata"], metadata)
    return tmp_path, config


def prepared(source):
    root, config = source
    weekly, audit, coverage = load_weekly(config, root)
    weekly["split"] = weekly.week_start.map(split_calendar(weekly))
    return weekly, audit, coverage


def test_pandemic_boundaries():
    dates = pd.DatetimeIndex(["2019-12-22", "2019-12-29", "2020-01-05", "2022-12-25", "2023-01-01"])
    assert eligible(dates).tolist() == [True, False, False, False, True]


def test_reshape_exclusion_and_zero(source):
    root, config = source
    path = root / config["sources"][0]
    frame = pd.read_csv(path)
    frame.loc[0, CATEGORIES[0]] = 0
    frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    frame.to_csv(path, index=False)
    weekly, audit, _ = load_weekly(config, root)
    assert audit["exact_duplicates_removed"] == 1
    assert weekly.trend_index.eq(0).sum() == 1
    assert eligible(weekly.week_start).all()
    assert weekly.groupby(["market", "week_start"]).size().eq(7).all()
    assert audit["pandemic_category_rows_excluded"] == 157 * 21


def test_conflicting_duplicates_fail(source):
    root, config = source
    path = root / config["sources"][0]
    frame = pd.read_csv(path)
    extra = frame.iloc[:1].copy()
    extra[CATEGORIES[0]] = 99
    pd.concat([frame, extra]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        load_weekly(config, root)


def test_missing_invalid_and_missing_week_retained(source):
    root, config = source
    path = root / config["sources"][0]
    frame = pd.read_csv(path)
    frame.loc[0, CATEGORIES[0]] = 101
    frame = frame.drop(index=1)
    frame.to_csv(path, index=False)
    weekly, audit, _ = load_weekly(config, root)
    assert audit["invalid_category_values"] == 1
    assert (~weekly.trend_observed).sum() == 8


def test_fixed_diagnostic_and_missing_gate(source):
    root, config = source
    weekly, _, _ = prepared(source)
    mask = weekly.market.eq("JP") & weekly.category.eq("coastal_island") & weekly.week_start.eq(pd.Timestamp("2023-01-01"))
    weekly.loc[mask, "trend_index"] = np.nan
    report, _, _ = diagnostic(weekly, config, root)
    assert report["year"] == 2023 and report["category"] == "coastal_island"
    assert report["missing_weeks"]["JP"] == 1
    assert report["blocking_issues"] and report["leaders"] is None


def test_leadership_scenarios():
    stable = pd.DataFrame({"JP": [3] * 12, "KR": [2] * 12, "US": [1] * 12})
    assert leader_summary(stable)["persistent_leaders"] == ["JP"]
    stable.loc[6:, "KR"] = 4
    result = leader_summary(stable)
    assert result["leader_set_changes"] == 1
    assert result["longest_leading_run_months"]["JP"] == 6
    assert not result["persistent_leaders"]
    stable["US"] = stable.max(axis=1)
    assert leader_summary(stable)["tie_months"] == 12
    stable.loc[0, "JP"] = np.nan
    assert leader_summary(stable)["monthly_leaders"][0] == []


def test_windows_splits_and_inference_parity(source):
    weekly, _, _ = prepared(source)
    arrays, manifest, excluded = make_windows(weekly)
    assert "pandemic_gap" in set(excluded.reason)
    assert "target_crosses_split" in set(excluded.reason)
    assert all(len(arrays[s]["X"]) for s in arrays)
    for row in manifest.iloc[::100].itertuples(index=False):
        history = weekly[weekly.market.eq(row.market) & weekly.category.eq(row.category)
                         & weekly.week_start.between(row.history_start, row.origin_week)]
        expected, market_id, category_id = transform_history(history)
        np.testing.assert_array_equal(expected, arrays[row.split]["X"][row.array_index])
        assert arrays[row.split]["market_id"][row.array_index] == market_id
        assert arrays[row.split]["category_id"][row.array_index] == category_id
        assert (row.target_end - row.history_start).days == 63 * 7
        assert weekly.loc[weekly.week_start.between(row.target_start, row.target_end), "split"].eq(row.split).all()
    again, again_manifest, _ = make_windows(weekly)
    pd.testing.assert_frame_equal(manifest, again_manifest)
    np.testing.assert_array_equal(arrays["train"]["X"], again["train"]["X"])


def test_scalers_train_only_and_constant(source):
    weekly, _, _ = prepared(source)
    for feature in ECONOMIC_FEATURES:
        weekly[feature] = 2.0
    before = fit_scalers(weekly)
    weekly.loc[weekly.split.ne("train"), ECONOMIC_FEATURES] = 10000
    assert fit_scalers(weekly) == before
    assert all(p == {"mean": 2.0, "scale": 1.0} for p in before.values())


def test_missing_economics_preserves_core_and_economic_parity(source):
    weekly, _, _ = prepared(source)
    weekly = weekly[(weekly.market == "JP") & (weekly.category == CATEGORIES[0])].copy()
    for feature in ECONOMIC_FEATURES:
        weekly[feature] = 2.0
    scalers = fit_scalers(weekly)
    core, core_manifest, _ = make_windows(weekly)
    economic, manifest, _ = make_windows(weekly, scalers)
    assert len(manifest) == len(core_manifest)
    row = manifest.iloc[0]
    history = weekly[weekly.week_start.between(row.history_start, row.origin_week)]
    np.testing.assert_array_equal(transform_history(history, scalers)[0], economic[row.split]["X"][0])
    weekly[ECONOMIC_FEATURES] = np.nan
    missing, _, excluded = make_windows(weekly, scalers)
    unchanged, _, _ = make_windows(weekly)
    assert sum(len(a["X"]) for a in missing.values()) == 0
    assert "missing_economics" in set(excluded.reason)
    np.testing.assert_array_equal(core["train"]["X"], unchanged["train"]["X"])


def test_economic_release_after_cutoff_rejected(source):
    weekly, _, _ = prepared(source)
    history = weekly[(weekly.market == "JP") & (weekly.category == CATEGORIES[0])].iloc[:52].copy()
    history["gdp_available_at"] = history.cutoff + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="after historical cutoff"):
        transform_history(history)


def test_short_history_and_diagnostic_outside_training_fail(source):
    weekly, _, _ = prepared(source)
    with pytest.raises(ValueError, match="Insufficient"):
        split_calendar(weekly[weekly.week_start >= "2025-01-01"])
    with pytest.raises(ValueError, match="diagnostic"):
        split_calendar(weekly[weekly.week_start < "2024-01-01"])


def test_week53_and_unsupported_identity(source):
    weekly, _, _ = prepared(source)
    history = weekly[(weekly.market == "JP") & (weekly.category == CATEGORIES[0])].iloc[1:53].copy()
    assert (history.week_start.dt.isocalendar().week == 53).any()
    assert np.isfinite(transform_history(history)[0]).all()
    history["market"] = "XX"
    with pytest.raises(ValueError, match="Unsupported"):
        transform_history(history)


def test_economic_availability_revisions_and_units(tmp_path):
    records = [
        ["JP", "2022Q4", "2023-01-01T00:00:00Z", "gdp_growth", 9, "percent"],
        ["JP", "2023Q1", "2023-04-01T00:00:00Z", "gdp_growth", 2, "percent"],
        ["JP", "2023Q1", "2023-05-01T00:00:00Z", "gdp_growth", 5, "percent"],
    ]
    for day in range(2, 5):
        records.append(["JP", f"2023-04-0{day}", f"2023-04-0{day}T12:00:00Z", "exchange_rate", 2, "JPY/PHP"])
    records.append(["JP", "2023-04-02", "2023-06-01T00:00:00Z", "exchange_rate", 10, "JPY/PHP"])
    data = pd.DataFrame(records, columns=["market", "reference_period", "available_at", "indicator", "value", "unit"])
    data["source"], data["collected_at"], data["growth_definition"] = "fixture", "2026-01-01T00:00:00Z", "quarterly_yoy"
    path = tmp_path / "economics.csv"
    data.to_csv(path, index=False)
    economic, count = load_economics(path)
    assert count == 1
    weekly = pd.DataFrame({"market": ["JP"] * 3,
                           "week_start": pd.to_datetime(["2023-03-19", "2023-04-02", "2023-05-07"]),
                           "cutoff": pd.to_datetime(["2023-03-26", "2023-04-09", "2023-05-14"], utc=True)})
    joined = join_economics(weekly, economic)
    assert pd.isna(joined.gdp_growth.iloc[0])
    assert joined.gdp_growth.iloc[1:].tolist() == [2, 5]
    assert joined.exchange_rate.iloc[1] == .5
    assert pd.isna(joined.exchange_rate.iloc[2])


def test_review_gate_and_stale_approval(source, monkeypatch):
    root, config = source
    weekly, audit, coverage = prepared(source)
    monkeypatch.setattr("model.preprocessing.analysis.plot_diagnostic", lambda *a: None)
    run_analysis(weekly, audit, coverage, config, root)
    with pytest.raises(ValueError, match="blocked"):
        require_approval(config, root)
    with pytest.raises(ValueError, match="Stable leadership"):
        approve(config, root, "test reviewer", "test decision")
    approve(config, root, "test reviewer", "Accept stable synthetic fixture", True)
    require_approval(config, root)
    changed = deepcopy(config)
    changed["as_of"] = "2026-09-09T00:00:00Z"
    with pytest.raises(ValueError, match="blocked"):
        require_approval(changed, root)
    with pytest.raises(ValueError, match="already exists"):
        run_analysis(weekly, audit, coverage, config, root)


def test_unverified_metadata_cannot_be_approved(source, monkeypatch):
    root, config = source
    (root / config["metadata"]).unlink()
    weekly, audit, coverage = prepared(source)
    monkeypatch.setattr("model.preprocessing.analysis.plot_diagnostic", lambda *a: None)
    _, report = run_analysis(weekly, audit, coverage, config, root)
    assert report["leaders"] is None
    with pytest.raises(ValueError, match="blocking issues"):
        approve(config, root, "reviewer", "cannot override metadata", True)


def test_build_roundtrip_core_without_economics(source, monkeypatch):
    root, config = source
    weekly, audit, coverage = prepared(source)
    monkeypatch.setattr("model.preprocessing.analysis.plot_diagnostic", lambda *a: None)
    run_analysis(weekly, audit, coverage, config, root)
    approve(config, root, "fixture reviewer", "Synthetic build verification", True)
    output = build(weekly, audit, coverage, config, root)
    saved = pd.read_parquet(output / "weekly_series.parquet")
    assert eligible(saved.week_start).all()
    with np.load(output / "core/train.npz", allow_pickle=False) as arrays:
        assert arrays["X"].shape[1:] == (52, 3)
        assert arrays["y"].shape[1:] == (12,)
    assert not (output / "economic").exists()
    with pytest.raises(ValueError, match="already exists"):
        build(weekly, audit, coverage, config, root)
