"""Point-in-time joins for explicitly defined economic records."""

import numpy as np
import pandas as pd

from .common import ALIASES, EXCLUSION_END, EXCLUSION_START


def load_economics(path):
    records = pd.read_csv(path)
    required = {"market", "reference_period", "available_at", "indicator", "value", "unit",
                "source", "collected_at"}
    if not required.issubset(records):
        raise ValueError(f"Economic columns missing: {required - set(records)}")
    records = records.drop_duplicates().copy()
    records["market"] = records.market.astype(str).str.strip().str.upper().map(ALIASES)
    records["value"] = pd.to_numeric(records.value, errors="raise").astype(float)
    if records.market.isna().any() or not np.isfinite(records.value).all():
        raise ValueError("Invalid economic market or value")
    if not records.indicator.isin(["gdp_growth", "exchange_rate"]).all():
        raise ValueError("Economic indicators must be gdp_growth or exchange_rate")
    for column in ("available_at", "collected_at"):
        if any(pd.Timestamp(value).tzinfo is None for value in records[column]):
            raise ValueError(f"{column} requires explicit timezone offsets")
        records[column] = pd.to_datetime(records[column], utc=True)
    if records.source.isna().any() or records.unit.isna().any():
        raise ValueError("Economic source and units are required")
    periods = records.reference_period.map(parse_period)
    records["period_start"] = [p[0] for p in periods]
    records["period_end"] = [p[1] for p in periods]
    overlap = (records.period_start < EXCLUSION_END) & (records.period_end > EXCLUSION_START)
    excluded = int(overlap.sum())
    records = records.loc[~overlap].copy()
    keys = ["market", "indicator", "reference_period", "available_at"]
    if records.duplicated(keys).any():
        raise ValueError("Conflicting economic release snapshots")
    gdp = records.loc[records.indicator.eq("gdp_growth")]
    if len(gdp) and ("growth_definition" not in records or gdp.growth_definition.isna().any()
                     or gdp.growth_definition.nunique() != 1 or not gdp.unit.eq("percent").all()):
        raise ValueError("GDP requires one growth_definition across markets and unit=percent")
    rates = records.loc[records.indicator.eq("exchange_rate")]
    if not (rates.period_end - rates.period_start).eq(pd.Timedelta(days=1)).all():
        raise ValueError("Exchange rates require daily YYYY-MM-DD reference periods")
    currencies = {"JP": "JPY", "KR": "KRW", "US": "USD"}
    for i, row in rates.iterrows():
        currency = currencies[row.market]
        if row.value <= 0:
            raise ValueError("Exchange rates must be positive")
        if row.unit == f"PHP/{currency}":
            pass
        elif row.unit == f"{currency}/PHP":
            records.loc[i, "value"] = 1 / row.value
        elif row.unit == f"PHP/100 {currency}":
            records.loc[i, "value"] = row.value / 100
        else:
            raise ValueError(f"Unsupported exchange-rate unit: {row.unit}")
        records.loc[i, "unit"] = f"PHP/{currency}"
    return records, excluded


def parse_period(value):
    import re
    value = str(value)
    if re.fullmatch(r"\d{4}", value):
        start = pd.Timestamp(f"{value}-01-01")
        return start, start + pd.DateOffset(years=1)
    if re.fullmatch(r"\d{4}Q[1-4]", value):
        start = pd.Timestamp(year=int(value[:4]), month=3 * (int(value[-1]) - 1) + 1, day=1)
        return start, start + pd.DateOffset(months=3)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        start = pd.Timestamp(value)
        return start, start + pd.Timedelta(days=1)
    raise ValueError("reference_period must be YYYY, YYYYQn, or YYYY-MM-DD")


def join_economics(weekly, records):
    joined = []
    for row in weekly[["market", "week_start", "cutoff"]].drop_duplicates().itertuples(index=False):
        known = records[(records.market == row.market) & (records.available_at <= row.cutoff)]
        known = known[known.period_end <= row.cutoff.tz_localize(None)]
        # Keep this segment free of economic references from the other side of the pandemic gap.
        if row.week_start >= EXCLUSION_END:
            known = known[known.period_start >= EXCLUSION_END]
        gdp = known[known.indicator.eq("gdp_growth")].sort_values(["period_start", "available_at"])
        rates = known[known.indicator.eq("exchange_rate")
                      & known.period_start.ge(row.week_start)
                      & known.period_start.lt(row.week_start + pd.Timedelta(days=7))]
        rates = rates.sort_values("available_at").drop_duplicates("period_start", keep="last")
        item = {"market": row.market, "week_start": row.week_start,
                "gdp_growth": np.nan, "gdp_age_days": np.nan, "exchange_rate": np.nan,
                "gdp_available_at": pd.NaT, "exchange_rate_available_at": pd.NaT}
        if len(gdp):
            latest = gdp.iloc[-1]
            item.update(gdp_growth=float(latest.value), gdp_available_at=latest.available_at,
                        gdp_age_days=(row.cutoff - latest.available_at).total_seconds() / 86400)
        if len(rates) >= 3:
            item.update(exchange_rate=float(rates.value.mean()),
                        exchange_rate_available_at=rates.available_at.max())
        joined.append(item)
    return weekly.merge(pd.DataFrame(joined), on=["market", "week_start"], validate="many_to_one")
