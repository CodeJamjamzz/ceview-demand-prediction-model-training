"""Frozen research metrics: scaled inputs, raw inverse-scaled comparisons."""

import numpy as np

from .experiment import DEFAULT_PROTOCOL, load_protocol


def _scores(actual, predicted, threshold):
    error = predicted - actual
    absolute = np.abs(error)
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.divide(200 * absolute, denominator, out=np.zeros_like(absolute),
                      where=denominator != 0)
    included = np.abs(actual) >= threshold
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "smape_percent": float(smape.mean()),
        "mape_percent": (float(np.mean(100 * absolute[included] / np.abs(actual[included])))
                         if included.any() else None),
        "mape_coverage": float(included.mean()),
        "mape_included": int(included.sum()),
        "n_values": int(actual.size),
    }


def _report(actual, predicted, threshold):
    horizons = [_scores(actual[:, h], predicted[:, h], threshold)
                for h in range(actual.shape[1])]
    return {
        "overall": _scores(actual, predicted, threshold),
        "per_horizon": horizons,
        "mean_of_horizon_metrics": {
            name: float(np.mean([row[name] for row in horizons]))
            for name in ("mae", "rmse", "smape_percent")
        },
        "demand_mean_4w": _scores(actual[:, :4].mean(axis=1),
                                 predicted[:, :4].mean(axis=1), threshold),
        "demand_mean_12w": _scores(actual.mean(axis=1), predicted.mean(axis=1), threshold),
    }


def forecast_metrics(actual_scaled, predicted_scaled, *, market_id=None,
                     category_id=None, protocol_path=DEFAULT_PROTOCOL):
    """Metrics over supplied observations only; never opens dataset files.

    Grouped metrics use sample weighting within each group. No missing or
    nonfinite observations are silently discarded.
    """
    protocol = load_protocol(protocol_path)
    actual = np.asarray(actual_scaled, dtype=np.float64)
    predicted = np.asarray(predicted_scaled, dtype=np.float64)
    horizon = protocol["dataset"]["horizon"]
    if actual.ndim != 2 or actual.shape[1] != horizon or not len(actual):
        raise ValueError(f"Expected nonempty [examples, {horizon}] targets")
    if predicted.shape != actual.shape:
        raise ValueError("Prediction and target shapes must match")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite targets and predictions")
    if ((actual < 0) | (actual > 1)).any():
        raise ValueError("Targets must be scaled indices in [0, 1]")
    scale = protocol["dataset"]["target_inverse_scale"]
    actual, predicted = actual * scale, predicted * scale
    threshold = protocol["metrics"]["mape_min_abs_target_index_points"]
    result = _report(actual, predicted, threshold)
    identities = {}
    for name, supplied in (("market", market_id), ("category", category_id)):
        if supplied is None:
            continue
        ids = np.asarray(supplied)
        vocabulary = protocol["dataset"]["vocabulary"][name]
        if (ids.shape != (len(actual),) or not np.issubdtype(ids.dtype, np.integer)
                or ((ids < 0) | (ids >= len(vocabulary))).any()):
            raise ValueError(f"Invalid {name} IDs")
        identities[name] = ids
        result[f"by_{name}"] = {
            vocabulary[index]: _report(actual[ids == index], predicted[ids == index], threshold)
            for index in np.unique(ids)
        }
    if len(identities) == 2:
        market, category = identities["market"], identities["category"]
        vocabulary = protocol["dataset"]["vocabulary"]
        result["by_series"] = {}
        for m, c in sorted(set(zip(market.tolist(), category.tolist()))):
            mask = (market == m) & (category == c)
            key = f"{vocabulary['market'][m]}/{vocabulary['category'][c]}"
            result["by_series"][key] = _report(actual[mask], predicted[mask], threshold)
    return result
