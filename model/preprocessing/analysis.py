"""A fixed, retrospective diagnostic. No training arrays are created here."""

from hashlib import sha256
import json

import numpy as np
import pandas as pd

from .common import (DIAGNOSTIC_CATEGORY, DIAGNOSTIC_YEAR, MARKETS, fingerprint,
                     metadata_status, new_output, read_json, write_json)


def leader_summary(monthly):
    leaders = []
    for _, row in monthly.iterrows():
        if row.isna().any():
            leaders.append([])
        else:
            leaders.append([market for market in MARKETS
                            if np.isclose(row[market], row.max(), rtol=0, atol=1e-8)])
    longest, current = dict.fromkeys(MARKETS, 0), dict.fromkeys(MARKETS, 0)
    for group in leaders:
        for market in MARKETS:
            current[market] = current[market] + 1 if market in group else 0
            longest[market] = max(longest[market], current[market])
    complete = bool(leaders) and all(leaders)
    return {
        "monthly_leaders": leaders,
        "tie_months": sum(len(group) > 1 for group in leaders),
        "leader_set_changes": sum(a != b for a, b in zip(leaders, leaders[1:]) if a and b),
        "leading_month_share": {m: sum(m in group for group in leaders) / len(leaders)
                                if leaders else 0 for m in MARKETS},
        "longest_leading_run_months": longest,
        "persistent_leaders": [m for m in MARKETS if complete and all(m in g for g in leaders)],
    }


def diagnostic(weekly, config, root):
    _, metadata_problems, comparable = metadata_status(config, root)
    part = weekly[(weekly.category == DIAGNOSTIC_CATEGORY)
                  & (weekly.week_start.dt.year == DIAGNOSTIC_YEAR)].copy()
    start_weekday = (config["calendar"]["weekday"]
                     - (6 if config["calendar"]["date_label"] == "end" else 0)) % 7
    days = pd.date_range(f"{DIAGNOSTIC_YEAR}-01-01", f"{DIAGNOSTIC_YEAR}-12-31")
    expected = days[days.dayofweek == start_weekday]
    values = part.pivot(index="week_start", columns="market", values="trend_index")
    values = values.reindex(index=expected, columns=MARKETS)
    monthly = values.resample("MS").mean()
    missing = {m: int(values[m].isna().sum()) for m in MARKETS}
    problems = list(metadata_problems)
    if any(missing.values()):
        problems.append("Diagnostic year has missing or invalid weekly observations.")
    leaders = leader_summary(monthly) if comparable and not any(missing.values()) else None
    review_reasons = []
    if leaders and leaders["persistent_leaders"]:
        review_reasons.append("At least one country leads or ties for the lead in every month.")
    report = {
        "category": DIAGNOSTIC_CATEGORY, "year": DIAGNOSTIC_YEAR,
        "expected_weeks_per_country": len(expected), "missing_weeks": missing,
        "cross_market_comparable": comparable,
        "blocking_issues": problems, "review_reasons": review_reasons,
        "leaders": leaders,
        "interpretation": "Retrospective search-interest description, not visitor demand, a structural-shift test, or a production recommendation.",
        "monthly_assignment": "Underlying week-start month; complete weekly observations are not prorated.",
        "within_country_changes": {
            market: {
                "annual_mean": float(values[market].mean()) if values[market].notna().any() else None,
                "weekly_range": float(values[market].max() - values[market].min()) if values[market].notna().any() else None,
                "monthly_increases": int(monthly[market].diff().gt(0).sum()),
                "monthly_decreases": int(monthly[market].diff().lt(0).sum()),
            } for market in MARKETS
        },
    }
    return report, values, monthly


def run_analysis(weekly, audit, coverage, config, root):
    destination = root / "data/analysis" / config["dataset_version"]
    report, values, monthly = diagnostic(weekly, config, root)
    digest, provenance = fingerprint(config, root)
    report["fingerprint"] = digest
    report["audit"] = audit
    with new_output(destination) as output:
        write_json(output / "report.json", report)
        write_json(output / "provenance.json", provenance)
        write_json(output / "review.json", {"status": "pending", "fingerprint": digest})
        coverage.to_csv(output / "coverage.csv", index=False)
        values.to_csv(output / "weekly_comparison.csv")
        monthly.to_csv(output / "monthly_comparison.csv")
        monthly.diff().to_csv(output / "monthly_changes.csv")
        values.agg(["mean", "std", "min", "max"]).to_csv(output / "variability.csv")
        plot_diagnostic(values, output)
        issues = report["blocking_issues"] + report["review_reasons"]
        lines = ["# Coastal-island search-interest diagnostic, 2023", "",
                 report["interpretation"], "",
                 f"Expected weeks per country: {len(values)}.",
                 f"Missing weeks: {report['missing_weeks']}.", "",
                 "Calendar assumptions: " + str(config["calendar"]), "",
                 "## Review status", "", "Dataset generation requires explicit diagnostic approval.", ""]
        lines += [f"- {issue}" for issue in issues]
        if not report["cross_market_comparable"]:
            lines += ["", "Country rankings are withheld because collection metadata does not establish comparable scales."]
        if report["leaders"]:
            lines += ["", "## Leadership summary", "", "```json",
                      json.dumps(report["leaders"], indent=2), "```"]
        lines += ["", "## Within-country changes", "",
                  "Country | Mean index | Weekly range | Increasing months | Decreasing months",
                  "--- | ---: | ---: | ---: | ---:"]
        for market, stats in report["within_country_changes"].items():
            mean = f"{stats['annual_mean']:.2f}" if stats["annual_mean"] is not None else "Unavailable"
            spread = f"{stats['weekly_range']:.2f}" if stats["weekly_range"] is not None else "Unavailable"
            lines.append(f"{market} | {mean} | {spread} | {stats['monthly_increases']} | {stats['monthly_decreases']}")
        lines += ["", "Monthly changes are index-point differences, not changes in visitor counts.",
                  "Annual-mean deviations use the full diagnostic year and are not model features.", "",
                  "![Weekly indices and trailing means](weekly_trends.png)", "",
                  "![Within-country deviations](relative_interest.png)", ""]
        (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return destination, report


def plot_diagnostic(values, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, sharey=True)
    for ax, market in zip(axes, MARKETS):
        ax.plot(values.index, values[market], alpha=0.45, label="Weekly index")
        ax.plot(values.index, values[market].rolling(4, min_periods=4).mean(), label="Trailing 4 weeks")
        ax.set_ylabel(f"{market}\nIndex points")
        ax.legend(loc="upper left")
    fig.suptitle("Coastal island, 2023: supplied indices, not visitor counts")
    fig.tight_layout()
    fig.savefig(output / "weekly_trends.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    for market in MARKETS:
        ax.plot(values.index, values[market] - values[market].mean(), label=market)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set(title="Retrospective deviation from each country's own 2023 mean",
           ylabel="Index-point deviation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "relative_interest.png", dpi=150)
    plt.close(fig)


def approve(config, root, reviewer, note, acknowledge_stable=False):
    folder = root / "data/analysis" / config["dataset_version"]
    report = read_json(folder / "report.json")
    digest, _ = fingerprint(config, root)
    if report["fingerprint"] != digest:
        raise ValueError("Sources, configuration, or implementation changed; run a new analysis version")
    if report["blocking_issues"]:
        raise ValueError("Resolve diagnostic blocking issues and run a new version before approval")
    if report["review_reasons"] and not acknowledge_stable:
        raise ValueError("Stable leadership requires explicit --acknowledge-stable-leader")
    if not reviewer.strip() or not note.strip():
        raise ValueError("Reviewer and decision note are required")
    if read_json(folder / "review.json")["status"] == "approved":
        raise ValueError("This diagnostic already has an approved review")
    write_json(folder / "review.json", {
        "status": "approved", "fingerprint": digest, "reviewer": reviewer, "note": note,
        "reviewed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "stable_leader_acknowledged": acknowledge_stable,
        "report_checksum": sha256((folder / "report.json").read_bytes()).hexdigest(),
    })


def require_approval(config, root):
    folder = root / "data/analysis" / config["dataset_version"]
    review = read_json(folder / "review.json")
    report = read_json(folder / "report.json")
    digest, _ = fingerprint(config, root)
    checksum = sha256((folder / "report.json").read_bytes()).hexdigest()
    if (review.get("status") != "approved" or review.get("fingerprint") != digest
            or report.get("fingerprint") != digest or review.get("report_checksum") != checksum
            or report["blocking_issues"]
            or (report["review_reasons"] and not review.get("stable_leader_acknowledged"))):
        raise ValueError("Dataset build blocked: missing, stale, or invalid diagnostic approval")
