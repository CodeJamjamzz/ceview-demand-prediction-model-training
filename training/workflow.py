"""Durable Colab orchestration, W&B results, and a one-time test ledger."""

import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import uuid

import torch

from model.architectures import Forecaster
from model.experiment import file_digest, load_protocol, verify_dataset
from .engine import (baseline_predictions, load_split, predict, score,
                     select_champion, train_one)
from .tracking import configure_wandb, log_artifact, log_report, tracked_run


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment_info(device):
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "pyarrow", "wandb", "python-dotenv")},
    }


def preflight(root, protocol_path, device):
    protocol = load_protocol(protocol_path)
    verify_dataset(root, protocol_path)
    problems = []
    dataset = protocol["dataset"]
    if dataset.get("diagnostic_approval") != "approved":
        problems.append("The frozen dataset diagnostic is not approved.")
    if dataset.get("collection_provenance_verified") is not True:
        problems.append("Verified collection provenance is not recorded.")
    if dataset.get("data_permissions_confirmed") is not True:
        problems.append("Data-owner permission for training and external use is not recorded.")
    if dataset.get("original_metadata_verified") is not True:
        problems.append("Original saved preprocessing metadata is not verified.")
    if device.type != "cuda" or not torch.cuda.is_available():
        problems.append("Select a Colab GPU runtime for the full experiment.")
    if torch.__version__.split("+")[0] != protocol["environment"]["framework"].split("==")[1]:
        problems.append("PyTorch differs from the frozen version; install training/requirements.txt and restart.")
    for name in ("numpy", "pandas", "pyarrow", "wandb", "python-dotenv"):
        if name not in protocol["environment"]:
            continue
        if importlib.metadata.version(name) != protocol["environment"][name]:
            problems.append(f"{name} differs from the frozen version; restart after dependency installation.")
    if problems:
        raise ValueError("Training preflight:\n- " + "\n- ".join(problems)
                         + "\nSee training/README.md. Do not edit approval flags without source evidence.")
    return protocol


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing experiment record differs: {path.name}")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)


def evaluate_once(ledger, identity, evaluate):
    """Claim before opening test targets; a completed report is returned on reruns."""
    ledger = Path(ledger)
    ledger.mkdir(parents=True, exist_ok=True)
    claim, result = ledger / "claim.json", ledger / "result.json"
    if claim.exists():
        if read_json(claim) != identity:
            raise ValueError("This held-out set was already claimed for a different selected artifact")
        if result.exists():
            return read_json(result)
        raise RuntimeError("Test evaluation was interrupted after its claim. Review the ledger; no automatic retest.")
    with claim.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, indent=2)
    report = evaluate()
    write_json(result, report)
    return report


def _completed(directory):
    path = directory / "result.json"
    if not path.exists():
        return None
    result = read_json(path)
    checkpoint = (directory / result["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(directory.resolve()) or file_digest(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("Saved run checkpoint is missing or changed")
    return result


def run_suite(root, protocol_path, output_root, device=None, *, experiment="A"):
    """Run only the selected architecture; finalize when all nine runs exist."""
    if experiment not in ("A", "B", "C"):
        raise ValueError("Select exactly one experiment: A, B, or C")
    root, protocol_path, output_root = Path(root).resolve(), Path(protocol_path).resolve(), Path(output_root).resolve()
    device = device or torch.device("cuda")
    protocol = preflight(root, protocol_path, device)
    settings = configure_wandb(root / ".env")
    if output_root.is_relative_to(root / "dataset") or output_root.is_relative_to(root / "data"):
        raise ValueError("Output must not be inside the source or dataset directory")
    suite = output_root / protocol["protocol_version"]
    suite.mkdir(parents=True, exist_ok=True)
    source_paths = sorted((root / "model").glob("*.py")) + sorted((root / "training").glob("*.py"))
    identity = {"protocol_sha256": protocol["protocol_sha256"],
                "environment": environment_info(device),
                "code_sha256": {p.relative_to(root).as_posix(): file_digest(p) for p in source_paths}}
    immutable_json(suite / "suite.json", identity)
    # Only train and validation are opened during development.
    train = load_split(root, protocol, "train")
    validation = load_split(root, protocol, "validation")
    group = protocol["protocol_version"] + "-" + protocol["protocol_sha256"][:12]
    baseline_path = suite / "baselines.json"
    if baseline_path.exists():
        baselines = read_json(baseline_path)
    else:
        baselines = {name: score(validation, prediction, protocol_path)
                     for name, prediction in baseline_predictions(validation["X"]).items()}
        with tracked_run(settings, suite, "validation-baselines", group, {"protocol": protocol}) as run:
            for name, report in baselines.items():
                log_report(run, f"validation/{name}", report)
            temporary = suite / "baseline-report.json"
            write_json(temporary, baselines)
            log_artifact(run, f"baselines-{run.id}", [temporary, protocol_path], "evaluation")
        write_json(baseline_path, baselines)
    architecture = experiment
    for seed in protocol["training"]["seeds"]:
        directory = suite / f"{architecture}-{seed}"
        directory.mkdir(exist_ok=True)
        result = _completed(directory)
        if result is None:
            attempt = directory / ("attempt-" + uuid.uuid4().hex[:12])
            attempt.mkdir()
            with tracked_run(settings, attempt, f"{architecture}-seed-{seed}", group,
                             {"protocol": protocol, "architecture": architecture, "seed": seed,
                              "runtime": identity["environment"], "code_sha256": identity["code_sha256"]}) as run:
                with (attempt / "epochs.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = None
                    def log_epoch(record):
                        nonlocal writer
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(record))
                            writer.writeheader()
                        writer.writerow(record)
                        stream.flush()
                        run.log(record, step=record["epoch"])
                        print(f"{architecture}/{seed} epoch {record['epoch']}: "
                              f"train MAE={record['train/mae_index']:.4f}, "
                              f"validation MAE={record['validation/mae_index']:.4f}")
                    result = train_one(architecture, seed, train, validation, protocol_path, attempt,
                                       device, log_epoch)
                result.update({"wandb_url": run.url, "wandb_run_id": run.id,
                               "checkpoint": (attempt.relative_to(directory) / "best.pt").as_posix()})
                log_report(run, "train", result["train"])
                log_report(run, "validation", result["validation"])
                run.summary.update({key: result[key] for key in (
                    "best_epoch", "epochs", "parameter_count", "training_seconds",
                    "inference_seconds_per_example", "checkpoint_sha256")})
                report_file = attempt / "metrics.json"
                write_json(report_file, result)
                log_artifact(run, f"forecast-{architecture.lower()}-{seed}-{run.id}",
                             [attempt / "best.pt", attempt / "epochs.csv", report_file, protocol_path])
            write_json(directory / "result.json", result)
        else:
            print(f"Reloaded completed {architecture}/{seed} from Drive.")
    results, missing = [], []
    for architecture in ("A", "B", "C"):
        for seed in protocol["training"]["seeds"]:
            saved = _completed(suite / f"{architecture}-{seed}")
            if saved is None:
                missing.append(f"{architecture}/{seed}")
            else:
                results.append(saved)
    if missing:
        print(f"Experiment {experiment} saved. Still pending: {', '.join(missing)}.")
        print("Only the selected experiment was run. Final comparison and test wait for all nine runs.")
        return suite
    decision = select_champion(results, baselines, protocol)
    if decision["architecture"]:
        selected = next(r for r in results if r["architecture"] == decision["architecture"]
                        and r["seed"] == decision["seed"])
        decision["checkpoint_sha256"] = selected["checkpoint_sha256"]
    immutable_json(suite / "selection.json", decision)
    if decision["architecture"]:
        directory = suite / f"{decision['architecture']}-{decision['seed']}"
        selected = _completed(directory)
        checkpoint = directory / selected["checkpoint"]
        test_relative = f"{protocol['dataset']['path']}/core/test.npz"
        ledger = output_root / "test-evaluations" / protocol["dataset"]["files_sha256"][test_relative]
        def evaluate():
            model = Forecaster(decision["architecture"], protocol_path).to(device)
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            if saved["protocol_sha256"] != protocol["protocol_sha256"]:
                raise ValueError("Checkpoint belongs to a different protocol")
            model.load_state_dict(saved["state_dict"])
            test = load_split(root, protocol, "test")
            prediction = predict(model, test, protocol["training"]["batch_size"], device)
            return score(test, prediction, protocol_path)
        test_report = evaluate_once(ledger, decision, evaluate)
        write_json(suite / "test.json", test_report)
    else:
        test_report = None
        print("No qualifying neural champion. Held-out test remains sealed.")
    # Results can be re-synced after a W&B failure without evaluating test again.
    summary_marker = suite / "summary-synced.json"
    if not summary_marker.exists():
        with tracked_run(settings, suite, "comparison-and-final-test", group, {"protocol": protocol}) as run:
            import wandb
            run.log({"architecture_comparison": wandb.Table(
                columns=list(decision["summary"][0]),
                data=[list(row.values()) for row in decision["summary"]])})
            run.summary["selection_reason"] = decision["reason"]
            run.summary["selected_architecture"] = decision["architecture"] or "none"
            if test_report is not None:
                log_report(run, "test", test_report)
            files = [suite / "selection.json", suite / "suite.json", protocol_path]
            if test_report is not None:
                files.append(suite / "test.json")
            log_artifact(run, f"comparison-{run.id}", files, "evaluation")
            summary = {"wandb_url": run.url}
        write_json(summary_marker, summary)
    return suite


def display_results(suite):
    """Notebook tables and figures use saved reports, never test arrays."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import Markdown, display
    suite = Path(suite)
    baselines = read_json(suite / "baselines.json")
    decision = read_json(suite / "selection.json") if (suite / "selection.json").exists() else None
    display(Markdown("## Validation baselines"))
    display(pd.DataFrame({name: report["overall"] for name, report in baselines.items()}).T)
    results = [read_json(path) for path in sorted(suite.glob("[ABC]-*/result.json"))]
    display(Markdown("## Training and validation results"))
    display(pd.DataFrame([{"architecture": r["architecture"], "seed": r["seed"],
                           "best_epoch": r["best_epoch"], "parameters": r["parameter_count"],
                           "train_mae": r["train"]["overall"]["mae"],
                           "validation_mae": r["validation"]["overall"]["mae"],
                           "validation_rmse": r["validation"]["overall"]["rmse"],
                           "validation_smape": r["validation"]["overall"]["smape_percent"],
                           "seconds": r["training_seconds"], "W&B": r["wandb_url"]} for r in results]))
    if not results:
        display(Markdown("No completed training runs yet."))
        return
    rows = (len(results) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.5 * rows), squeeze=False, constrained_layout=True)
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    for ax, result in zip(axes.flat, results):
        directory = suite / f"{result['architecture']}-{result['seed']}"
        curve = pd.read_csv((directory / result["checkpoint"]).parent / "epochs.csv")
        ax.plot(curve["epoch"], curve["train/mae_index"], label="Training")
        ax.plot(curve["epoch"], curve["validation/mae_index"], label="Validation")
        ax.set(title=f"{result['architecture']}, seed {result['seed']}", xlabel="Epoch", ylabel="MAE (index points)")
        ax.legend()
    display(fig)
    plt.close(fig)
    display(Markdown("## Detailed validation results"))
    metric_names = ("mae", "rmse", "smape_percent", "mape_percent", "mape_coverage")
    for group in ("market", "category", "series"):
        display(Markdown(f"### Validation by {group}"))
        display(pd.DataFrame([
            {"architecture": r["architecture"], "seed": r["seed"], group: name,
             **{key: scores["overall"][key] for key in metric_names}}
            for r in results for name, scores in r["validation"][f"by_{group}"].items()
        ]))
    display(Markdown("### Validation by forecast horizon"))
    display(pd.DataFrame([
        {"architecture": r["architecture"], "seed": r["seed"], "horizon": horizon,
         **{key: scores[key] for key in metric_names}}
        for r in results for horizon, scores in enumerate(r["validation"]["per_horizon"], 1)
    ]))
    if decision is None:
        display(Markdown("Final comparison and testing are pending until A, B, and C each have all three seeds. "
                         "Choose the next EXPERIMENT in the notebook when ready."))
        return
    display(Markdown("## Architecture comparison: mean and sample standard deviation"))
    display(pd.DataFrame(decision["summary"]))
    display(Markdown(decision["reason"]))
    test_path = suite / "test.json"
    if test_path.exists():
        report = read_json(test_path)
        display(Markdown(f"## Final test: {decision['architecture']}, seed {decision['seed']}"))
        display(pd.DataFrame([report["overall"]]))
        for group in ("market", "category", "series"):
            display(Markdown(f"### Test by {group}"))
            display(pd.DataFrame({name: r["overall"] for name, r in report[f"by_{group}"].items()}).T)
        display(Markdown("### Demand-mean forecast errors"))
        display(pd.DataFrame({name: report[name] for name in ("demand_mean_4w", "demand_mean_12w")}).T)
        horizons = pd.DataFrame(report["per_horizon"], index=range(1, 13))
        display(horizons)
        ax = horizons[["mae", "rmse"]].plot(xlabel="Forecast week", ylabel="Index points",
                                           title="Selected model: final test errors")
        display(ax.figure)
        plt.close(ax.figure)
    else:
        display(Markdown("Test was not evaluated because no neural candidate qualified."))
    if (suite / "summary-synced.json").exists():
        display(Markdown("W&B comparison: " + read_json(suite / "summary-synced.json")["wandb_url"]))
