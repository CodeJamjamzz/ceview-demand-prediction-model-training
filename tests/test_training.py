from contextlib import contextmanager
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch
from torch import nn

from model.architectures import Forecaster
from model.experiment import file_digest, load_protocol, protocol_digest
from training.engine import (EarlyStopping, baseline_predictions, make_optimizer,
                             select_champion, train_one)
from training import workflow
from training.tracking import configure_wandb

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def synthetic(n):
    rng = np.random.default_rng(42)
    return {"X": rng.random((n, 52, 3), dtype=np.float32),
            "y": rng.random((n, 12), dtype=np.float32),
            "market_id": np.arange(n, dtype=np.int64) % 3,
            "category_id": np.arange(n, dtype=np.int64) % 7}


@pytest.fixture
def small_protocol(tmp_path):
    protocol = load_protocol(ROOT / "experiments/forecasting-v2-wandb.json")
    protocol["training"]["max_epochs"] = 2
    protocol["training"]["batch_size"] = 2
    protocol["protocol_sha256"] = protocol_digest(protocol)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


@pytest.mark.parametrize("architecture", ["A", "B", "C"])
def test_architecture_gradients_and_batch_independence(architecture):
    model = Forecaster(architecture).eval()
    data = synthetic(3)
    history, market, category = [torch.from_numpy(data[k]) for k in ("X", "market_id", "category_id")]
    prediction = model(history, market, category)
    assert prediction.shape == (3, 12)
    separately = torch.cat([model(history[i:i+1], market[i:i+1], category[i:i+1]) for i in range(3)])
    torch.testing.assert_close(prediction, separately, atol=2e-6, rtol=2e-5)
    prediction.abs().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_optimizer_excludes_lstm_biases_and_layernorm():
    model = Forecaster("C")
    optimizer = make_optimizer(model, load_protocol()["training"])
    no_decay = {id(p) for group in optimizer.param_groups if group["weight_decay"] == 0 for p in group["params"]}
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            assert (id(parameter) in no_decay) == ("bias" in name or isinstance(module, nn.LayerNorm))


def test_early_stop_saves_small_improvements_without_resetting_patience():
    state = EarlyStopping(2, .0001)
    assert state.update(.2) == (True, False)
    assert state.update(.19995) == (True, False)
    assert state.update(.19994) == (True, True)
    state = EarlyStopping(2, .0001)
    state.update(.2)
    state.update(.19995)
    assert state.update(.1998) == (True, False)


def test_baseline_horizon_alignment():
    x = np.zeros((1, 52, 3), dtype=np.float32)
    x[0, :, 0] = np.arange(52) / 100
    values = baseline_predictions(x)
    np.testing.assert_array_equal(values["seasonal_naive_52"], x[:, :12, 0])
    np.testing.assert_allclose(values["last_value"], .51)
    np.testing.assert_allclose(values["moving_average_7"], .48)


@pytest.mark.parametrize("architecture", ["A", "B", "C"])
def test_small_training_restores_best_checkpoint(architecture, small_protocol, tmp_path):
    records = []
    result = train_one(architecture, 42, synthetic(3), synthetic(2), small_protocol, tmp_path,
                       torch.device("cpu"), records.append)
    saved = torch.load(tmp_path / "best.pt", weights_only=True)
    assert saved["epoch"] == min(records, key=lambda row: row["validation/mae_scaled"])["epoch"]
    assert result["validation"]["overall"]["mae"] == pytest.approx(saved["validation_mae_scaled"] * 100)
    assert len(records) == 2
    assert result["checkpoint_sha256"] == file_digest(tmp_path / "best.pt")


def comparison_rows():
    return [{"architecture": architecture, "seed": seed, "parameter_count": 20,
             "validation": {"overall": {"mae": 1, "rmse": 2, "smape_percent": 3}}}
            for architecture in ("A", "B", "C") for seed in (42, 123, 2026)]


def test_selection_requires_nine_and_never_uses_test():
    baselines = {name: {"overall": {"mae": 2}} for name in load_protocol()["selection"]["baseline_names"]}
    rows = comparison_rows()
    for row in rows:
        row["test"] = {"mae": -999 if row["architecture"] == "C" else 999}
    selected = select_champion(rows, baselines, load_protocol())
    assert selected["architecture"] == "A" and selected["seed"] == 42
    with pytest.raises(ValueError, match="nine"):
        select_champion(rows[:-1], baselines, load_protocol())
    for report in baselines.values():
        report["overall"]["mae"] = .5
    assert select_champion(rows, baselines, load_protocol())["architecture"] is None


def test_test_ledger_caches_and_blocks_changed_selection(tmp_path):
    calls = []
    def evaluate():
        calls.append(1)
        return {"mae": 1}
    assert workflow.evaluate_once(tmp_path, {"checkpoint": "a"}, evaluate) == {"mae": 1}
    assert workflow.evaluate_once(tmp_path, {"checkpoint": "a"}, evaluate) == {"mae": 1}
    assert calls == [1]
    with pytest.raises(ValueError, match="different"):
        workflow.evaluate_once(tmp_path, {"checkpoint": "b"}, evaluate)


def test_interrupted_test_cannot_automatically_repeat(tmp_path):
    def fail():
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        workflow.evaluate_once(tmp_path, {}, fail)
    with pytest.raises(RuntimeError, match="no automatic retest"):
        workflow.evaluate_once(tmp_path, {}, lambda: pytest.fail("Retested"))


def test_credentials_never_returned_or_printed(tmp_path, monkeypatch, capsys):
    for key in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / ".env"
    path.write_text("WANDB_API_KEY=synthetic-test-key\nWANDB_PROJECT=test-project\n")
    assert configure_wandb(path) == {"project": "test-project", "entity": None}
    assert "synthetic-test-key" not in capsys.readouterr().out
    monkeypatch.delenv("WANDB_API_KEY")


def test_notebook_is_valid_clear_and_compilable():
    notebook = nbformat.read(ROOT / "training/forecasting_colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, "<colab-cell>", "exec")
    assert any("display_results(suite)" in cell.source for cell in notebook.cells)
    assert any('EXPERIMENT = "A"' in cell.source for cell in notebook.cells)
    assert any("experiment=EXPERIMENT" in cell.source for cell in notebook.cells)


def test_existing_provisional_dataset_fails_preflight():
    with pytest.raises(ValueError, match="diagnostic is not approved"):
        workflow.preflight(ROOT, ROOT / "experiments/forecasting-v2-wandb.json", torch.device("cpu"))


def test_full_synthetic_workflow_resumes_without_retesting(tmp_path, monkeypatch):
    protocol = load_protocol(ROOT / "experiments/forecasting-v2-wandb.json")
    protocol["training"]["max_epochs"] = 1
    protocol["training"]["batch_size"] = 2
    protocol["dataset"]["path"] = "dataset/synthetic"
    folder = tmp_path / "dataset/synthetic/core"
    folder.mkdir(parents=True)
    protocol["dataset"]["files_sha256"] = {}
    for split in ("train", "validation", "test"):
        path = folder / f"{split}.npz"
        np.savez(path, **synthetic(2))
        protocol["dataset"]["files_sha256"][path.relative_to(tmp_path).as_posix()] = file_digest(path)
        protocol["dataset"]["counts"][split] = 2
    protocol["protocol_sha256"] = protocol_digest(protocol)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    # Only this synthetic test replaces external GPU/provenance and network checks.
    monkeypatch.setattr(workflow, "preflight", lambda *args: load_protocol(protocol_path))
    monkeypatch.setattr(workflow, "configure_wandb", lambda *args: {})
    monkeypatch.setattr(workflow, "environment_info", lambda *args: {"device": "synthetic-cpu"})
    logs, artifacts = [], []
    class FakeRun:
        id = "synthetic"
        url = "https://example.invalid/synthetic"
        summary = {}
        def log(self, value, **kwargs):
            logs.append(value)
    @contextmanager
    def fake_run(*args, **kwargs):
        yield FakeRun()
    monkeypatch.setattr(workflow, "tracked_run", fake_run)
    monkeypatch.setattr(workflow, "log_report", lambda *args: None)
    monkeypatch.setattr(workflow, "log_artifact", lambda run, name, paths, *args: artifacts.extend(paths))
    # Deliberately poor synthetic baselines ensure the final-test branch is exercised.
    monkeypatch.setattr(workflow, "baseline_predictions",
                        lambda x: {name: np.full((len(x), 12), 100.)
                                   for name in protocol["selection"]["baseline_names"]})
    original_load = workflow.load_split
    splits = []
    def counted_load(root, config, split):
        splits.append(split)
        return original_load(root, config, split)
    monkeypatch.setattr(workflow, "load_split", counted_load)
    output = tmp_path / "output"
    started = []
    original_train = workflow.train_one
    def counted_train(architecture, seed, *args, **kwargs):
        started.append((architecture, seed))
        return original_train(architecture, seed, *args, **kwargs)
    monkeypatch.setattr(workflow, "train_one", counted_train)
    suite = workflow.run_suite(tmp_path, protocol_path, output, torch.device("cpu"))
    assert started == [("A", seed) for seed in (42, 123, 2026)]
    assert len(list(suite.glob("[ABC]-*/result.json"))) == 3
    assert not list(suite.glob("B-*")) and not list(suite.glob("C-*"))
    assert not (suite / "selection.json").exists()
    assert not (suite / "test.json").exists()
    assert "test" not in splits

    import matplotlib
    matplotlib.use("Agg")
    import IPython.display
    displayed = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    workflow.display_results(suite)
    assert len(displayed) > 5
    assert any("pending until" in getattr(item, "data", "") for item in displayed if hasattr(item, "data"))

    workflow.run_suite(tmp_path, protocol_path, output, torch.device("cpu"), experiment="A")
    assert len(started) == 3
    workflow.run_suite(tmp_path, protocol_path, output, torch.device("cpu"), experiment="B")
    assert started[3:] == [("B", seed) for seed in (42, 123, 2026)]
    assert len(list(suite.glob("[ABC]-*/result.json"))) == 6
    assert not (suite / "selection.json").exists()
    assert "test" not in splits
    workflow.display_results(suite)

    workflow.run_suite(tmp_path, protocol_path, output, torch.device("cpu"), experiment="C")
    assert started[6:] == [("C", seed) for seed in (42, 123, 2026)]
    assert (suite / "test.json").is_file()
    assert len(list(suite.glob("[ABC]-*/result.json"))) == 9
    assert splits.count("test") == 1
    assert artifacts and all(path.suffix in (".pt", ".csv", ".json") for path in artifacts)
    workflow.run_suite(tmp_path, protocol_path, output, torch.device("cpu"), experiment="C")
    assert splits.count("test") == 1
    assert len(started) == 9
    workflow.display_results(suite)


@pytest.mark.parametrize("experiment", ["D", "ALL", ""])
def test_invalid_experiment_stops_before_preflight(tmp_path, monkeypatch, experiment):
    monkeypatch.setattr(workflow, "preflight", lambda *args: pytest.fail("Preflight should not run"))
    with pytest.raises(ValueError, match="exactly one"):
        workflow.run_suite(tmp_path, tmp_path / "protocol.json", tmp_path / "output",
                           experiment=experiment)


def test_wandb_logging_uses_online_runs_and_explicit_artifacts(tmp_path, monkeypatch):
    import wandb
    from model.metrics import forecast_metrics
    from training.tracking import tracked_run, log_report, log_artifact
    calls = {}
    class Run:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def log(self, payload):
            calls.setdefault("logs", []).append(payload)
        def log_artifact(self, artifact):
            calls["artifact"] = artifact
            return self
        def wait(self):
            calls["waited"] = True
    def init(**kwargs):
        calls["init"] = kwargs
        return Run()
    class Artifact:
        def __init__(self, **kwargs):
            self.files = []
        def add_file(self, path, name):
            self.files.append(name)
    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "Artifact", Artifact)
    path = tmp_path / "best.pt"
    path.write_bytes(b"synthetic checkpoint")
    (tmp_path / ".env").write_text("WANDB_API_KEY=not-uploaded")
    with tracked_run({"project": "test", "entity": None}, tmp_path, "test", "group", {}) as run:
        log_report(run, "validation", forecast_metrics(np.zeros((1, 12)), np.zeros((1, 12))))
        log_artifact(run, "test-model", [path])
    assert calls["init"]["mode"] == "online"
    assert calls["init"]["save_code"] is False
    assert calls["init"]["settings"].disable_git is True
    assert calls["artifact"].files == ["best.pt"]
    assert calls["waited"] is True
    assert "validation/per_horizon" in calls["logs"][1]
