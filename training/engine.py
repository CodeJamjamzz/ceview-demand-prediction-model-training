"""Seeded training, baseline prediction, and validation-only selection."""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.architectures import Forecaster
from model.experiment import file_digest, load_protocol
from model.metrics import forecast_metrics


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_split(root, protocol, split):
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown dataset partition")
    path = Path(root) / protocol["dataset"]["path"] / "core" / f"{split}.npz"
    relative = path.relative_to(Path(root)).as_posix()
    if file_digest(path) != protocol["dataset"]["files_sha256"][relative]:
        raise ValueError(f"Frozen {split} partition changed")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ("X", "market_id", "category_id", "y")}
    n = protocol["dataset"]["counts"][split]
    for key, shape, dtype in (
        ("X", (n, 52, 3), np.float32), ("y", (n, 12), np.float32),
        ("market_id", (n,), np.int64), ("category_id", (n,), np.int64),
    ):
        value = arrays[key]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid {split} {key} shape, dtype, or values")
    if not ((arrays["y"] >= 0) & (arrays["y"] <= 1)).all():
        raise ValueError("Targets must be scaled to [0, 1]")
    for key, vocabulary in (("market_id", "market"), ("category_id", "category")):
        if ((arrays[key] < 0) | (arrays[key] >= len(protocol["dataset"]["vocabulary"][vocabulary]))).any():
            raise ValueError("Identity outside frozen vocabulary")
    return arrays


def loader(arrays, batch_size, shuffle=False, seed=42):
    tensors = [torch.from_numpy(arrays[key]) for key in ("X", "market_id", "category_id", "y")]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def make_optimizer(model, settings):
    excluded = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if "bias" in name or isinstance(module, nn.LayerNorm):
                excluded.add(id(parameter))
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if id(parameter) in excluded else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": settings["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=settings["learning_rate"], betas=tuple(settings["betas"]),
        eps=settings["eps"], amsgrad=settings["amsgrad"],
    )


def predict(model, arrays, batch_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for history, market, category, _ in loader(arrays, batch_size):
            predictions.append(model(history.to(device), market.to(device), category.to(device)).cpu().numpy())
    return np.concatenate(predictions)


def score(arrays, prediction, protocol_path):
    return forecast_metrics(arrays["y"], prediction, market_id=arrays["market_id"],
                            category_id=arrays["category_id"], protocol_path=protocol_path)


def baseline_predictions(history):
    return {
        "last_value": np.repeat(history[:, -1, 0, None], 12, axis=1),
        "moving_average_7": np.repeat(history[:, -7:, 0].mean(axis=1)[:, None], 12, axis=1),
        "seasonal_naive_52": history[:, :12, 0].copy(),
    }


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.best = float("inf")
        self.significant_best = float("inf")
        self.bad_epochs = 0

    def update(self, value):
        if not np.isfinite(value):
            raise ValueError("Nonfinite validation loss")
        checkpoint = value < self.best
        if checkpoint:
            self.best = value
        if value < self.significant_best - self.min_delta:
            self.significant_best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return checkpoint, self.bad_epochs >= self.patience


def train_one(architecture, seed, train, validation, protocol_path, directory,
              device, log_epoch):
    protocol = load_protocol(protocol_path)
    settings = protocol["training"]
    seed_everything(seed)
    model = Forecaster(architecture, protocol_path).to(device)
    optimizer = make_optimizer(model, settings)
    batches = loader(train, settings["batch_size"], True, seed)
    stopper = EarlyStopping(settings["early_stopping"]["patience"],
                            settings["early_stopping"]["min_delta"])
    path = Path(directory) / "best.pt"
    best_epoch, history = 0, []
    started = time.perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for x, market, category, targets in batches:
            x, market, category, targets = [value.to(device) for value in (x, market, category, targets)]
            optimizer.zero_grad(set_to_none=True)
            output = model(x, market, category)
            loss = (output - targets).abs().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"],
                                     norm_type=settings["gradient_clip_norm_type"], error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * targets.numel()
            count += targets.numel()
        validation_prediction = predict(model, validation, settings["batch_size"], device)
        validation_mae = float(np.abs(validation_prediction.astype(np.float64) - validation["y"]).mean())
        save, stop = stopper.update(validation_mae)
        if save:
            best_epoch = epoch
            temporary = path.with_suffix(".tmp")
            torch.save({"state_dict": model.state_dict(), "architecture": architecture, "seed": seed,
                        "epoch": epoch, "validation_mae_scaled": validation_mae,
                        "protocol_sha256": protocol["protocol_sha256"]}, temporary)
            temporary.replace(path)
        record = {"epoch": epoch, "train/mae_scaled": total / count,
                  "validation/mae_scaled": validation_mae,
                  "train/mae_index": total / count * 100,
                  "validation/mae_index": validation_mae * 100}
        history.append(record)
        log_epoch(record)
        if stop:
            break
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validation_prediction = predict(model, validation, settings["batch_size"], device)
    train_prediction = predict(model, train, settings["batch_size"], device)
    report = {
        "architecture": architecture, "seed": seed, "best_epoch": best_epoch,
        "epochs": len(history), "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_seconds": time.perf_counter() - started,
        "train": score(train, train_prediction, protocol_path),
        "validation": score(validation, validation_prediction, protocol_path),
        "checkpoint_sha256": file_digest(path),
    }
    # Same full validation loader and hardware for every run; includes batching/transfer.
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    predict(model, validation, settings["batch_size"], device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    report["inference_seconds_per_example"] = (time.perf_counter() - start) / len(validation["y"])
    return report


def select_champion(results, baselines, protocol):
    expected = {(architecture, seed) for architecture in ("A", "B", "C")
                for seed in protocol["training"]["seeds"]}
    received = [(r["architecture"], r["seed"]) for r in results]
    if set(received) != expected or len(received) != len(expected):
        raise ValueError("Selection requires exactly all nine completed runs")
    if set(baselines) != set(protocol["selection"]["baseline_names"]):
        raise ValueError("All three required baselines must be scored")
    summary = []
    for architecture in ("A", "B", "C"):
        rows = [r for r in results if r["architecture"] == architecture]
        entry = {"architecture": architecture, "parameter_count": rows[0]["parameter_count"]}
        for metric in ("mae", "rmse", "smape_percent"):
            values = [r["validation"]["overall"][metric] for r in rows]
            if not np.isfinite(values).all():
                raise ValueError("Cannot select using nonfinite metrics")
            entry[f"{metric}_mean"] = float(np.mean(values))
            entry[f"{metric}_std"] = float(np.std(values, ddof=1))
        summary.append(entry)
    winner = min(summary, key=lambda r: (r["mae_mean"], r["smape_percent_mean"],
                                        r["parameter_count"], r["architecture"]))
    baseline_mae = min(r["overall"]["mae"] for r in baselines.values())
    if not np.isfinite(baseline_mae):
        raise ValueError("Invalid baseline score")
    selected = winner["architecture"] if winner["mae_mean"] < baseline_mae else None
    return {"architecture": selected, "seed": protocol["selection"]["final_seed"],
            "reason": "Validation winner beats baseline" if selected else "No neural model beats the best baseline",
            "best_baseline_mae": baseline_mae, "summary": summary,
            "protocol_sha256": protocol["protocol_sha256"]}
