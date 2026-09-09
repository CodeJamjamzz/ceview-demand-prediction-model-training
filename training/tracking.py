"""Explicit W&B logging; credentials and dataset files are never artifacts."""

from contextlib import contextmanager
import os

from dotenv import dotenv_values


def configure_wandb(env_path):
    values = dotenv_values(env_path) if env_path.is_file() else {}
    for name in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"):
        if not os.environ.get(name) and values.get(name):
            os.environ[name] = values[name]
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("Add WANDB_API_KEY to the project .env or Colab Secrets")
    return {"project": os.environ.get("WANDB_PROJECT", "ceview-forecasting"),
            "entity": os.environ.get("WANDB_ENTITY") or None}


@contextmanager
def tracked_run(settings, directory, name, group, config):
    import wandb
    with wandb.init(project=settings["project"], entity=settings["entity"], name=name,
                    group=group, config=config, dir=str(directory), mode="online",
                    save_code=False, settings=wandb.Settings(disable_git=True, console="off")) as run:
        yield run


def log_report(run, prefix, report):
    import wandb
    run.log({f"{prefix}/{key}": value for key, value in report["overall"].items()
             if value is not None})
    columns = ["mae", "rmse", "smape_percent", "mape_percent", "mape_coverage"]
    horizons = [[i + 1] + [r[key] for key in columns] for i, r in enumerate(report["per_horizon"])]
    tables = {f"{prefix}/per_horizon": wandb.Table(columns=["horizon"] + columns, data=horizons)}
    for group in ("market", "category", "series"):
        if f"by_{group}" in report:
            tables[f"{prefix}/by_{group}"] = wandb.Table(
                columns=[group] + columns,
                data=[[name] + [r["overall"][key] for key in columns]
                      for name, r in report[f"by_{group}"].items()])
    for window in ("demand_mean_4w", "demand_mean_12w"):
        tables.update({f"{prefix}/{window}/{key}": value
                       for key, value in report[window].items() if value is not None})
    run.log(tables)


def log_artifact(run, name, paths, artifact_type="model"):
    import wandb
    artifact = wandb.Artifact(name=name, type=artifact_type)
    for path in paths:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact).wait()
