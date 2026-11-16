"""Explicit experiment logging; raw data, row predictions and test files stay local."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .config import ROOT, WANDB_ENTITY, WANDB_PROJECT

TIMING_KEYS = ["fit_seconds", "validation_predict_seconds", "model_bytes"]


@dataclass
class PreparedRun:
    """All per-model upload inputs, checked before any remote run is created."""

    name: str
    folder: Path
    metrics_path: Path
    metrics: dict
    tables: dict
    config: dict
    run_id: str


@dataclass
class SavedBundle:
    """A saved candidate or comparison and its content-derived tracking identity."""

    directory: Path
    config_path: Path
    runs: list
    group: str
    model_tag: str
    prefix: str
    comparison: bool
    bootstrap: object


def tracking_directory():
    """Keep SDK logs with other ignored caches rather than adding a root folder."""
    directory = ROOT / ".cache/wandb"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def prepare_saved_runs(directory):
    """Read and preflight every run before allowing the first upload; no network I/O."""
    import pandas as pd

    directory = Path(directory).resolve()
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("smoke_test"):
        raise ValueError("Synthetic smoke results must not be uploaded as experiments")
    if config.get("validation_year") != 2024 or config.get("train_years") != list(
        range(2017, 2024)
    ):
        raise ValueError("Expected 2017-2023 training and 2024 validation")
    comparison = "feature_sets" in config
    names = list(config["feature_sets"]) if comparison else [config["feature_set"]]
    if not names:
        raise ValueError("Saved comparison contains no models")
    comparison_rows = (
        pd.read_csv(directory / "comparison.csv").set_index("model")
        if comparison
        else None
    )
    bootstrap_path = directory / "bootstrap_comparisons.csv"
    bootstrap = pd.read_csv(bootstrap_path) if bootstrap_path.exists() else None
    if bootstrap is not None and not {
        "model",
        "comparator",
        "ci_low",
        "ci_high",
    } <= set(bootstrap):
        raise ValueError("Malformed bootstrap comparison table")
    identity = hashlib.sha256(config_path.read_bytes())
    identity.update((directory / "validation_predictions.csv").read_bytes())
    identity.update(b"feature-comparison" if comparison else b"training-candidate")
    group = f"gb-{identity.hexdigest()[:16]}"
    prepared = []
    for name in names:
        folder = directory / name if comparison else directory
        metrics_path = folder / (
            "metrics.json" if comparison else "validation_metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("rows") != config["validation_rows"]:
            raise ValueError(f"{name}: evaluation row count mismatch")
        if not (folder / "model.joblib").is_file():
            raise ValueError(f"{name}: saved model missing")
        tables = {
            table: pd.read_csv(folder / f"{table}.csv")
            for table in [
                "calibration_deciles",
                "by_family",
                "by_fema_region",
                "by_year",
            ]
        }
        run_config = {
            **config,
            "feature_set": name,
            "features": config["feature_sets"][name]
            if comparison
            else config["features"],
            "eval_split": "validation_2024",
            "logged_from_saved_outputs": True,
            "experiment_kind": "feature-comparison"
            if comparison
            else "training-candidate",
            "source_directory": directory.name,
        }
        run_config.pop("feature_sets", None)
        if comparison:
            if (
                not set(TIMING_KEYS) <= set(comparison_rows)
                or name not in comparison_rows.index
            ):
                raise ValueError(f"{name}: missing comparison metadata")
            if not {"mean_pred", "observed"} <= set(tables["calibration_deciles"]):
                raise ValueError(f"{name}: malformed calibration table")
            for key in TIMING_KEYS:
                run_config[key] = comparison_rows.loc[name, key].item()
        elif not set(TIMING_KEYS) <= set(config):
            raise ValueError("Missing run timing/size metadata")
        run_id = hashlib.sha256(f"{group}:{name}".encode()).hexdigest()[:16]
        prepared.append(
            PreparedRun(name, folder, metrics_path, metrics, tables, run_config, run_id)
        )
    model_tag = (
        "logistic-regression"
        if config.get("model") == "LogisticRegression"
        else "gradient-boosting"
    )
    return SavedBundle(
        directory,
        config_path,
        prepared,
        group,
        model_tag,
        "lr" if model_tag == "logistic-regression" else "gb",
        comparison,
        bootstrap,
    )


def log_saved_evidence(run, prepared, bundle):
    """Attach only aggregate evidence and the matching model/config/metrics files."""
    import wandb
    import matplotlib.pyplot as plt
    from .plots import calibration_plot

    run.summary.update(prepared.metrics)
    run.summary.update({key: prepared.config[key] for key in TIMING_KEYS})
    for name, table in prepared.tables.items():
        run.log({name: wandb.Table(dataframe=table)})
    if bundle.bootstrap is not None:
        subset = bundle.bootstrap.loc[bundle.bootstrap.model == prepared.name]
        if not subset.empty:
            run.log({"bootstrap_comparisons": wandb.Table(dataframe=subset)})
            reference = subset.loc[subset.comparator == "reference"]
            if not reference.empty:
                run.summary.update(
                    {
                        "pr_auc_gain_ci_low": float(reference.iloc[0].ci_low),
                        "pr_auc_gain_ci_high": float(reference.iloc[0].ci_high),
                    }
                )
    if bundle.comparison:
        fig = calibration_plot(prepared.tables["calibration_deciles"], prepared.name)
        try:
            run.log({"calibration_plot": wandb.Image(fig)})
        finally:
            plt.close(fig)
    else:
        for filename, key in [
            ("calibration.png", "calibration_plot"),
            ("precision_recall.png", "pr_curve"),
        ]:
            if (prepared.folder / filename).is_file():
                run.log({key: wandb.Image(str(prepared.folder / filename))})
    artifact = wandb.Artifact(
        f"{bundle.model_tag}-{prepared.run_id}",
        type="model",
        metadata={"feature_set": prepared.name, "eval_split": "validation_2024"},
    )
    for path, name in [
        (prepared.folder / "model.joblib", "model.joblib"),
        (bundle.config_path, "original_config.json"),
        (prepared.metrics_path, "validation_metrics.json"),
    ]:
        artifact.add_file(str(path), name=name)
    run.log_artifact(artifact)


def upload_one(prepared, bundle):
    """Upload one preflighted model, close failures, and verify the remote summary."""
    import wandb

    print(
        f"Uploading {prepared.config['experiment_kind']}: {prepared.name}", flush=True
    )
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=prepared.run_id,
        resume="allow",
        name=f"{bundle.prefix}_{prepared.config['experiment_kind']}_{prepared.name}_{bundle.directory.name}",
        group=bundle.group,
        job_type=prepared.config["experiment_kind"],
        config=prepared.config,
        tags=[bundle.model_tag, prepared.name, "validation-2024", "saved-run-backfill"],
        mode="online",
        save_code=False,
        dir=tracking_directory(),
        settings=wandb.Settings(disable_git=True, disable_code=True),
    )
    try:
        log_saved_evidence(run, prepared, bundle)
        url = run.url
    except Exception:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
    remote = wandb.Api().run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{prepared.run_id}")
    if (
        remote.state != "finished"
        or abs(remote.summary["pr_auc"] - prepared.metrics["pr_auc"]) > 1e-12
    ):
        raise RuntimeError(f"Could not verify finished remote run {prepared.run_id}")
    return {
        "feature_set": prepared.name,
        "run_id": prepared.run_id,
        "url": url,
        "pr_auc": prepared.metrics["pr_auc"],
        "state": remote.state,
    }


def upload_saved_runs(directory):
    """Upload preflighted runs with stable IDs and preserve original training provenance.

    No fitting or dataset loading occurs. Each verified URL is saved immediately
    so interrupted multi-model uploads can be retried without duplicate runs.
    """
    bundle = prepare_saved_runs(directory)
    results = []
    for prepared in bundle.runs:
        results.append(upload_one(prepared, bundle))
        (bundle.directory / "wandb_uploads.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    return results


def log_run(
    name,
    job_type,
    frame,
    p,
    train,
    config,
    p_ref=None,
    data_paths=(),
    extra_tables=None,
    *,
    bootstrap=False,
):
    """Evaluate predictions, then log metrics, tables and charts to W&B as one run. Returns (metrics, tables, url).
    extra_tables = optional {name: DataFrame} of model-specific tables (e.g. threshold comparison) to log alongside."""
    import wandb
    import matplotlib.pyplot as plt
    import numpy as np
    from .config import WANDB_ENTITY, WANDB_PROJECT, data_version, git_commit
    from .evaluation import evaluate
    from .plots import calibration_plot, pr_curve_plot

    metrics, tables = evaluate(frame, p, train, p_ref=p_ref, bootstrap=bootstrap)
    cfg = {
        **config,
        "git_commit": git_commit(),
        **data_version(data_paths),
        "eval_split": "validation_2024",
    }
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=name,
        job_type=job_type,
        config=cfg,
        reinit="finish_previous",
        dir=tracking_directory(),
    )
    try:
        run.summary.update(metrics)
        for tname, t in {**tables, **(extra_tables or {})}.items():
            run.log({tname: wandb.Table(dataframe=t)})
        y = frame["target"].to_numpy()
        for key, fig in [
            (
                "calibration_plot",
                calibration_plot(tables["calibration_deciles"], f"Calibration: {name}"),
            ),
            (
                "pr_curve",
                pr_curve_plot(
                    y,
                    np.asarray(p, float),
                    None if p_ref is None else np.asarray(p_ref, float),
                    f"PR curve: {name}",
                ),
            ),
        ]:
            run.log({key: wandb.Image(fig)})
            plt.close(fig)
        url = run.url
    except Exception:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
    return metrics, tables, url


def main():
    """Upload completed local runs only on explicit command invocation."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        for result in upload_saved_runs(directory):
            print(result["url"])


if __name__ == "__main__":
    main()
