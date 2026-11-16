"""Fit baseline or selected boosting models; save runs only on explicit request.

Training reads only 2017-2023 and validation only 2024. No test data is resolved.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import (
    ProjectPaths,
    FEATURES,
    BASE_FEATURES,
    STORM_FEATURES as STORM_FEATURES,
    HISTORY_FEATURES as HISTORY_FEATURES,
    FEATURE_SET,
    SEED,
    ROOT,
    sha256,
    git_commit,
    data_version,
    staged_directory,
)
from .evaluation import validate_frame, validate_probabilities, evaluate, naive_rules


def build_model(n_estimators=100, learning_rate=0.1, max_depth=3):
    """Median imputation is learned only by fit(train); trees need no scaling."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=SEED,
            # No random validation split or early stopping across time.
            n_iter_no_change=None,
        ),
    )


def synthetic_frames():
    """Small deterministic schema-compatible fixtures, unrelated to real outages."""
    import numpy as np
    import pandas as pd
    from .config import FAMILIES

    rng = np.random.default_rng(SEED)
    frames = []
    for name, n in [("train", 280), ("validation", 80)]:
        frame = pd.DataFrame(
            {
                "month": rng.integers(1, 13, n),
                "local_hour": rng.integers(0, 24, n),
                "pre_peak_24h": rng.uniform(0, 0.08, n),
                "pct_out_at_start": rng.uniform(0, 0.04, n),
                "year": np.resize(np.arange(2017, 2024), n)
                if name == "train"
                else 2024,
                "target": np.arange(n) % 2,
                "fips": [f"{i:05d}" for i in range(n)],
                "episode_id": np.arange(n) + (1000 if name == "validation" else 0),
                "fema_region": np.resize([1, 2], n),
                "hist_episodes": rng.integers(1, 100, n),
                "hist_outage_rate": rng.uniform(0.05, 0.4, n),
                "hist_mean_peak": rng.uniform(0, 0.08, n),
            }
        )
        for family in FAMILIES:
            frame[f"n_{family}"] = rng.integers(1, 4, n)
        frame.loc[::11, "pre_peak_24h"] = np.nan
        frames.append(frame)
    return frames


@dataclass
class ModelResult:
    """A fitted pipeline, keyed validation predictions, evaluation and run provenance."""

    model: object
    predictions: pd.DataFrame
    metrics: dict
    tables: dict
    config: dict


def load_model_data(paths=None):
    """Read explicit train/validation filenames and return actual source descriptors."""
    paths = paths or ProjectPaths()
    sources = [
        paths.resolve("processed/model_table/" + name)
        for name in ["train.parquet", "validation.parquet"]
    ]
    train, validation = [pd.read_parquet(source.path) for source in sources]
    return train, validation, sources


def fit_model(
    train,
    validation,
    *,
    kind="boosting",
    sources=(),
    bootstrap=False,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    smoke_test=False,
):
    """Fit on training only and return results; never write files or upload implicitly."""
    if kind not in {"boosting", "baseline"}:
        raise ValueError("Model kind must be baseline or boosting")
    features = FEATURES if kind == "boosting" else BASE_FEATURES
    validate_frame(train, "train", features)
    validate_frame(validation, "validation", features)
    if set(train.episode_id) & set(validation.episode_id):
        raise ValueError("Training and validation share storm episodes")
    if kind == "boosting":
        model = build_model(n_estimators, learning_rate, max_depth)
    else:
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )
    started = time.perf_counter()
    model.fit(train[features], train.target)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    p = validate_probabilities(
        validation,
        model.predict_proba(validation[features])[:, list(model.classes_).index(1)],
    )
    predict_seconds = time.perf_counter() - started
    metrics, tables = evaluate(validation, p, train, bootstrap=bootstrap)
    if kind == "baseline":
        from sklearn.metrics import confusion_matrix

        comparisons = []
        for name, scores in naive_rules(train, validation).items():
            values, _ = evaluate(validation, scores, train)
            comparisons.append({"rule": name, **values})
        comparisons.append({"rule": "logistic_regression", **metrics})
        tables["baseline_comparison"] = pd.DataFrame(comparisons)
        thresholds = []
        for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            tn, fp, fn, tp = confusion_matrix(
                validation.target, p >= threshold, labels=[0, 1]
            ).ravel()
            thresholds.append(
                {
                    "threshold": threshold,
                    "recall": tp / (tp + fn),
                    "precision": tp / (tp + fp) if tp + fp else np.nan,
                    "missed_outages": fn,
                    "false_alarms": fp,
                    "false_alarms_per_1000": 1000 * fp / len(validation),
                }
            )
        tables["threshold_comparison"] = pd.DataFrame(thresholds)
    predictions = validation[["fips", "episode_id", "year"]].copy()
    predictions["probability"] = p
    config = {
        "model": type(model[-1]).__name__,
        "feature_set": FEATURE_SET if kind == "boosting" else "base",
        "features": list(features),
        "train_years": sorted(int(y) for y in train.year.unique()),
        "validation_year": 2024,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "smoke_test": smoke_test,
        "parameters": model[-1].get_params(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        **data_version(sources),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (ROOT / "outage_prediction").glob("*.py")
        },
        "fit_seconds": fit_seconds,
        "validation_predict_seconds": predict_seconds,
        "bootstrap": bootstrap,
        "limitations": [
            "Whole-episode storm inputs are retrospective proxies.",
            "Histories use other training years and full-training smoothing priors.",
            "Labels use 24 hourly bins from floor_hour(t0).",
        ],
    }
    return ModelResult(model, predictions, metrics, tables, config)


def save_run(result, directory, validation, train):
    """Save a complete validated run, including model round-trip and scoring evidence."""
    from .plots import calibration_plot, pr_curve_plot
    import matplotlib.pyplot as plt

    if Path(directory).exists():
        raise FileExistsError(
            f"Run already exists: {directory}; choose a new directory"
        )
    with staged_directory(directory) as stage:
        result.predictions.to_csv(stage / "validation_predictions.csv", index=False)
        joblib.dump(result.model, stage / "model.joblib")
        reloaded = joblib.load(stage / "model.joblib")
        np.testing.assert_allclose(
            reloaded.predict_proba(validation[result.config["features"]])[:, 1],
            result.predictions.probability,
            rtol=1e-12,
            atol=1e-12,
        )
        result.config["model_bytes"] = (stage / "model.joblib").stat().st_size
        (stage / "config.json").write_text(
            json.dumps(result.config, indent=2), encoding="utf-8"
        )
        (stage / "validation_metrics.json").write_text(
            json.dumps(result.metrics, indent=2), encoding="utf-8"
        )
        for name, table in result.tables.items():
            table.to_csv(stage / f"{name}.csv", index=False)
        for name, fig in [
            (
                "calibration",
                calibration_plot(
                    result.tables["calibration_deciles"], result.config["model"]
                ),
            ),
            (
                "precision_recall",
                pr_curve_plot(
                    validation.target.to_numpy(),
                    result.predictions.probability.to_numpy(),
                    naive_rules(train, validation)["family_x_history"],
                    result.config["model"],
                ),
            ),
        ]:
            try:
                fig.savefig(stage / f"{name}.png", dpi=150, bbox_inches="tight")
            finally:
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind", nargs="?", choices=["boosting", "baseline"], default="boosting"
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--log-to-wandb", action="store_true")
    parser.add_argument("--upload-existing", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()
    if args.upload_existing:
        if (
            args.smoke_test
            or args.bootstrap
            or args.log_to_wandb
            or args.output_dir
            or args.kind != "boosting"
            or args.offline
            or args.n_estimators != 100
            or args.learning_rate != 0.1
            or args.max_depth != 3
        ):
            parser.error("Saved-run upload cannot be combined with training options")
        from .tracking import upload_saved_runs

        for directory in args.upload_existing:
            for result in upload_saved_runs(directory):
                print(result["url"])
        return
    if (
        args.n_estimators < 1
        or args.learning_rate <= 0
        or not np.isfinite(args.learning_rate)
        or args.max_depth < 1
    ):
        parser.error(
            "Estimator count, learning rate and depth must be finite and positive"
        )
    if args.smoke_test and args.log_to_wandb:
        parser.error("Synthetic runs cannot be uploaded")
    paths = ProjectPaths(offline=args.offline)
    output = args.output_dir or paths.models / (
        "smoke" if args.smoke_test else args.kind
    )
    if output.exists():
        parser.error("Choose a new output directory; existing runs are preserved")
    if args.smoke_test:
        train, validation = synthetic_frames()
        sources = []
    else:
        train, validation, sources = load_model_data(paths)
    result = fit_model(
        train,
        validation,
        kind=args.kind,
        sources=sources,
        bootstrap=args.bootstrap,
        n_estimators=5 if args.smoke_test else args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        smoke_test=args.smoke_test,
    )
    save_run(result, output, validation, train)
    print(
        f"{args.kind}: AP={result.metrics['pr_auc']:.6f}, Brier={result.metrics['brier']:.6f}; saved {output}"
    )
    if args.log_to_wandb:
        from .tracking import upload_saved_runs

        for uploaded in upload_saved_runs(output):
            print(uploaded["url"])


if __name__ == "__main__":
    main()
