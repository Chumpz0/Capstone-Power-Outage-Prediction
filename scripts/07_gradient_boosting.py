"""Train a four-feature gradient boosting classifier; never open the 2025 holdout.

Default inputs match step 05: local train/validation parquet, then Hugging Face.
Use --smoke-test for a small, entirely synthetic offline integration check.
Full training, tuning, and interpretation belong to the modeling teammate.
See scripts/GRADIENT_BOOSTING.md for commands and artifact descriptions.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["month", "local_hour", "pre_peak_24h", "pct_out_at_start"]
SEED = 42


def validate_frame(frame, name):
    """Fail closed on wrong years, malformed labels, or unusable features."""
    import numpy as np
    import pandas as pd

    required = FEATURES + ["year", "target", "fips", "episode_id"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    allowed = range(2017, 2024) if name == "train" else [2024]
    if frame.empty or not frame["year"].isin(allowed).all():
        raise ValueError(f"{name}: expected only years {list(allowed)} and nonempty data")
    if frame["target"].isna().any() or set(frame["target"].unique()) != {0, 1}:
        raise ValueError(f"{name}: target must contain both binary classes, without nulls")
    if frame[["fips", "episode_id"]].isna().any().any() or frame.duplicated(["fips", "episode_id"]).any():
        raise ValueError(f"{name}: county/episode keys must be present and unique")
    for feature in FEATURES:
        if not pd.api.types.is_numeric_dtype(frame[feature]):
            raise ValueError(f"{name}: {feature} must be numeric")
        values = frame[feature].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or np.isnan(values).all():
            raise ValueError(f"{name}: {feature} is infinite or entirely missing")


def build_model(n_estimators=100, learning_rate=0.1, max_depth=3):
    """Median imputation is learned only by fit(train); trees need no scaling."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=SEED,
            # No random validation split or early stopping across time.
            n_iter_no_change=None,
        ),
    )


def synthetic_frames():
    """Small deterministic schema-compatible fixtures, unrelated to real outages."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from src.evaluation import FAMILIES

    rng = np.random.default_rng(SEED)
    frames = []
    for name, n in [("train", 280), ("validation", 80)]:
        frame = pd.DataFrame({
            "month": rng.integers(1, 13, n), "local_hour": rng.integers(0, 24, n),
            "pre_peak_24h": rng.uniform(0, 0.08, n),
            "pct_out_at_start": rng.uniform(0, 0.04, n),
            "year": np.resize(np.arange(2017, 2024), n) if name == "train" else 2024,
            "target": np.arange(n) % 2, "fips": [f"{i:05d}" for i in range(n)],
            "episode_id": np.arange(n) + (1000 if name == "validation" else 0),
            "fema_region": np.resize([1, 2], n), "hist_outage_rate": 0.5,
        })
        for family in FAMILIES:
            frame[f"n_{family}"] = 1
        frame.loc[::11, "pre_peak_24h"] = np.nan
        frames.append(frame)
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run shared 2024 validation checks (no bootstrap or W&B).")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, help="New directory; existing directories are refused.")
    args = parser.parse_args()
    if args.n_estimators < 1 or args.learning_rate <= 0 or args.max_depth < 1:
        parser.error("estimator count, learning rate, and depth must be positive")

    import joblib
    import numpy as np
    import pandas as pd

    if args.smoke_test:
        train, val = synthetic_frames()
        sources = ["synthetic_train", "synthetic_validation"]
    else:
        def source(filename):
            local = ROOT / "processed" / "model_table" / filename
            return str(local) if local.exists() else f"hf://datasets/DSCI-Capstone/county-outage-data/processed/model_table/{filename}"
        # Explicit filenames: never enumerate or load any holdout files.
        sources = [source("train.parquet"), source("validation.parquet")]
        train, val = [pd.read_parquet(path) for path in sources]

    validate_frame(train, "train")
    validate_frame(val, "validation")
    output = args.output_dir or ROOT / "work" / "gradient_boosting" / ("smoke" if args.smoke_test else "initial")
    output.mkdir(parents=True, exist_ok=False)
    model = build_model(5 if args.smoke_test else args.n_estimators, args.learning_rate, args.max_depth)
    model.fit(train[FEATURES], train["target"])
    probabilities = model.predict_proba(val[FEATURES])[:, list(model.classes_).index(1)]
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid predicted probabilities")
    predictions = val[["fips", "episode_id", "year"]].copy()
    predictions["probability"] = probabilities
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    joblib.dump(model, output / "model.joblib")
    config = {
        "model": "GradientBoostingClassifier", "features": FEATURES,
        "train_years": sorted(int(y) for y in train["year"].unique()),
        "validation_year": 2024, "train_rows": len(train), "validation_rows": len(val),
        "sources": sources, "smoke_test": args.smoke_test,
        "preprocessing": "training-only median imputation plus missing indicators",
        "parameters": model[-1].get_params(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Smoke checks serialization and the shared evaluation integration too.
    if args.smoke_test:
        np.testing.assert_allclose(joblib.load(output / "model.joblib").predict_proba(val[FEATURES])[:, 1], probabilities)
    if args.evaluate or args.smoke_test:
        import matplotlib
        matplotlib.use("Agg")
        sys.path.insert(0, str(ROOT))
        from src.evaluation import evaluate
        metrics, tables = evaluate(val, probabilities, train)
        (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False)
    print(f"{'SYNTHETIC SMOKE TEST' if args.smoke_test else 'Training'} complete: {output.resolve()}")


if __name__ == "__main__":
    main()
