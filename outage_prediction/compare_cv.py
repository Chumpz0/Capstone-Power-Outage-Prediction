"""Compare saved nested outer predictions with fold-local reference rules; never fit models."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .config import N_BOOT, SEED, ProjectPaths, data_version, sha256, staged_directory
from .evaluation import (
    bootstrap_gain, naive_rules, recall_at_top, validate_frame, validate_probabilities,
)
from .nested_cv import rebuild_history, year_folds


def compare_predictions(train, predictions, *, outer_folds=3, n_boot=N_BOOT, seed=SEED,
                        progress=None):
    """Paired comparisons on exact outer rows, with whole-episode AP-gain intervals.

    Intervals condition on the saved fitted models and do not repeat selection or
    training. Each year uses its own past-only, episode-purged reference training.
    """
    validate_frame(train, "train")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    keys = ["fips", "episode_id", "year"]
    required = keys + ["target", "probability"]
    if not set(required) <= set(predictions):
        raise ValueError(f"Predictions require columns {required}")
    if predictions[keys].isna().any().any() or predictions.duplicated(keys).any():
        raise ValueError("Prediction keys must be present and unique")
    folds = year_folds(train, outer_folds)
    expected = pd.concat([held for _, held, _ in folds])
    if len(predictions) != len(expected):
        raise ValueError("Prediction rows do not match outer-fold rows")
    expected_index = pd.MultiIndex.from_frame(expected[keys])
    indexed = predictions.set_index(keys)
    if not expected_index.isin(indexed.index).all():
        raise ValueError("Prediction keys do not match outer-fold keys")
    aligned = indexed.loc[expected_index].reset_index()
    if not np.array_equal(aligned.target.to_numpy(), expected.target.to_numpy()):
        raise ValueError("Saved prediction targets do not match source outcomes")
    validate_probabilities(expected, aligned.probability.to_numpy())
    rows, scored = [], []

    def score(frame, scope, purged=None):
        y = frame.target.to_numpy()
        p = frame.probability.to_numpy()
        reference = frame.reference_probability.to_numpy()
        low, high = bootstrap_gain(
            y, p, reference, frame.episode_id.to_numpy(), n_boot=n_boot, seed=seed,
        )
        row = {
            "scope": scope, "rows": len(frame), "episodes": frame.episode_id.nunique(),
            "positive_rate": float(y.mean()), "purged_train_rows": purged,
            "model_ap": float(average_precision_score(y, p)),
            "reference_ap": float(average_precision_score(y, reference)),
            "model_brier": float(brier_score_loss(y, p)),
            "reference_brier": float(brier_score_loss(y, reference)),
            "model_recall_top10": float(recall_at_top(y, p)),
            "reference_recall_top10": float(recall_at_top(y, reference)),
            "ap_gain_ci_low": float(low), "ap_gain_ci_high": float(high),
        }
        row["ap_gain"] = row["model_ap"] - row["reference_ap"]
        row["brier_reduction"] = row["reference_brier"] - row["model_brier"]
        row["recall_top10_gain"] = row["model_recall_top10"] - row["reference_recall_top10"]
        row["passes_all_point_criteria"] = bool(
            row["ap_gain"] > 0 and row["brier_reduction"] >= 0 and row["recall_top10_gain"] >= 0
        )
        row["ap_gain_ci_above_zero"] = bool(low > 0)
        if progress:
            progress(row)
        rows.append(row)

    offset = 0
    for fold_train, held, purged in folds:
        local_train, local_held = rebuild_history(fold_train, held)
        block = aligned.iloc[offset:offset + len(held)].copy()
        block["reference_probability"] = validate_probabilities(
            local_held, naive_rules(local_train, local_held)["family_x_history"],
        )
        score(block, str(int(held.year.iloc[0])), purged)
        scored.append(block)
        offset += len(held)
    score(pd.concat(scored, ignore_index=True), "pooled")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory; existing results are preserved")
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")
    config_path = args.run_dir / "config.json"
    prediction_path = args.run_dir / "nested_cv_predictions.csv"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cv = config["nested_cross_validation"]
    source = ProjectPaths(offline=args.offline).resolve("processed/model_table/train.parquet")
    version = data_version([source])
    recorded = next(s for s in config["sources"] if s["logical_path"] == source.logical_path)
    actual = version["sources"][0]
    if (recorded.get("sha256") and recorded["sha256"] != actual.get("sha256")) or (
        recorded.get("hf_revision") and recorded["hf_revision"] != actual.get("hf_revision")
    ):
        raise ValueError("Training data provenance differs from the saved run")
    train = pd.read_parquet(source.path)
    predictions = pd.read_csv(prediction_path, dtype={"fips": str})
    table = compare_predictions(
        train, predictions, outer_folds=cv["outer_folds"], n_boot=args.n_boot,
        progress=lambda row: print(json.dumps(row), flush=True),
    )
    yearly = table[table.scope.ne("pooled")]
    summary = {
        "reference": "family_x_history rebuilt from each fold's training outcomes",
        "n_boot": args.n_boot, "seed": SEED,
        "interval": "paired 95% percentile interval for AP gain; whole episode clusters",
        "limitations": [
            "Conditions on fitted models; does not include retraining or selection uncertainty.",
            "Yearwise intervals are exploratory and unadjusted for multiple comparisons.",
            "Pooled resampling retains whole episodes across years; year composition can vary.",
            "Histories and whole-episode storm inputs retain retrospective limitations.",
        ],
        "outer_mean_model_ap": float(yearly.model_ap.mean()),
        "outer_mean_reference_ap": float(yearly.reference_ap.mean()),
        "outer_mean_ap_gain": float(yearly.ap_gain.mean()),
        "all_years_pass_point_criteria": bool(yearly.passes_all_point_criteria.all()),
        "all_years_ap_gain_ci_above_zero": bool(yearly.ap_gain_ci_above_zero.all()),
        "source_run": str(args.run_dir),
        "run_config_sha256": sha256(config_path),
        "predictions_sha256": sha256(prediction_path),
        "comparison_source_sha256": sha256(Path(__file__)),
        "training_source": {k: v for k, v in actual.items() if k != "path"},
        "comparisons": table.astype(object).where(pd.notna(table), None).to_dict(orient="records"),
        "test_2025": "not accessed",
    }
    with staged_directory(args.output_dir) as stage:
        table.to_csv(stage / "reference_comparison.csv", index=False)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
