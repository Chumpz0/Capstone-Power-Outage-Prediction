"""Nested expanding-year model selection using 2017-2023; 2025 stays sealed."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import ParameterSampler

from .config import FEATURES, HISTORY_FEATURES, ProjectPaths, SEED
from .evaluation import validate_frame, validate_probabilities
from .features import add_history
from .train import build_model, fit_model, load_model_data, save_run


def year_folds(frame, n_splits):
    """Expanding training windows, purging episodes that appear in the held-out year."""
    years = sorted(int(y) for y in frame.year.unique())
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError("Fold count must be a positive integer")
    if len(years) <= n_splits:
        raise ValueError("Not enough years for the requested expanding-year folds")
    folds = []
    for year in years[-n_splits:]:
        held = frame.loc[frame.year.eq(year)].copy()
        past = frame.loc[frame.year.lt(year)]
        train = past.loc[~past.episode_id.isin(held.episode_id)].copy()
        validate_frame(train, "train")
        validate_frame(held, "train")  # Both sides must remain within 2017-2023.
        folds.append((train, held, len(past) - len(train)))
    return folds


def rebuild_history(train, held):
    """Discard released histories and derive replacements from fold training outcomes only.

    Retains the released leave-one-training-year-out definition and fold-training
    smoothing priors. Held-out targets and peaks cannot affect either transform.
    """
    for frame in (train, held):
        if "retro_peak_24h" not in frame:
            raise ValueError("Fold history requires retro_peak_24h in labeled source data")
    inputs = pd.concat([
        train.assign(split="train", peak_24h=train.retro_peak_24h),
        held.assign(split="validation", target=np.nan, peak_24h=np.nan),
    ], ignore_index=True)
    rebuilt = add_history(inputs.drop(columns=HISTORY_FEATURES))
    left, right = train.copy(), held.copy()
    left[HISTORY_FEATURES] = rebuilt.iloc[:len(train)][HISTORY_FEATURES].to_numpy()
    right[HISTORY_FEATURES] = rebuilt.iloc[len(train):][HISTORY_FEATURES].to_numpy()
    return left, right


def _predict(train, held, parameters):
    model = build_model(**parameters)
    model.fit(train[FEATURES], train.target)
    return validate_probabilities(held, model.predict_proba(held[FEATURES])[:, 1])


def _search(frame, candidates, inner_folds, stage, progress):
    # Rebuild only once per fold, independently of candidate parameters.
    folds = [rebuild_history(train, held) for train, held, _ in year_folds(frame, inner_folds)]
    rows = []
    best_score, best_parameters = -np.inf, None
    for trial, parameters in enumerate(candidates, 1):
        scores = []
        for train, held in folds:
            p = _predict(train, held, parameters)
            score = float(average_precision_score(held.target, p))
            scores.append(score)
            rows.append({
                "stage": stage, "trial": trial, **parameters,
                "inner_validation_year": int(held.year.iloc[0]),
                "train_years": ",".join(str(y) for y in sorted(train.year.unique())),
                "train_rows": len(train), "validation_rows": len(held),
                "pr_auc": score, "brier": float(brier_score_loss(held.target, p)),
            })
        mean = float(np.mean(scores))
        if not np.isfinite(mean):
            raise ValueError("Nonfinite inner-fold average precision")
        if mean > best_score:
            best_score, best_parameters = mean, parameters
        if progress:
            progress({"event": "trial_completed", "stage": stage, "trial": trial,
                      "trials": len(candidates), "mean_inner_ap": mean, **parameters})
    for row in rows:
        row["selected"] = all(row[key] == value for key, value in best_parameters.items())
    return best_parameters, best_score, rows


def nested_cv(train, validation, *, n_trials=12, outer_folds=3, inner_folds=3,
              sources=(), bootstrap=False, progress=None, smoke_test=False):
    """Estimate selection performance on outer years, then tune all training data.

    2024 never participates in candidate selection. Outer scores describe the
    selection procedure, rather than a single model fit on all training years.
    """
    validate_frame(train, "train")
    validate_frame(validation, "validation")
    if set(train.episode_id) & set(validation.episode_id):
        raise ValueError("Training and validation share storm episodes")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    space = {
        "n_estimators": [2, 5] if smoke_test else [50, 100, 200, 300],
        "learning_rate": [0.03, 0.1] if smoke_test else [0.01, 0.03, 0.1, 0.2],
        "max_depth": [1, 2] if smoke_test else [1, 2, 3],
    }
    candidates = list(ParameterSampler(
        space, n_iter=min(n_trials, int(np.prod([len(v) for v in space.values()]))),
        random_state=SEED,
    ))
    outer = year_folds(train, outer_folds)
    # Check feasibility before starting a potentially long run.
    for fold_train, _, _ in outer:
        year_folds(fold_train, inner_folds)
    year_folds(train, inner_folds)
    started = time.perf_counter()
    trial_rows, outer_rows, predictions = [], [], []
    for fold_train, held, purged in outer:
        year = int(held.year.iloc[0])
        parameters, inner_ap, rows = _search(
            fold_train, candidates, inner_folds, f"outer_{year}", progress
        )
        trial_rows.extend(rows)
        local_train, local_held = rebuild_history(fold_train, held)
        p = _predict(local_train, local_held, parameters)
        outer_rows.append({
            "validation_year": year, "train_rows": len(fold_train),
            "train_years": ",".join(str(y) for y in sorted(fold_train.year.unique())),
            "validation_rows": len(held), "purged_train_rows": purged,
            "inner_mean_pr_auc": inner_ap, **parameters,
            "pr_auc": float(average_precision_score(held.target, p)),
            "brier": float(brier_score_loss(held.target, p)),
        })
        pred = held[["fips", "episode_id", "year", "target"]].copy()
        pred["probability"] = p
        predictions.append(pred)
        if progress:
            progress({"event": "outer_fold_completed", **outer_rows[-1]})
    parameters, inner_ap, rows = _search(train, candidates, inner_folds, "final", progress)
    trial_rows.extend(rows)
    final_train, final_validation = rebuild_history(train, validation)
    result = fit_model(
        final_train, final_validation, **parameters, sources=sources,
        bootstrap=bootstrap, smoke_test=smoke_test,
    )
    oof = pd.concat(predictions, ignore_index=True)
    outer_table = pd.DataFrame(outer_rows)
    result.tables.update({
        "nested_cv_trials": pd.DataFrame(trial_rows),
        "nested_cv_outer_folds": outer_table,
        "nested_cv_predictions": oof,
    })
    result.config["nested_cross_validation"] = {
        "method": "nested_expanding_year", "seed": SEED,
        "objective": "mean_inner_year_average_precision",
        "outer_folds": outer_folds, "inner_folds": inner_folds,
        "parameter_space": space, "requested_trials": n_trials,
        "candidates_per_search": len(candidates), "best_parameters": parameters,
        "final_inner_mean_pr_auc": inner_ap,
        "outer_mean_pr_auc": float(outer_table.pr_auc.mean()),
        "outer_std_pr_auc": float(outer_table.pr_auc.std(ddof=0)),
        "outer_mean_brier": float(outer_table.brier.mean()),
        "outer_pooled_pr_auc": float(average_precision_score(oof.target, oof.probability)),
        "outer_pooled_brier": float(brier_score_loss(oof.target, oof.probability)),
        "history_policy": "fold-local leave-one-training-year-out, fold-training priors",
        "validation_role": "2024 evaluation only, excluded from search",
        "search_seconds": time.perf_counter() - started,
    }
    result.config["limitations"][1] = (
        "Histories rebuilt in each fold: other fold-training years and fold-training priors; "
        "training-row histories are not strictly chronological."
    )
    result.config["limitations"].append(
        "2024 has been used in prior project development; this is not a fresh untouched holdout."
    )
    return result, final_train, final_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory; existing runs are preserved")
    if min(args.n_trials, args.outer_folds, args.inner_folds) < 1:
        parser.error("Trial and fold counts must be positive")
    train, validation, sources = load_model_data(ProjectPaths(offline=args.offline))

    def report(event):
        print(json.dumps(event), flush=True)

    report({"event": "started", "train_rows": len(train), "validation_rows": len(validation),
            "n_trials": args.n_trials, "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds})
    result, final_train, final_validation = nested_cv(
        train, validation, n_trials=args.n_trials, outer_folds=args.outer_folds,
        inner_folds=args.inner_folds, sources=sources, bootstrap=args.bootstrap, progress=report,
    )
    save_run(result, args.output_dir, final_validation, final_train)
    report({"event": "completed", "directory": str(args.output_dir),
            "nested_cv": result.config["nested_cross_validation"],
            "validation_ap": result.metrics["pr_auc"], "validation_brier": result.metrics["brier"]})


if __name__ == "__main__":
    main()
