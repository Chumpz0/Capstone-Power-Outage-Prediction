# Baseline model pipelines + one function that trains, times, evaluates and (optionally) logs a run.
# Used by scripts/train.py and notebook 05, so both give identical results for the same settings.
import os
import platform
import time
from functools import partial

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import model_table_paths
from src.evaluation import SEED, evaluate, log_run, naive_rules
from src.features import (COUNT_FEATURES, CYCLIC_FEATURES, EXCLUDED, MODEL_FEATURES, ONE_HOT_FEATURES,
                          REPORT_MAGNITUDES, TARGET, UNOBSERVED_FEATURES, check_columns, plain_features)

MODELS = ("logreg", "hgb")


# ---- Preprocessing helpers ---------------------------------------------------------
def _select_float(X, columns):
    """Keep the model features as float64 (booleans -> 0/1, nullable ints -> NaN when empty)."""
    return X[columns].astype("float64")


def _selected_names(transformer, names_in, columns):
    return np.asarray(columns, dtype=object)


def _hour_sin_cos(X):
    h = np.asarray(X, dtype=float) * 2 * np.pi / 24
    return np.column_stack([np.sin(h).ravel(), np.cos(h).ravel()])


def _hour_names(transformer, names_in):
    return np.array([f"{n}_{part}" for n in names_in for part in ("sin", "cos")])


# ---- Pipelines -----------------------------------------------------------------------
def build_logreg(C=1.0, class_weight=None, features=None):
    """Logistic regression on transformed, scaled features (decisions documented in reports/06 summary):
    counts -> log1p; report magnitudes empty -> 0 + missing flag; unobserved -> median + missing flag;
    local hour -> sin/cos; month -> one-hot; everything scaled."""
    features = MODEL_FEATURES if features is None else features
    keep = lambda cols: [c for c in cols if c in features]  # noqa: E731
    pre = ColumnTransformer([
        ("counts", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()), keep(COUNT_FEATURES)),
        ("magnitudes", make_pipeline(SimpleImputer(strategy="constant", fill_value=0, add_indicator=True, keep_empty_features=True),
                                     StandardScaler()), keep(REPORT_MAGNITUDES)),
        ("unobserved", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), keep(UNOBSERVED_FEATURES)),
        ("hour", FunctionTransformer(_hour_sin_cos, feature_names_out=_hour_names), keep(CYCLIC_FEATURES)),
        ("month", OneHotEncoder(handle_unknown="ignore"), keep(ONE_HOT_FEATURES)),
        ("plain", StandardScaler(), plain_features(features)),
    ])
    return Pipeline([
        ("select", FunctionTransformer(partial(_select_float, columns=features),
                                   feature_names_out=partial(_selected_names, columns=features))),
        ("pre", pre),
        ("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000)),
    ])


def build_hgb(learning_rate=0.1, max_iter=500, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0,
              class_weight=None, features=None):
    """Gradient boosting on the raw features: it handles missing values and non-linear effects itself.
    Early stopping holds out a random 10% of the training rows (provisional)."""
    features = MODEL_FEATURES if features is None else features
    return Pipeline([
        ("select", FunctionTransformer(partial(_select_float, columns=features),
                                   feature_names_out=partial(_selected_names, columns=features))),
        ("model", HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization, class_weight=class_weight,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)),
    ])


def build_model(model, **params):
    if model == "logreg":
        return build_logreg(**params)
    if model == "hgb":
        return build_hgb(**params)
    raise ValueError(f"unknown model {model!r}; choose from {MODELS}")


# ---- Run one experiment --------------------------------------------------------------------
def hardware():
    info = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9)
    except ImportError:
        pass
    return info


def run_experiment(model, params, train, val, name=None, log=True, notes=""):
    """Fit on train (2017-2023), score validation (2024), evaluate against the success criteria and,
    if log=True, send the run to W&B. Returns a dict with metrics, tables, the run url and the fitted model."""
    check_columns(train)
    check_columns(val)
    pipe = build_model(model, **params)

    start = time.perf_counter()
    pipe.fit(train, train[TARGET])
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    p = pipe.predict_proba(val)[:, 1]
    predict_seconds = time.perf_counter() - start

    p_ref = naive_rules(train, val)["family_x_history"]
    timing = {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds}
    if model == "hgb":
        timing["n_boosting_iterations"] = int(pipe.named_steps["model"].n_iter_)

    name = name or f"{model}_" + "_".join(f"{k}{v}" for k, v in params.items())
    config = {"model": model, "model_type": "ml", "params": params, "features": MODEL_FEATURES,
              "n_features": len(MODEL_FEATURES), "excluded_features": EXCLUDED, "train_rows": len(train),
              "train_pos_rate": float(train[TARGET].mean()), "seed": SEED, "hardware": hardware(), "notes": notes}
    if log:
        metrics, tables, url = log_run(name, "model", val, p, train, config, p_ref=p_ref,
                                       data_paths=model_table_paths(), extra_summary=timing)
    else:
        metrics, tables = evaluate(val, p, train, p_ref=p_ref)
        url = None
    metrics.update(timing)
    return {"name": name, "metrics": metrics, "tables": tables, "url": url, "model": pipe, "pred": p}
