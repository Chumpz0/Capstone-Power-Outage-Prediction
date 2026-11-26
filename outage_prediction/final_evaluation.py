"""Freeze a retrospective predictor before explicitly scoring sealed 2025 outcomes.

The freeze and benchmark actions never resolve holdout labels. Score has no fit,
calibration, threshold-selection, rebuilding, or upload path.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss

from .config import (FAMILIES, FEATURES, HISTORY_FEATURES, N_BOOT, ROOT, SEED, ProjectPaths,
                     data_version, git_commit, sha256, staged_directory)
from .evaluation import (calibration_deciles, evaluate, family_rates, naive_rules,
                         precision_at_top, recall_at_top, validate_features, validate_frame,
                         validate_probabilities)
from .train import load_model_data

KEYS = ["fips", "episode_id"]
CUTOFF = pd.Timestamp("2024-07-01", tz="UTC")
TOLERANCE = 1e-12
LIMITATIONS = [
    "Held-out retrospective performance, not demonstrated advance-warning forecasting.",
    "Storm-family counts summarize whole episodes and can include reports after t0.",
    "Training histories use later training years and full-training smoothing priors.",
    "Missing outage readings may become zero under a daily coverage heuristic.",
    "Labels use 24 hourly bins from floor_hour(t0), including part of the preceding hour.",
    "2024 has been used in prior development; calibration comparison is a development estimate.",
    "Bootstrap intervals condition on frozen predictions and exclude training/selection uncertainty.",
]


def write_json(path, value):
    """Write portable JSON, rejecting NaN rather than silently publishing invalid metrics."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def timestamp():
    """Record UTC execution timestamps independently of the host's local timezone."""
    return datetime.now(timezone.utc).isoformat()


def ranking_metrics(y, p):
    """Small selection contract shared by the pre-test calibration comparison."""
    y, p = np.asarray(y), np.asarray(p)
    return {"pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "recall_top10": float(recall_at_top(y, p))}


def prefer_calibrated(raw, calibrated):
    """Require strictly better Brier and no ranking/recall loss beyond numeric tolerance."""
    return bool(calibrated["brier"] < raw["brier"]
                and calibrated["pr_auc"] >= raw["pr_auc"] - TOLERANCE
                and calibrated["recall_top10"] >= raw["recall_top10"] - TOLERANCE)


def choose_calibration(model, validation, features):
    """Fit sigmoid on Jan-Jun only, purge shared episodes, compare on Jul-Dec only.

    The returned selected estimator is never refit on the comparison period.
    Infeasible/invalid calibration retains the unchanged raw predictor with a reason.
    """
    validate_frame(validation, "validation", features)
    times = pd.to_datetime(validation.t0, utc=True, errors="raise")
    if times.isna().any() or not times.dt.year.eq(2024).all():
        raise ValueError("Calibration requires valid 2024 t0 values")
    later = times.ge(CUTOFF)
    shared = validation.loc[later, "episode_id"]
    early = ~later & ~validation.episode_id.isin(shared)
    membership = validation[KEYS + ["year", "t0"]].copy()
    membership["role"] = np.where(later, "comparison", np.where(early, "calibration_fit", "purged"))
    fit, compare = validation.loc[early], validation.loc[later]
    decision = {"selected": "raw", "cutoff_utc": CUTOFF.isoformat(),
                "fit_rows": len(fit), "comparison_rows": len(compare),
                "purged_rows": int((~later & ~early).sum()),
                "method": "sigmoid on FrozenEstimator; base model unchanged",
                "rule": "strictly lower Brier; AP and top10 recall no lower within 1e-12",
                "refit_after_comparison": False, "comparison_metrics": {}}
    tables = {"calibration_membership": membership}
    if any(set(frame.target.unique()) != {0, 1} for frame in (fit, compare)):
        decision["reason"] = "Both calibration periods must contain both classes; retained raw."
        return model, decision, tables
    raw = validate_probabilities(compare, model.predict_proba(compare[features])[:, 1])
    decision["comparison_metrics"]["raw"] = ranking_metrics(compare.target, raw)
    tables["comparison_raw_deciles"] = calibration_deciles(compare.target.to_numpy(), raw)
    try:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
        calibrated.fit(fit[features], fit.target)
        p = validate_probabilities(compare, calibrated.predict_proba(compare[features])[:, 1])
    except (ValueError, FloatingPointError) as error:
        decision["reason"] = f"Calibration infeasible/invalid; retained raw: {error}"
        return model, decision, tables
    decision["comparison_metrics"]["sigmoid"] = ranking_metrics(compare.target, p)
    tables["comparison_sigmoid_deciles"] = calibration_deciles(compare.target.to_numpy(), p)
    if prefer_calibrated(decision["comparison_metrics"]["raw"],
                         decision["comparison_metrics"]["sigmoid"]):
        decision.update(selected="sigmoid", reason="Passed the fixed calibration selection rule.")
    else:
        decision["reason"] = "Sigmoid did not pass the fixed selection rule; retained raw."
    return (calibrated if decision["selected"] == "sigmoid" else model), decision, tables


def reference_parameters(train):
    """Freeze only training-derived family rates and prevalence, never evaluation outcomes."""
    return {"positive_rate": float(train.target.mean()),
            "family_rates": {k: float(v) if pd.notna(v) else None
                             for k, v in family_rates(train).items()},
            "history": "released 2017-2023-derived hist_outage_rate"}


def reference_predict(frame, parameters):
    """Reproduce family_x_history from frozen rates and the released training-only history."""
    present = np.column_stack([
        np.where(frame[f"n_{family}"] > 0,
                 parameters["family_rates"][family] if parameters["family_rates"][family] is not None else np.nan,
                 np.nan) for family in FAMILIES
    ])
    if np.isnan(present).all(axis=1).any() or not 0 < parameters["positive_rate"] < 1:
        raise ValueError("Unsupported reference family or invalid training prevalence")
    history = validate_probabilities(frame, frame.hist_outage_rate, "history")
    return validate_probabilities(frame, np.clip(np.nanmax(present, axis=1) * history
                                               / parameters["positive_rate"], 0, 1), "reference")


def validate_test_features(frame, development, expected_rows):
    """Check unlabeled test inputs without requiring or resolving any test outcomes."""
    if any(c == "target" or c.startswith("retro_") for c in frame):
        raise ValueError("Test features must not contain outcomes")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} test rows; found {len(frame)}")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Test keys must be nonmissing and unique")
    if not frame.year.eq(2025).all() or not pd.to_datetime(frame.t0, utc=True).dt.year.eq(2025).all():
        raise ValueError("Test features must contain 2025 only")
    if set(frame.episode_id) & set(development):
        raise ValueError("Test and development share episodes")
    validate_features(frame, FEATURES, "test")


def coverage_sensitivity(validation, probabilities, observations):
    """Compare fixed 2024 predictions on a stricter subset; never alter released labels."""
    if observations.duplicated(["fips", "day"]).any():
        raise ValueError("Coverage observations have duplicate county/day keys")
    daily = observations.assign(day=pd.to_datetime(observations.day, utc=True).dt.normalize())
    readings = daily.set_index(["fips", "day"]).rows_non_null
    start = pd.to_datetime(validation.t0, utc=True).dt.floor("h")
    end = start + pd.Timedelta(hours=23)
    masks = [readings.reindex(pd.MultiIndex.from_arrays([validation.fips, dates.dt.normalize()]))
             .fillna(0).to_numpy() > 0 for dates in (start, end)]
    strict = masks[0] & masks[1]
    rows = []
    for name, mask in [("all_validation", np.ones(len(validation), dtype=bool)),
                       ("county_readings_on_both_dates", strict)]:
        y, p = validation.target.to_numpy()[mask], np.asarray(probabilities)[mask]
        rows.append({"subset": name, "rows": int(mask.sum()),
                     "positive_rate": float(y.mean()) if len(y) else None,
                     **(ranking_metrics(y, p) if set(y) == {0, 1} else {})})
    return {"definition": "At least one county non-null reading on each label-window UTC date.",
            "interpretation": "Exploratory subset comparison, not continuous coverage or relabeled outcomes.",
            "excluded_rows": int((~strict).sum()), "results": rows}


def freeze(run_dir, output_dir, paths=None, *, expected_rows=32098, n_boot=N_BOOT,
           observations=None):
    """Verify a saved run, select calibration, and publish label-free test predictions."""
    paths, run_dir, output_dir = paths or ProjectPaths(offline=True), Path(run_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if expected_rows < 2 or n_boot < 1:
        raise ValueError("Positive bootstrap draws and at least two test rows are required")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("smoke_test"):
        raise ValueError("Final evaluation requires a real-data saved run")
    if config["features"] != FEATURES:
        raise ValueError("Saved feature order differs from the released selected model")
    train, validation, sources = load_model_data(paths)
    for frame, name in [(train, "train"), (validation, "validation")]:
        validate_frame(frame, name)
    actual_sources = data_version(sources)["sources"]
    for source in actual_sources:
        recorded = next(s for s in config["sources"] if s["logical_path"] == source["logical_path"])
        if not recorded.get("sha256") or recorded["sha256"] != source.get("sha256"):
            raise ValueError("Saved training/validation data hashes do not match")
    model = joblib.load(run_dir / "model.joblib")
    raw = validate_probabilities(validation, model.predict_proba(validation[FEATURES])[:, 1])
    saved = pd.read_csv(run_dir / "validation_predictions.csv", dtype={"fips": str})
    if not saved[KEYS + ["year"]].equals(validation[KEYS + ["year"]].reset_index(drop=True)):
        raise ValueError("Saved validation keys differ")
    np.testing.assert_allclose(raw, saved.probability, rtol=1e-12, atol=1e-12)
    reproduced, _ = evaluate(validation, raw, train)
    recorded = json.loads((run_dir / "validation_metrics.json").read_text())
    for key, value in recorded.items():
        if isinstance(value, (float, int)):
            np.testing.assert_allclose(reproduced[key], value, rtol=1e-12, atol=1e-12)
        elif reproduced[key] != value:
            raise ValueError(f"Saved validation metric differs: {key}")
    predictor, calibration, tables = choose_calibration(model, validation, FEATURES)
    # FrozenEstimator must not change the base model's predictions.
    np.testing.assert_array_equal(raw, model.predict_proba(validation[FEATURES])[:, 1])
    selected_validation = validate_probabilities(validation, predictor.predict_proba(validation[FEATURES])[:, 1])
    if observations is None:
        coverage_source = paths.resolve("processed/county_observed_daily.parquet")
        observations = pd.read_parquet(
            coverage_source.path, columns=["fips", "day", "rows_any", "null_readings"],
            filters=[("day", ">=", datetime(2024, 1, 1).date()),
                     ("day", "<", datetime(2025, 1, 1).date())],
        )
        observations["rows_non_null"] = observations.rows_any - observations.null_readings
    sensitivity = coverage_sensitivity(validation, selected_validation, observations)
    test_source = paths.resolve("processed/model_table/test_features.parquet")
    test = pd.read_parquet(test_source.path)
    development = sorted(set(train.episode_id) | set(validation.episode_id))
    validate_test_features(test, development, expected_rows)
    from .nested_cv import rebuild_history
    history_train, history_test = rebuild_history(
        train, test.assign(target=np.nan, retro_peak_24h=np.nan)
    )
    for released, rebuilt in [(train, history_train), (test, history_test)]:
        np.testing.assert_allclose(released[HISTORY_FEATURES], rebuilt[HISTORY_FEATURES],
                                   rtol=1e-12, atol=1e-12)
    reference = reference_parameters(train)
    np.testing.assert_allclose(reference_predict(validation, reference),
                               naive_rules(train, validation)["family_x_history"], rtol=1e-12, atol=1e-12)
    predictions = test[KEYS + ["year"]].copy()
    predictions["probability"] = validate_probabilities(test, predictor.predict_proba(test[FEATURES])[:, 1])
    predictions["reference_probability"] = reference_predict(test, reference)
    with staged_directory(output_dir) as stage:
        joblib.dump(predictor, stage / "predictor.joblib")
        reloaded = joblib.load(stage / "predictor.joblib")
        np.testing.assert_allclose(reloaded.predict_proba(test[FEATURES])[:, 1], predictions.probability,
                                   rtol=1e-12, atol=1e-12)
        shutil.copyfile(run_dir / "model.joblib", stage / "base_model.joblib")
        shutil.copyfile(run_dir / "config.json", stage / "original_config.json")
        test.to_parquet(stage / "test_features.parquet", index=False)
        predictions.to_parquet(stage / "test_predictions.parquet", index=False)
        write_json(stage / "reference.json", reference)
        write_json(stage / "development_episodes.json", [int(x) for x in development])
        write_json(stage / "calibration.json", calibration)
        write_json(stage / "coverage_sensitivity.json", sensitivity)
        write_json(stage / "reproduced_validation_metrics.json", reproduced)
        for name, table in tables.items():
            table.to_csv(stage / f"{name}.csv", index=False)
        source_dir = stage / "source"
        source_dir.mkdir()
        for source in (ROOT / "outage_prediction").glob("*.py"):
            shutil.copyfile(source, source_dir / source.name)
        manifest = {
            "frozen_at_utc": timestamp(), "git_commit": git_commit(),
            "original_run": str(run_dir), "original_model_sha256": sha256(run_dir / "model.joblib"),
            "features": FEATURES, "expected_test_rows": expected_rows,
            "sources": actual_sources + [test_source.metadata()],
            "calibration": calibration, "limitations": LIMITATIONS,
            "bootstrap": {"n_boot": n_boot, "seed": SEED, "cluster": "episode_id", "interval": "95% percentile AP gain"},
            "success_criteria": ["AP > reference", "Brier <= reference", "top10 recall >= reference"],
            "top_share": 0.10, "tie_policy": "stable feature-file row order",
            "brier_skill": "1 - Brier / (evaluation_prevalence * (1-evaluation_prevalence))",
            "test_labels_accessed": False,
            "test_histories_verified_from_training_only": True,
            "artifacts": {p.relative_to(stage).as_posix(): sha256(p)
                          for p in stage.rglob("*") if p.is_file()},
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
    return manifest


def verify_bundle(directory):
    """Reject artifact or scoring-source drift before any holdout label access."""
    directory = Path(directory)
    if sha256(directory / "manifest.json") != (directory / "manifest.sha256").read_text().strip():
        raise ValueError("Frozen manifest hash mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    for relative, expected in manifest["artifacts"].items():
        file = (directory / relative).resolve()
        if not file.is_relative_to(directory.resolve()) or sha256(file) != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {relative}")
    current = {p.name: sha256(p) for p in (ROOT / "outage_prediction").glob("*.py")}
    frozen = {Path(k).name: v for k, v in manifest["artifacts"].items() if k.startswith("source/")}
    if current != frozen:
        raise ValueError("Scoring source differs from the frozen source snapshot")
    return manifest


def align_labels(features, predictions, labels):
    """Join exact key sets one-to-one in feature order, never silently discarding rows."""
    expected = pd.MultiIndex.from_frame(features[KEYS])
    aligned = []
    for name, frame in [("predictions", predictions), ("labels", labels)]:
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError(f"{name}: duplicate or missing keys")
        index = pd.MultiIndex.from_frame(frame[KEYS])
        if len(index) != len(expected) or not expected.isin(index).all():
            raise ValueError(f"{name}: key set differs from test features")
        aligned.append(frame.set_index(KEYS).loc[expected].reset_index())
    pred, outcomes = aligned
    if not pred.year.eq(2025).all():
        raise ValueError("Saved prediction years must be 2025")
    if outcomes.target.isna().any() or set(outcomes.target.unique()) != {0, 1}:
        raise ValueError("Labels must contain both binary classes without nulls")
    labeled = features.assign(target=outcomes.target.to_numpy())
    validate_frame(labeled, "test")
    for column in ["probability", "reference_probability"]:
        validate_probabilities(labeled, pred[column].to_numpy(), column)
    return labeled, pred


def score(frozen_dir, output_dir, paths=None, *, allow_holdout=False):
    """Score frozen keyed predictions once labels are explicitly unsealed; never fit."""
    if not allow_holdout:
        raise ValueError("Final scoring requires explicit --allow-holdout")
    frozen_dir, output_dir = Path(frozen_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = verify_bundle(frozen_dir)
    paths = paths or ProjectPaths(offline=True)
    features = pd.read_parquet(frozen_dir / "test_features.parquet")
    predictions = pd.read_parquet(frozen_dir / "test_predictions.parquet")
    development = json.loads((frozen_dir / "development_episodes.json").read_text())
    validate_test_features(features, development, manifest["expected_test_rows"])
    train_source = paths.resolve("processed/model_table/train.parquet")
    recorded_train = next(s for s in manifest["sources"] if s["logical_path"] == train_source.logical_path)
    if train_source.metadata().get("sha256") != recorded_train.get("sha256"):
        raise ValueError("Training source changed since freeze")
    train = pd.read_parquet(train_source.path)
    reference = json.loads((frozen_dir / "reference.json").read_text())
    np.testing.assert_allclose(predictions.reference_probability, reference_predict(features, reference),
                               rtol=1e-12, atol=1e-12)
    # This is the sole real holdout-label resolution/read in the final-evaluation module.
    label_source = paths.resolve("holdout/test_labels.parquet", allow_holdout=True)
    label_accessed_at = timestamp()
    labels = pd.read_parquet(label_source.path, columns=KEYS + ["target"])
    labeled, aligned = align_labels(features, predictions, labels)
    p, p_ref = aligned.probability.to_numpy(), aligned.reference_probability.to_numpy()
    metrics, tables = evaluate(labeled, p, train, p_ref=p_ref, split="test",
                               bootstrap=True, n_boot=manifest["bootstrap"]["n_boot"])
    # Recompute primary metrics with an independent sklearn call path.
    independent = ranking_metrics(labeled.target, p)
    independent["precision_top10"] = float(precision_at_top(labeled.target.to_numpy(), p))
    for key, value in independent.items():
        np.testing.assert_allclose(metrics[key], value, rtol=1e-12, atol=1e-12)
    from .plots import calibration_plot, pr_curve_plot
    import matplotlib.pyplot as plt
    with staged_directory(output_dir) as stage:
        joined = aligned.assign(target=labeled.target.to_numpy())
        joined.to_parquet(stage / "test_predictions.parquet", index=False)
        write_json(stage / "test_metrics.json", metrics)
        for name, table in tables.items():
            table.to_csv(stage / f"{name}.csv", index=False)
        for name, figure in [("calibration", calibration_plot(tables["calibration_deciles"], "2025 frozen predictor")),
                             ("precision_recall", pr_curve_plot(labeled.target.to_numpy(), p, p_ref, "2025 held-out retrospective evaluation"))]:
            try:
                figure.savefig(stage / f"{name}.png", dpi=150, bbox_inches="tight")
            finally:
                plt.close(figure)
        write_json(stage / "provenance.json", {
            "scored_at_utc": timestamp(), "labels_accessed_at_utc": label_accessed_at,
            "frozen_at_utc": manifest["frozen_at_utc"],
            "frozen_manifest_sha256": sha256(frozen_dir / "manifest.json"),
            "labels_sha256": sha256(label_source.path), "rows": len(labeled),
            "bootstrap": manifest["bootstrap"], "independent_metrics": independent,
            "calibration": manifest["calibration"], "limitations": LIMITATIONS,
            "artifacts": {p.name: sha256(p) for p in stage.iterdir() if p.is_file()},
        })
        shutil.copyfile(frozen_dir / "manifest.json", stage / "frozen_manifest.json")
    return metrics


def process_memory():
    """Return Windows working set/peak working set, or Linux current RSS/peak RSS."""
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ["PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"rss_bytes": counters.WorkingSetSize, "peak_rss_bytes": counters.PeakWorkingSetSize}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current = int(Path("/proc/self/statm").read_text().split()[1]) * __import__("os").sysconf("SC_PAGE_SIZE")
    return {"rss_bytes": current, "peak_rss_bytes": int(peak * 1024)}


def benchmark(frozen_dir, output_dir, paths=None):
    """Measure 30 warm predictions in a dedicated CLI process, without label access."""
    frozen_dir, output_dir = Path(frozen_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = verify_bundle(frozen_dir)
    baseline = process_memory()
    predictor = joblib.load(frozen_dir / "predictor.joblib")
    loaded = process_memory()
    source = (paths or ProjectPaths(offline=True)).resolve("processed/model_table/validation.parquet")
    recorded = next(s for s in manifest["sources"] if s["logical_path"] == source.logical_path)
    if source.metadata().get("sha256") != recorded.get("sha256"):
        raise ValueError("Benchmark validation data differs from frozen source")
    frame = pd.read_parquet(source.path)[manifest["features"]]
    timings = {}
    for name, batch in [("single_row", frame.iloc[:1]), ("validation_batch", frame)]:
        predictor.predict_proba(batch)
        samples = []
        for _ in range(30):
            started = time.perf_counter()
            predictor.predict_proba(batch)
            samples.append(time.perf_counter() - started)
        timings[name] = {"rows": len(batch), "median_seconds": float(np.median(samples)),
                         "p95_seconds": float(np.percentile(samples, 95)), "samples_seconds": samples}
    import sklearn
    result = {"measured_at_utc": timestamp(), "platform": platform.platform(),
              "processor": platform.processor(), "python": platform.python_version(),
              "sklearn": sklearn.__version__, "warmups_per_workload": 1, "repeats": 30,
              "memory_before_model_load": baseline, "memory_after_model_load": loaded,
              "memory_after_workloads": process_memory(),
              "memory_interpretation": "Whole-process working set and lifetime peak; includes imports, data and predictor, not isolated model RAM.",
              "predictor_bytes": (frozen_dir / "predictor.joblib").stat().st_size,
              "timings": timings, "frozen_manifest_sha256": sha256(frozen_dir / "manifest.json"),
              "test_labels_accessed": False}
    with staged_directory(output_dir) as stage:
        write_json(stage / "benchmark.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("freeze", help="Select calibration and freeze test predictions without labels")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--expected-rows", type=int, default=32098)
    for name in ["score", "benchmark"]:
        action = sub.add_parser(name)
        action.add_argument("--frozen-dir", type=Path, required=True)
        if name == "score":
            action.add_argument("--allow-holdout", action="store_true")
    for action in sub.choices.values():
        action.add_argument("--output-dir", type=Path, required=True)
        action.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    paths = ProjectPaths(offline=args.offline)
    if args.action == "freeze":
        result = freeze(args.run_dir, args.output_dir, paths, expected_rows=args.expected_rows)
        print(json.dumps({"frozen_at_utc": result["frozen_at_utc"], "calibration": result["calibration"],
                          "test_labels_accessed": False}, indent=2))
    elif args.action == "score":
        print(json.dumps(score(args.frozen_dir, args.output_dir, paths, allow_holdout=args.allow_holdout), indent=2))
    else:
        result = benchmark(args.frozen_dir, args.output_dir, paths)
        print(json.dumps({k: v for k, v in result.items() if k != "timings"}, indent=2))


if __name__ == "__main__":
    main()
