"""Exercise freeze-to-score boundaries with synthetic years and temporary files."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from outage_prediction.config import FEATURES, ProjectPaths
from outage_prediction.evaluation import evaluate
from outage_prediction.final_evaluation import (
    align_labels, choose_calibration, coverage_sensitivity, freeze,
    prefer_calibrated, reference_parameters, reference_predict, score, verify_bundle,
)
from outage_prediction.train import fit_model, save_run, synthetic_frames


class FinalEvaluationTests(unittest.TestCase):
    def fixture(self, root):
        """Prepare a real-schema tiny saved run and separate synthetic holdout labels."""
        train, validation = synthetic_frames()
        for frame in (train, validation):
            frame["retro_peak_24h"] = np.where(frame.target.eq(1), 0.1, 0.01)
        train["t0"] = pd.to_datetime(train.year.astype(str) + "-03-01", utc=True)
        validation["t0"] = pd.date_range("2024-01-01", "2024-12-31", periods=len(validation), tz="UTC")
        test = validation.copy()
        test["year"] = 2025
        test["t0"] = pd.date_range("2025-01-01", "2025-12-31", periods=len(test), tz="UTC")
        test["episode_id"] += 10000
        from outage_prediction.nested_cv import rebuild_history
        train, validation = rebuild_history(train, validation)
        _, test = rebuild_history(train, test)
        processed, holdout = root / "data/processed", root / "data/holdout"
        processed.mkdir(parents=True)
        holdout.mkdir(parents=True)
        train.to_parquet(processed / "train.parquet", index=False)
        validation.to_parquet(processed / "validation.parquet", index=False)
        test.drop(columns=["target", "retro_peak_24h"]).to_parquet(processed / "test_features.parquet", index=False)
        test[["fips", "episode_id", "target"]].to_parquet(holdout / "test_labels.parquet", index=False)
        paths = ProjectPaths(root, offline=True)
        sources = [paths.resolve(f"processed/model_table/{name}.parquet") for name in ["train", "validation"]]
        run = root / "run"
        fitted = fit_model(train, validation, n_estimators=5, sources=sources)
        save_run(fitted, run, validation, train)
        start = validation.t0.dt.floor("h")
        observations = pd.concat([
            pd.DataFrame({"fips": validation.fips, "day": dates.dt.normalize(), "rows_non_null": 1})
            for dates in [start, start + pd.Timedelta(hours=23)]
        ]).drop_duplicates(["fips", "day"])
        interim = root / "data/interim"
        interim.mkdir()
        observations.assign(day=observations.day.dt.date, rows_any=1, null_readings=0).drop(
            columns="rows_non_null"
        ).to_parquet(interim / "county_observed_daily.parquet", index=False)
        return paths, run, train, validation, test, observations, fitted.model

    def test_complete_workflow_keeps_labels_sealed_until_explicit_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, run, train, val, test, observations, _ = self.fixture(root)
            frozen = root / "frozen"
            original_read = pd.read_parquet
            reads = []

            def guarded_read(path, *args, **kwargs):
                reads.append(str(path))
                if "test_labels" in str(path):
                    raise AssertionError("Freeze attempted to read holdout labels")
                return original_read(path, *args, **kwargs)

            with patch("pandas.read_parquet", side_effect=guarded_read):
                manifest = freeze(run, frozen, paths, expected_rows=len(test), n_boot=20)
            self.assertFalse(manifest["test_labels_accessed"])
            self.assertFalse(any("holdout" in path for path in reads))
            self.assertEqual(verify_bundle(frozen), manifest)
            with patch("pandas.read_parquet") as read:
                with self.assertRaisesRegex(ValueError, "allow-holdout"):
                    score(frozen, root / "refused", paths)
                read.assert_not_called()
            # A shuffled label file must score exactly the same keyed rows.
            test[["fips", "episode_id", "target"]].sample(frac=1, random_state=1).to_parquet(
                root / "data/holdout/test_labels.parquet", index=False)
            with patch("sklearn.ensemble.GradientBoostingClassifier.fit", side_effect=AssertionError("No refitting")):
                first = score(frozen, root / "scored", paths, allow_holdout=True)
                second = score(frozen, root / "reproduced", paths, allow_holdout=True)
            self.assertEqual(first, second)
            joined = pd.read_parquet(root / "scored/test_predictions.parquet")
            self.assertEqual(first["rows"], len(test))
            self.assertEqual(first["pr_auc"], average_precision_score(joined.target, joined.probability))
            self.assertEqual(first["brier"], brier_score_loss(joined.target, joined.probability))
            self.assertIn("passes_all_test", first)
            self.assertNotIn("passes_all_validation", first)
            with self.assertRaises(FileExistsError):
                score(frozen, root / "scored", paths, allow_holdout=True)

    def test_calibration_never_refits_base_or_uses_later_labels_for_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, val, _, _, model = self.fixture(Path(tmp))
            original = model.predict_proba(val[FEATURES]).copy()
            # Introduce an episode shared across the fixed cutoff.
            val.loc[0, "episode_id"] = val.episode_id.iloc[-1]
            fits = []
            from sklearn.calibration import CalibratedClassifierCV
            original_fit = CalibratedClassifierCV.fit

            def capture(estimator, x, y, **kwargs):
                fits.append((x.index.tolist(), y.copy()))
                return original_fit(estimator, x, y, **kwargs)

            with (patch("sklearn.ensemble.GradientBoostingClassifier.fit", side_effect=AssertionError("Frozen base")),
                  patch.object(CalibratedClassifierCV, "fit", capture)):
                _, decision, tables = choose_calibration(model, val, FEATURES)
                changed = val.copy()
                changed.loc[changed.t0.ge("2024-07-01"), "target"] ^= 1
                choose_calibration(model, changed, FEATURES)
            self.assertEqual(fits[0][0], fits[1][0])
            pd.testing.assert_series_equal(fits[0][1], fits[1][1])
            self.assertNotIn(0, fits[0][0])
            self.assertTrue(val.loc[fits[0][0], "t0"].lt("2024-07-01").all())
            self.assertEqual(decision["purged_rows"], 1)
            self.assertFalse(decision["refit_after_comparison"])
            self.assertEqual(tables["calibration_membership"].role.iloc[0], "purged")
            np.testing.assert_array_equal(original, model.predict_proba(val[FEATURES]))
            _, fallback, _ = choose_calibration(model, val.assign(target=val.t0.ge("2024-07-01").astype(int)), FEATURES)
            self.assertEqual(fallback["selected"], "raw")
            self.assertIn("both classes", fallback["reason"])

    def test_selection_rule(self):
        raw = {"pr_auc": 0.6, "brier": 0.1, "recall_top10": 0.4}
        self.assertTrue(prefer_calibrated(raw, {**raw, "brier": 0.09}))
        for values in [raw, {**raw, "brier": 0.11},
                       {**raw, "brier": 0.09, "pr_auc": 0.59},
                       {**raw, "brier": 0.09, "recall_top10": 0.39}]:
            self.assertFalse(prefer_calibrated(raw, values))

    def test_alignment_and_year_boundaries_reject_bad_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, train, val, test, _, model = self.fixture(Path(tmp))
            features = test.drop(columns=["target", "retro_peak_24h"])
            p = model.predict_proba(features[FEATURES])[:, 1]
            pred = test[["fips", "episode_id", "year"]].assign(probability=p, reference_probability=0.2)
            labels = test[["fips", "episode_id", "target"]]
            for bad in [labels.iloc[:-1], pd.concat([labels, labels.iloc[:1]]),
                        labels.assign(fips="99999"), labels.assign(target=2), labels.assign(target=np.nan)]:
                with self.assertRaises(ValueError):
                    align_labels(features, pred, bad)
            for bad in [pred.assign(probability=np.nan), pred.assign(probability=1.01), pred.assign(year=2024)]:
                with self.assertRaises(ValueError):
                    align_labels(features, bad, labels)
            with self.assertRaises(ValueError):
                evaluate(test, p, train)
            with self.assertRaises(ValueError):
                evaluate(val, p, train, split="test")
            with self.assertRaises(ValueError):
                fit_model(train, test)
            with self.assertRaises(ValueError):
                evaluate(test.assign(episode_id=train.episode_id.iloc[0]), p, train, split="test")

    def test_tampering_and_development_overlap_fail_before_label_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, run, _, _, test, observations, _ = self.fixture(root)
            frozen = root / "frozen"
            freeze(run, frozen, paths, expected_rows=len(test), n_boot=20, observations=observations)
            with (frozen / "reference.json").open("a") as stream:
                stream.write(" ")
            with patch("pandas.read_parquet") as read:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    score(frozen, root / "scored", paths, allow_holdout=True)
                read.assert_not_called()
            from outage_prediction.final_evaluation import validate_test_features
            with self.assertRaisesRegex(ValueError, "share episodes"):
                validate_test_features(test.drop(columns=["target", "retro_peak_24h"]), [test.episode_id.iloc[0]], len(test))
            with self.assertRaisesRegex(ValueError, "Expected"):
                validate_test_features(test.drop(columns=["target", "retro_peak_24h"]), [], len(test) + 1)
            with self.assertRaisesRegex(ValueError, "outcomes"):
                validate_test_features(test, [], len(test))

    def test_reference_and_coverage_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, train, val, _, observations, model = self.fixture(Path(tmp))
            from outage_prediction.evaluation import naive_rules
            np.testing.assert_array_equal(reference_predict(val, reference_parameters(train)),
                                          naive_rules(train, val)["family_x_history"])
            p = model.predict_proba(val[FEATURES])[:, 1]
            summary = coverage_sensitivity(val, p, observations)
            self.assertEqual(summary["excluded_rows"], 0)
            summary = coverage_sensitivity(val, p, observations.loc[observations.fips.ne(val.fips.iloc[0])])
            self.assertEqual(summary["excluded_rows"], 1)


if __name__ == "__main__":
    unittest.main()
