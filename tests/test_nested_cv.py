"""Check temporal/episode boundaries and history leakage in nested model selection."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from outage_prediction.config import FEATURES, HISTORY_FEATURES
from outage_prediction.nested_cv import nested_cv, rebuild_history, year_folds
from outage_prediction.train import save_run, synthetic_frames


def frames():
    train, validation = synthetic_frames()
    for frame in (train, validation):
        frame["retro_peak_24h"] = np.where(frame.target.eq(1), 0.1, 0.01)
    return train, validation


class NestedCVTests(unittest.TestCase):
    def test_folds_are_past_only_and_purge_shared_episodes(self):
        train, _ = frames()
        earlier = train.index[train.year.eq(2022)][0]
        later = train.index[train.year.eq(2023)][0]
        train.loc[later, "episode_id"] = train.loc[earlier, "episode_id"]
        folds = year_folds(train, 3)
        self.assertEqual([int(held.year.iloc[0]) for _, held, _ in folds], [2021, 2022, 2023])
        for fitted, held, _ in folds:
            self.assertLess(fitted.year.max(), held.year.min())
            self.assertFalse(set(fitted.episode_id) & set(held.episode_id))
        self.assertEqual(folds[-1][2], 1)
        with self.assertRaises(ValueError):
            year_folds(train, 7)
        with self.assertRaises(ValueError):
            year_folds(train.assign(year=2025), 1)

    def test_histories_ignore_held_labels_peaks_and_released_histories(self):
        train, validation = frames()
        # Repeated counties give nontrivial history sums.
        train["fips"] = np.resize(["00001", "00002"], len(train))
        validation["fips"] = "00001"
        original = train.copy(deep=True)
        left, right = rebuild_history(train, validation)
        changed = validation.assign(target=1-validation.target, retro_peak_24h=0.99)
        corrupted = train.copy()
        corrupted[HISTORY_FEATURES] = 999
        left2, right2 = rebuild_history(corrupted, changed)
        pd.testing.assert_frame_equal(left[HISTORY_FEATURES], left2[HISTORY_FEATURES])
        pd.testing.assert_frame_equal(right[HISTORY_FEATURES], right2[HISTORY_FEATURES])
        pd.testing.assert_frame_equal(train, original)
        self.assertTrue(right.hist_episodes.eq(len(train) // 2).all())

    def test_nested_selection_excludes_2024_and_saves_fitted_model(self):
        train, validation = frames()
        events = []
        result, final_train, final_validation = nested_cv(
            train, validation, n_trials=2, smoke_test=True, progress=events.append,
        )
        changed, _, _ = nested_cv(
            train, validation.assign(target=1-validation.target, retro_peak_24h=0.8),
            n_trials=2, smoke_test=True,
        )
        pd.testing.assert_frame_equal(
            result.tables["nested_cv_trials"], changed.tables["nested_cv_trials"]
        )
        pd.testing.assert_frame_equal(
            result.tables["nested_cv_outer_folds"], changed.tables["nested_cv_outer_folds"]
        )
        trials = result.tables["nested_cv_trials"]
        self.assertEqual(len(trials), 4 * 2 * 3)
        self.assertLess(trials.inner_validation_year.max(), 2024)
        self.assertEqual(sum(e["event"] == "outer_fold_completed" for e in events), 3)
        outer = result.tables["nested_cv_outer_folds"]
        config = result.config["nested_cross_validation"]
        self.assertEqual(config["outer_mean_pr_auc"], outer.pr_auc.mean())
        means = trials[trials.stage.eq("final")].groupby("trial").pr_auc.mean()
        self.assertEqual(config["final_inner_mean_pr_auc"], means.max())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run"
            save_run(result, folder, final_validation, final_train)
            saved = joblib.load(folder / "model.joblib")
            np.testing.assert_allclose(
                saved.predict_proba(final_validation[FEATURES])[:, 1],
                result.predictions.probability,
            )
            self.assertTrue((folder / "nested_cv_predictions.csv").is_file())
            saved_config = json.loads((folder / "config.json").read_text())
            self.assertIn("nested_cross_validation", saved_config)


if __name__ == "__main__":
    unittest.main()
