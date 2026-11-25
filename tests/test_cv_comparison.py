"""Verify comparison alignment, fold-local reference scores, and paired intervals."""

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from outage_prediction.compare_cv import compare_predictions
from outage_prediction.evaluation import naive_rules
from outage_prediction.nested_cv import rebuild_history, year_folds
from outage_prediction.train import synthetic_frames


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        train, _ = synthetic_frames()
        train["retro_peak_24h"] = np.where(train.target.eq(1), 0.1, 0.01)
        blocks = []
        for past, held, _ in year_folds(train, 3):
            local_train, local_held = rebuild_history(past, held)
            block = held[["fips", "episode_id", "year", "target"]].copy()
            block["probability"] = naive_rules(local_train, local_held)["family_x_history"]
            blocks.append(block)
        return train, pd.concat(blocks, ignore_index=True)

    def test_identical_model_and_reference_have_zero_paired_gain(self):
        train, predictions = self.fixture()
        result = compare_predictions(train, predictions, n_boot=20)
        self.assertTrue(result.ap_gain.eq(0).all())
        self.assertTrue(result.brier_reduction.eq(0).all())
        self.assertTrue(result.recall_top10_gain.eq(0).all())
        self.assertTrue(result.ap_gain_ci_low.eq(0).all())
        self.assertTrue(result.ap_gain_ci_high.eq(0).all())
        self.assertFalse(result.ap_gain_ci_above_zero.any())
        self.assertEqual(result.iloc[-1].scope, "pooled")
        self.assertAlmostEqual(
            result.iloc[-1].model_ap,
            average_precision_score(predictions.target, predictions.probability),
        )

    def test_saved_rows_are_aligned_by_keys_and_bootstrap_is_reproducible(self):
        train, predictions = self.fixture()
        predictions["probability"] = np.where(predictions.target.eq(1), 0.9, 0.1)
        first = compare_predictions(train, predictions, n_boot=20)
        shuffled = compare_predictions(train, predictions.sample(frac=1, random_state=7), n_boot=20)
        pd.testing.assert_frame_equal(first, shuffled)
        self.assertTrue(first.model_ap.eq(1).all())
        self.assertTrue(first.ap_gain_ci_above_zero.all())

    def test_rejects_mismatched_or_invalid_prediction_rows(self):
        train, predictions = self.fixture()
        for invalid in [
            predictions.iloc[:-1],
            pd.concat([predictions.iloc[:-1], predictions.iloc[[0]]]),
            predictions.assign(target=1-predictions.target),
            predictions.assign(year=2025),
            predictions.assign(probability=np.nan),
        ]:
            with self.assertRaises(ValueError):
                compare_predictions(train, invalid, n_boot=2)


if __name__ == "__main__":
    unittest.main()
