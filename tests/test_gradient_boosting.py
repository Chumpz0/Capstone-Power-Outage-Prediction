"""Check the consolidated trainer's fixed features and independent upload action."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from outage_prediction import train as trainer


class TrainerTests(unittest.TestCase):
    def test_tuning_is_reproducible_and_selects_best_validation_ap(self):
        import numpy as np
        import pandas as pd

        train, validation = trainer.synthetic_frames()
        space = {"n_estimators": [2, 5], "learning_rate": [0.03, 0.1], "max_depth": [1]}
        before = train.copy(deep=True)
        with patch.object(trainer, "fit_model", wraps=trainer.fit_model) as fit:
            first = trainer.tune_model(train, validation, n_trials=3, parameter_space=space)
        self.assertEqual(fit.call_count, 3)
        for call in fit.call_args_list:
            self.assertIs(call.args[0], train)
            self.assertIs(call.args[1], validation)
            self.assertNotIn("bootstrap", call.kwargs)
        second = trainer.tune_model(train, validation, n_trials=3, parameter_space=space)
        columns = ["trial", "n_estimators", "learning_rate", "max_depth", "pr_auc", "selected"]
        trials = first.tables["hyperparameter_trials"]
        pd.testing.assert_frame_equal(trials[columns], second.tables["hyperparameter_trials"][columns])
        pd.testing.assert_frame_equal(before, train)
        self.assertEqual(trials.selected.sum(), 1)
        self.assertEqual(first.metrics["pr_auc"], trials.pr_auc.max())
        selected = trials.loc[trials.selected].iloc[0]
        self.assertEqual(selected.trial, trials.loc[trials.pr_auc.idxmax(), "trial"])
        for name in space:
            self.assertEqual(first.model[-1].get_params()[name], selected[name])
        np.testing.assert_allclose(
            first.model.predict_proba(validation[trainer.FEATURES])[:, 1],
            first.predictions.probability,
        )

    def test_tuning_validates_options_and_preserves_split_boundaries(self):
        train, validation = trainer.synthetic_frames()
        for options in [
            {"n_trials": 0}, {"n_trials": 1.5},
            {"parameter_space": {}},
            {"parameter_space": {"max_depth": []}},
            {"parameter_space": {"random_state": [1]}},
            {"parameter_space": {"max_depth": [1.5]}},
            {"parameter_space": {"learning_rate": [float("nan")]}},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                trainer.tune_model(train, validation, **options)
        for bad_validation in [validation.assign(year=2025), validation.assign(episode_id=train.episode_id.iloc[0])]:
            with self.assertRaises(ValueError):
                trainer.tune_model(train, bad_validation, n_trials=1, smoke_test=True)

    def test_tuned_smoke_run_saves_search_and_bootstraps_only_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "tuned"
            with (
                patch.object(sys, "argv", ["trainer", "--smoke-test", "--tune", "--n-trials", "2", "--bootstrap", "--output-dir", str(folder)]),
                patch("outage_prediction.evaluation.bootstrap_gain", return_value=(0.1, 0.2)) as boot,
            ):
                trainer.main()
            boot.assert_called_once()
            config = json.loads((folder / "config.json").read_text())
            search = config["hyperparameter_optimization"]
            self.assertEqual(search["completed_trials"], 2)
            self.assertEqual(search["selection_split"], "validation_2024")
            self.assertTrue(config["bootstrap"])
            self.assertTrue((folder / "hyperparameter_trials.csv").is_file())
            self.assertTrue((folder / "model.joblib").is_file())

    def test_cli_rejects_incompatible_tuning_options_before_loading(self):
        for options in [
            ["baseline", "--tune"], ["--tune", "--n-trials", "0"],
            ["--n-trials", "2"], ["--tune", "--max-depth", "2"],
            ["--upload-existing", "saved", "--tune"],
        ]:
            with (
                self.subTest(options=options),
                patch.object(sys, "argv", ["trainer", *options]),
                patch.object(trainer, "load_model_data") as load,
                self.assertRaises(SystemExit),
            ):
                trainer.main()
            load.assert_not_called()

    def test_fixed_features_match_selected_saved_candidate(self):
        self.assertEqual(len(trainer.FEATURES), 19)
        self.assertEqual(len(set(trainer.FEATURES)), 19)
        self.assertTrue(
            set(trainer.STORM_FEATURES + trainer.HISTORY_FEATURES)
            <= set(trainer.FEATURES)
        )
        self.assertFalse(any(f.startswith("retro_") for f in trainer.FEATURES))

    def test_upload_only_never_loads_data_or_builds_model(self):
        with (
            patch.object(sys, "argv", ["trainer", "--upload-existing", "saved-run"]),
            patch(
                "outage_prediction.tracking.upload_saved_runs", return_value=[]
            ) as upload,
            patch.object(trainer, "build_model") as build,
            patch("pandas.read_parquet") as read,
        ):
            trainer.main()
            upload.assert_called_once_with(Path("saved-run"))
            build.assert_not_called()
            read.assert_not_called()

    def test_smoke_saves_selected_pipeline_and_evaluation(self):
        import joblib
        import numpy as np
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run"
            with patch.object(
                sys, "argv", ["trainer", "--smoke-test", "--output-dir", str(folder)]
            ):
                trainer.main()
            config = json.loads((folder / "config.json").read_text())
            self.assertEqual(config["features"], trainer.FEATURES)
            self.assertEqual(config["feature_set"], "base_storm_history")
            self.assertTrue((folder / "validation_metrics.json").is_file())
            _, val = trainer.synthetic_frames()
            saved = joblib.load(folder / "model.joblib")
            np.testing.assert_allclose(
                saved.predict_proba(val[trainer.FEATURES])[:, 1],
                pd.read_csv(folder / "validation_predictions.csv").probability,
            )


if __name__ == "__main__":
    unittest.main()
