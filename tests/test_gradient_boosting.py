"""Check the consolidated trainer's fixed features and independent upload action."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from outage_prediction import train as trainer


class TrainerTests(unittest.TestCase):
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
