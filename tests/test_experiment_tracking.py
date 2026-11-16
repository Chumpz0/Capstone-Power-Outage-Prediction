"""Check upload boundaries and preserved provenance without network access."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from outage_prediction.tracking import upload_saved_runs


class TrackingTests(unittest.TestCase):
    def fixture(self, folder):
        config = {
            "train_years": list(range(2017, 2024)),
            "validation_year": 2024,
            "validation_rows": 2,
            "feature_set": "base",
            "features": ["month"],
            "git_commit": "original-training-commit",
            "fit_seconds": 1,
            "validation_predict_seconds": 0.1,
            "model_bytes": 5,
        }
        (folder / "config.json").write_text(json.dumps(config))
        (folder / "validation_metrics.json").write_text(
            json.dumps({"rows": 2, "pr_auc": 0.5})
        )
        (folder / "validation_predictions.csv").write_text(
            "private predictions stay local"
        )
        (folder / "model.joblib").write_bytes(b"model")
        for table in ["calibration_deciles", "by_family", "by_fema_region", "by_year"]:
            (folder / f"{table}.csv").write_text("rows,value\n2,0.5\n")
        return config

    def test_preserves_training_commit_and_excludes_predictions(self):
        import wandb

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            run = MagicMock(url="https://tracking.example/run")
            artifact = MagicMock()
            remote = MagicMock(state="finished", summary={"pr_auc": 0.5})
            with (
                patch.object(wandb, "init", return_value=run) as init,
                patch.object(wandb, "Artifact", return_value=artifact),
                patch.object(wandb, "Table"),
                patch.object(wandb, "Api") as api,
            ):
                api.return_value.run.return_value = remote
                first = upload_saved_runs(folder)
                second = upload_saved_runs(folder)
            self.assertEqual(first[0]["run_id"], second[0]["run_id"])
            self.assertEqual(
                init.call_args.kwargs["config"]["git_commit"],
                "original-training-commit",
            )
            self.assertEqual(init.call_args.kwargs["mode"], "online")
            self.assertEqual(
                {c.kwargs["name"] for c in artifact.add_file.call_args_list},
                {"model.joblib", "original_config.json", "validation_metrics.json"},
            )
            self.assertTrue((folder / "wandb_uploads.json").exists())

    def test_rejects_smoke_and_wrong_year_before_upload(self):
        import wandb

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = self.fixture(folder)
            for changes in [{"smoke_test": True}, {"validation_year": 2025}]:
                (folder / "config.json").write_text(json.dumps({**config, **changes}))
                with patch.object(wandb, "init") as init:
                    with self.assertRaises(ValueError):
                        upload_saved_runs(folder)
                    init.assert_not_called()

    def test_comparison_attaches_its_own_metrics_and_preflights_all_tables(self):
        import wandb
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = self.fixture(folder)
            config.pop("feature_set")
            config.pop("features")
            config["feature_sets"] = {"first": ["month"], "second": ["month"]}
            (folder / "config.json").write_text(json.dumps(config))
            rows = []
            for name, ap in [("first", 0.4), ("second", 0.6)]:
                sub = folder / name
                sub.mkdir()
                self.fixture(sub)
                (sub / "metrics.json").write_text(json.dumps({"rows": 2, "pr_auc": ap}))
                pd.DataFrame({"mean_pred": [0.1, 0.8], "observed": [0.0, 1.0]}).to_csv(
                    sub / "calibration_deciles.csv", index=False
                )
                rows.append(
                    {
                        "model": name,
                        "fit_seconds": 1.0,
                        "validation_predict_seconds": 0.1,
                        "model_bytes": 5,
                    }
                )
            pd.DataFrame(rows).to_csv(folder / "comparison.csv", index=False)
            artifacts = [MagicMock(), MagicMock()]
            with (
                patch.object(
                    wandb,
                    "init",
                    return_value=MagicMock(url="https://tracking.example/run"),
                ),
                patch.object(wandb, "Artifact", side_effect=artifacts),
                patch.object(wandb, "Table"),
                patch.object(wandb, "Image"),
                patch.object(wandb, "Api") as api,
            ):
                api.return_value.run.side_effect = [
                    MagicMock(state="finished", summary={"pr_auc": 0.4}),
                    MagicMock(state="finished", summary={"pr_auc": 0.6}),
                ]
                upload_saved_runs(folder)
            for name, artifact in zip(config["feature_sets"], artifacts):
                paths = [
                    Path(call.args[0])
                    for call in artifact.add_file.call_args_list
                    if call.kwargs["name"] == "validation_metrics.json"
                ]
                self.assertEqual(paths, [folder / name / "metrics.json"])
            (folder / "second/by_year.csv").unlink()
            with patch.object(wandb, "init") as init:
                with self.assertRaises(FileNotFoundError):
                    upload_saved_runs(folder)
                init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
