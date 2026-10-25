"""Reject observation summaries that cannot be tied to current calculation code."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_customer_targets


class SummaryProvenanceTests(unittest.TestCase):
    def run_with_metadata(self, metadata):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "calculation.py").write_text("changed calculation")
            (root / "docs/checkpoint_1_observation_reliability.json").write_text(
                json.dumps(metadata))
            with patch.object(check_customer_targets, "ROOT", root), \
                    patch.object(check_customer_targets, "OUTPUT", root / "output"):
                check_customer_targets.main()

    def test_legacy_summary_requires_rebuild(self):
        with self.assertRaisesRegex(ValueError, "code fingerprints are missing"):
            self.run_with_metadata({"files": []})

    def test_changed_code_rejects_cached_summary(self):
        with self.assertRaisesRegex(ValueError, "calculation.py"):
            self.run_with_metadata({
                "files": [],
                "code_files": [{"path": "calculation.py", "sha256": "outdated"}],
            })


if __name__ == "__main__":
    unittest.main()
