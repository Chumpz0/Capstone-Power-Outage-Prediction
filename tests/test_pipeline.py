"""Small fixtures exercise the actual raw-to-feature pipeline without project data."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from outage_prediction.config import ProjectPaths, staged_directory, data_version
from outage_prediction.dataset import (
    prepare,
    connection,
    attach_prepared,
    load_raw,
    clean_outages,
)
from outage_prediction.features import (
    build_table,
    add_history,
    validate_table,
    assemble_table,
)
from outage_prediction.analysis import psi, feature_drift, audit_raw
from outage_prediction.evaluation import evaluate, family_rule
from outage_prediction.train import synthetic_frames


def raw_fixture(root):
    """Write three synthetic years, two counties, and the minimal reference inputs."""
    raw = root / "data/raw/combined"
    external = root / "data/external"
    raw.mkdir(parents=True)
    (external / "eaglei_archive").mkdir(parents=True)
    (root / "data/raw/census_counties").mkdir(parents=True)
    pd.DataFrame(
        {
            "SUMLEV": ["050", "050"],
            "STATE": ["13", "13"],
            "COUNTY": ["001", "003"],
            "STNAME": ["Georgia"] * 2,
            "CTYNAME": ["Alpha", "Beta"],
            "POPESTIMATE2023": ["200", "200"],
        }
    ).to_csv(root / "data/raw/census_counties/co-est2023-alldata.csv", index=False)
    pd.DataFrame({"County_FIPS": ["13001", "13003"], "Customers": [100, 100]}).to_csv(
        external / "eaglei_archive/MCC.csv", index=False
    )
    pd.DataFrame(
        {
            "region_fips": pd.Series(dtype=str),
            "county_fips": pd.Series(dtype=str),
            "weight": pd.Series(dtype=float),
        }
    ).to_csv(external / "ct_region_to_county_weights.csv", index=False)
    pd.DataFrame(
        {
            "county_fips": ["13001", "13003"],
            "state": ["GA", "GA"],
            "state_zone": ["GA001", "GA003"],
            "effective_date": pd.to_datetime(["2014-01-01"] * 2),
            "next_effective_date": pd.to_datetime([None, None]),
            "partial_county": [False, False],
            "counties_in_zone": [1, 1],
        }
    ).to_parquet(external / "nws_zone_county_versions.parquet", index=False)
    pd.DataFrame(
        {
            "year": ["01/01/18", "01/01/22"],
            "state": ["GA", "GA"],
            "max_pct_covered": [1.0, 1.0],
        }
    ).to_csv(external / "eaglei_archive/coverage_history.csv", index=False)
    pd.DataFrame({"fema": [4, 4], "year": [2018, 2022], "DQI": [90.0, 90.0]}).to_csv(
        external / "eaglei_archive/DQI.csv", index=False
    )
    outages = []
    storms = []
    for year in [2017, 2024, 2025]:
        for county in ["13001", "13003"]:
            for ts in pd.date_range(
                f"{year}-01-01", f"{year}-01-04", freq="15min", inclusive="left"
            ):
                out = (
                    10
                    if county == "13001" and ts == pd.Timestamp(f"{year}-01-02 10:15")
                    else 0
                )
                outages.append(
                    {
                        "SOURCE_YEAR": str(year),
                        "fips_code": county,
                        "state": "Georgia",
                        "customers_out": str(out),
                        "run_start_time": str(ts),
                    }
                )
            storms.append(
                {
                    "EVENT_ID": str(year * 100 + int(county[-3:])),
                    "EPISODE_ID": str(year * 100 + int(county[-3:])),
                    "SOURCE_YEAR": str(year),
                    "EVENT_TYPE": "Thunderstorm Wind",
                    "CZ_TYPE": "Z" if county == "13003" else "C",
                    "STATE_FIPS": "13",
                    "CZ_FIPS": county[-3:],
                    "CZ_NAME": county,
                    "WFO": "FFC",
                    "BEGIN_DATE_TIME": f"02-Jan-{str(year)[-2:]} 04:45:00",
                    "END_DATE_TIME": f"02-Jan-{str(year)[-2:]} 05:00:00",
                    "CZ_TIMEZONE": "CST-6",
                    "MAGNITUDE": "55",
                    "MAGNITUDE_TYPE": "EG",
                    "CATEGORY": None,
                    "FLOOD_CAUSE": None,
                    "BEGIN_LAT": "32",
                    "BEGIN_LON": "-83",
                    "END_LAT": "32",
                    "END_LON": "-83",
                }
            )
    pd.DataFrame(outages).to_csv(raw / "eaglei_outages_2014_2025.csv", index=False)
    pd.DataFrame(storms).to_csv(raw / "storm_events_details_all_years.csv", index=False)


class PipelineTests(unittest.TestCase):
    def test_raw_to_features_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_fixture(root)
            paths = ProjectPaths(root, offline=True)
            prepare(paths, memory_limit="1GB")
            with connection(paths) as con:
                attach_prepared(con, paths)
                table, features, retro = build_table(con)
            self.assertEqual(len(table), 6)
            self.assertEqual(len(features), 36)
            tied = (
                table.iloc[:2].rename(columns={v: k for k, v in retro.items()}).copy()
            )
            tied["fips"] = "13001"
            ordered, _, _ = assemble_table(tied)
            reversed_order, _, _ = assemble_table(tied.iloc[::-1])
            pd.testing.assert_frame_equal(ordered, reversed_order)
            self.assertEqual(ordered.episode_id.tolist(), sorted(tied.episode_id))
            # 10:15 precedes exact t0=10:45, but belongs to preserved start-hour label.
            self.assertEqual(
                table.loc[table.fips.eq("13001"), "target"].tolist(), [1, 1, 1]
            )
            self.assertEqual(
                table.loc[table.fips.eq("13003"), "target"].tolist(), [0, 0, 0]
            )
            self.assertTrue(table.t0.dt.minute.eq(45).all())
            self.assertTrue(
                table.loc[table.fips.eq("13003"), "n_zone_coded"].eq(1).all()
            )
            for invalid in [
                pd.concat([table, table.iloc[[0]]]),
                table.assign(target=1 - table.target),
                table.assign(year=2030),
            ]:
                with self.assertRaises(ValueError):
                    validate_table(invalid, features)
            self.assertFalse((root / "reports").exists())
            raw_audit = audit_raw(paths)
            self.assertEqual(
                raw_audit["training_outage_quality"].year.tolist(), ["2017"]
            )
            self.assertEqual(
                raw_audit["training_storm_quality"].year.tolist(), ["2017"]
            )
            self.assertEqual(
                set(raw_audit["source_year_counts"].year), {"2017", "2024", "2025"}
            )
            source = root / "data/raw/combined/eaglei_outages_2014_2025.csv"
            with connection(paths, cached=True) as con:
                load_raw(con, paths)
                first = con.execute("SELECT count(*) FROM eaglei_raw").fetchone()[0]
                with source.open("a") as stream:
                    stream.write("2017,13001,Georgia,0,2017-01-01 00:00:00\n")
                load_raw(con, paths)
                self.assertEqual(
                    con.execute("SELECT count(*) FROM eaglei_raw").fetchone()[0],
                    first + 1,
                )

    def test_connecticut_allocation_and_snapshot_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = pd.DataFrame(
                {
                    "SOURCE_YEAR": ["2025", "2025"],
                    "fips_code": ["09110"] * 2,
                    "state": ["Connecticut"] * 2,
                    "customers_out": ["10", "8"],
                    "run_start_time": ["2025-01-01 10:00:00"] * 2,
                }
            )
            weights = pd.DataFrame(
                {
                    "region_fips": ["09110"] * 2,
                    "county_fips": ["09001", "09003"],
                    "weight": [0.6, 0.4],
                }
            )
            counties = pd.DataFrame(
                {
                    "fips": ["09001", "09003"],
                    "denominator": [100, 100],
                    "denominator_source": ["mcc"] * 2,
                }
            )
            with connection(ProjectPaths(root, offline=True)) as con:
                con.register("eaglei_raw", raw)
                con.register("ct_weights", weights)
                con.register("counties", counties)
                clean_outages(con, root)
                rows = con.execute(
                    "SELECT fips,customers_out FROM outages_15min ORDER BY fips"
                ).fetchall()
            self.assertEqual(rows, [("09001", 6), ("09003", 4)])

    def test_history_preserves_full_training_prior_dependency(self):
        original = pd.DataFrame(
            {
                "fips": ["13001"] * 2 + ["13003"] * 2,
                "year": [2017, 2018, 2017, 2018],
                "split": ["train"] * 4,
                "target": [0, 1, 0, 0],
                "peak_24h": [0.0, 0.5, 0.0, 0.0],
            }
        )
        before = add_history(original)
        after = add_history(original.assign(target=[1, 1, 0, 0]))
        self.assertEqual(before.loc[0, "hist_episodes"], 1)
        self.assertAlmostEqual(before.loc[0, "hist_outage_rate"], 6 / 21)
        self.assertAlmostEqual(after.loc[0, "hist_outage_rate"], 11 / 21)
        self.assertNotIn("hist_outage_rate", original)

    def test_resolver_precedence_remote_pin_and_holdout_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = ProjectPaths(root)
            legacy = root / "processed/model_table/train.parquet"
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b"old")
            self.assertEqual(
                paths.resolve("processed/model_table/train.parquet").path,
                legacy.as_posix(),
            )
            new = root / "data/processed/train.parquet"
            new.parent.mkdir(parents=True)
            new.write_bytes(b"new")
            source = paths.resolve("processed/model_table/train.parquet")
            self.assertEqual(source.path, new.as_posix())
            self.assertFalse(
                data_version([source])["sources"][0]["hf_revision_verified"]
            )
            remote = paths.resolve("processed/model_table/validation.parquet")
            self.assertIn("@" + paths.revision + "/", remote.path)
            with self.assertRaises(FileNotFoundError):
                ProjectPaths(root, offline=True).resolve("missing.parquet")
            for name in [
                "holdout/test_labels.parquet",
                "processed/model_table/test_labels.parquet",
                "../outside",
            ]:
                with self.assertRaises(ValueError):
                    paths.resolve(name)

    def test_publication_failure_and_recovery_preserve_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "output"
            target.mkdir()
            (target / "value").write_text("old")
            with self.assertRaises(ValueError):
                with staged_directory(target) as stage:
                    (stage / "value").write_text("invalid")
                    raise ValueError("failed validation")
            self.assertEqual((target / "value").read_text(), "old")
            original_rename = Path.rename

            def fail_stage(path, destination):
                if ".staging-" in path.name:
                    raise OSError("publication failed")
                return original_rename(path, destination)

            with patch.object(Path, "rename", fail_stage):
                with self.assertRaises(OSError):
                    with staged_directory(target) as stage:
                        (stage / "value").write_text("new")
            self.assertEqual((target / "value").read_text(), "old")
            with staged_directory(target) as stage:
                (stage / "value").write_text("new")
            self.assertEqual((target / "value").read_text(), "new")
            self.assertEqual(
                (target.with_name("output.previous") / "value").read_text(), "old"
            )

    def test_evaluation_bootstrap_alignment_and_unsupported_family(self):
        train, val = synthetic_frames()
        p = np.full(len(val), 0.5)
        with patch("outage_prediction.evaluation.bootstrap_gain") as boot:
            evaluate(val, p, train, p_ref=p)
            boot.assert_not_called()
        with patch(
            "outage_prediction.evaluation.bootstrap_gain", return_value=(0.1, 0.2)
        ) as boot:
            metrics, _ = evaluate(val, p, train, p_ref=p, bootstrap=True, n_boot=20)
            boot.assert_called_once()
            self.assertEqual(metrics["pr_auc_gain_ci_low"], 0.1)
        for scores in [
            p[:-1],
            np.full(len(val), np.nan),
            np.full(len(val), 1.1),
            pd.Series(p, index=val.index[::-1]),
        ]:
            with self.assertRaises(ValueError):
                evaluate(val, scores, train)
        missing = val.copy()
        missing.loc[:, [c for c in val if c.startswith("n_")]] = 0
        with self.assertRaisesRegex(ValueError, "no supported"):
            family_rule(train, missing)

    def test_empty_distributions_are_unavailable(self):
        self.assertEqual(psi(pd.Series([1, 2]), pd.Series([1, 2])), 0)
        self.assertTrue(np.isnan(psi(pd.Series([1, 2]), pd.Series([np.nan]))))
        frames = [
            pd.DataFrame({"month": [1.0, 2.0]}),
            pd.DataFrame({"month": [1.0, 2.0]}),
            pd.DataFrame({"month": [np.nan, np.nan]}),
        ]
        row = feature_drift(*frames, ["month"]).iloc[0]
        self.assertEqual(row["missing_test"], 2)
        self.assertEqual(row["status_test"], "no observed distribution")


if __name__ == "__main__":
    unittest.main()
