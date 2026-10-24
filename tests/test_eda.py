"""Small counterexamples for label missingness and overlapping storm windows."""
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from src.eda import build_storm_outage_tables, grouped_burden


class StormOutageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.con = duckdb.connect()
        times = pd.to_datetime([f"2024-01-02 {h:02}:00" for h in range(4)])
        outages = pd.DataFrame({"fips_code": ["01001"] * 4,
                                "run_start_time": times, "customers_out": [10., 0., 20., 40.]})
        outages = pd.concat([outages, pd.DataFrame({
            "fips_code": ["01003"], "run_start_time": [times[2]],
            "customers_out": [float("nan")],
        })], ignore_index=True)
        census = pd.DataFrame({"fips_code": ["01001", "01003"], "acs_year": [2024] * 2,
                               "housing_units": [100, 100], "population": [200, 200],
                               "county_name": ["Autauga County, Alabama", "Baldwin County, Alabama"]})
        events = pd.DataFrame({
            "event_id": [1, 2, 3, 4, 5], "episode_id": [1, 1, 1, 2, 3],
            "fips_code": ["01001", "01001", "01003", "01001", "01001"],
            "event_type": ["High Wind"] * 5, "year": [2024] * 5, "state_abbr": ["AL"] * 5,
            "begin_utc": [times[1], times[2], times[2], times[3], times[1]],
            "end_utc": [times[2], times[3], times[2], times[3] + pd.Timedelta(hours=1), times[1]],
            "begin_local": [times[1]] * 5, "magnitude": [50.] * 5,
            "magnitude_type": ["MG"] * 5, "tor_f_scale": [None] * 5,
            "damage_property_usd": [0.] * 5, "mapped_from": ["county"] * 5})
        for name, frame in [("outages", outages), ("events", events), ("census", census)]:
            frame.to_parquet(self.root / f"{name}.parquet", index=False)
        build_storm_outage_tables(self.con, *(str(self.root / f"{n}.parquet")
                                             for n in ["outages", "events", "census"]),
                                  pre_h=0, post_h=0, customer_factor=1)
        self.impact = self.con.sql("select * from imp order by event_id").df().set_index("event_id")

    def tearDown(self):
        self.con.close()
        self.temp.cleanup()

    def test_unknown_is_not_zero(self):
        unknown = self.impact.loc[3]
        self.assertTrue(pd.isna(unknown.peak))
        self.assertTrue(pd.isna(unknown.major))
        self.assertTrue(pd.isna(unknown.cust_hours))
        self.assertEqual(unknown.observed_hours, 0)
        zero = self.impact.loc[5]
        self.assertEqual(zero.peak, 0)
        self.assertFalse(zero.major)

    def test_partial_window_and_pre_mean(self):
        partial = self.impact.loc[4]
        self.assertEqual(partial.expected_hours, 2)
        self.assertEqual(partial.observed_hours, 1)
        self.assertTrue(partial.boundary_window)
        self.assertEqual(self.impact.loc[1].pre_level, 10)
        self.assertEqual(self.impact.loc[1].pre_observed_hours, 1)
        self.assertTrue(pd.isna(self.impact.loc[3].pre_level))

    def test_overlapping_windows_count_each_hour_once(self):
        burden = grouped_burden(self.con, ["episode_id"]).set_index("episode_id")
        self.assertEqual(burden.loc[1, "cust_hours"], 60)
        self.assertEqual(burden.loc[1, "observed_hours"], 3)
        self.assertEqual(burden.loc[1, "expected_hours"], 4)
        county = grouped_burden(self.con, ["fips_code"]).set_index("fips_code")
        self.assertTrue(pd.isna(county.loc["01003", "cust_hours"]))


if __name__ == "__main__":
    unittest.main()
