"""Boundary, missingness and anomaly counterexamples for checkpoint 1."""
import unittest

import numpy as np
import pandas as pd

from src.observation_quality import county_windows, summarize_windows


class ObservationQualityTests(unittest.TestCase):
    def frame(self, start="2023-01-01"):
        # Includes the origin and both horizon endpoints.
        return pd.DataFrame({"run_start_time": pd.date_range(start, periods=49, freq="15min"),
                             "customers_out": np.zeros(49)})

    def windows(self, frame, start="2023-01-01", **kwargs):
        return county_windows(frame, start=start, hours=2, envelope_start=start,
                              envelope_end=kwargs.pop("envelope_end", "2024-01-01"),
                              housing_units=100, **kwargs)

    def test_open_origin_closed_end_and_zero_is_valid(self):
        frame = self.frame()
        frame.loc[0, "customers_out"] = np.nan  # excluded origin
        windows = self.windows(frame)
        self.assertEqual(windows[6]["usable"][0], 24)
        self.assertEqual(windows[12]["usable"][0], 48)
        self.assertEqual(windows[6]["zero"][0], 24)
        frame.loc[24, "customers_out"] = np.nan  # included 06:00 endpoint
        self.assertEqual(self.windows(frame)[6]["usable"][0], 23)

    def test_duplicates_invalid_and_proxy_flags(self):
        frame = self.frame()
        frame.loc[1, "customers_out"] = -1
        frame.loc[2, "customers_out"] = .5
        frame.loc[3, "customers_out"] = np.inf
        frame.loc[4, "customers_out"] = 121  # usable, but above proxy
        frame = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
        w = self.windows(frame)[6]
        self.assertEqual(w["usable"][0], 20)
        self.assertEqual(w["invalid"][0], 3)
        self.assertEqual(w["duplicate"][0], 1)
        self.assertEqual(w["proxy_flag"][0], 1)
        self.assertTrue(w["flagged"][0])

    def test_off_grid_cannot_fill_a_missing_slot(self):
        frame = self.frame().drop(index=1)
        extra = pd.DataFrame({"run_start_time": pd.to_datetime(["2023-01-01 00:16:00", "2023-01-01 06:00:01"]),
                              "customers_out": [0., 0.]})
        w = self.windows(pd.concat([frame, extra], ignore_index=True))[6]
        self.assertEqual(w["usable"][0], 23)
        self.assertEqual(w["off_grid"][0], 1)  # second row falls outside first window

    def test_unknown_null_and_partial_are_distinct(self):
        frame = self.frame()
        frame["customers_out"] = np.nan
        w = self.windows(frame)[6]
        self.assertEqual(w["usable"][0], 0)
        self.assertEqual(w["null"][0], 24)
        self.assertEqual(summarize_windows(w, 6)["unknown"], 2)
        empty = self.windows(frame.iloc[:0])[6]
        self.assertEqual(empty["null"][0], 0)
        frame.loc[1, "customers_out"] = 0
        summary = summarize_windows(self.windows(frame)[6], 6)
        self.assertEqual(summary["partial"], 1)
        self.assertEqual(summary["unknown"], 1)

    def test_year_buffer_and_training_cutoff(self):
        start = "2022-12-31 23:00"
        w = self.windows(self.frame(start), start=start)[6]
        self.assertEqual(w["usable"][0], 24)
        self.assertTrue(w["in_bounds"][0])
        cutoff = self.windows(self.frame(start), start=start, envelope_end="2023-01-01")[6]
        self.assertFalse(cutoff["in_bounds"].any())
        self.assertEqual(summarize_windows(cutoff, 6)["boundary_excluded"], 2)

    def test_vectorized_counts_match_direct_timestamp_filtering(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({
            "run_start_time": pd.date_range("2023-01-01", periods=160, freq="15min"),
            "customers_out": rng.choice([0., 12., 200., np.nan, -1., .5], 160),
        })
        frame = frame.loc[rng.random(len(frame)) > .2]
        frame = pd.concat([frame, frame.iloc[:3], pd.DataFrame({
            "run_start_time": pd.to_datetime(["2023-01-01 01:16", "2023-01-01 08:01"]),
            "customers_out": [0., 10.],
        })], ignore_index=True)
        actual = county_windows(frame, start="2023-01-01", hours=24,
                                envelope_start="2023-01-01", envelope_end="2024-01-01",
                                housing_units=100)
        for horizon in (6, 12):
            for hour in range(24):
                origin = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=hour)
                subset = frame[frame.run_start_time.gt(origin) & frame.run_start_time.le(
                    origin + pd.Timedelta(hours=horizon))]
                aligned = subset.run_start_time.eq(subset.run_start_time.dt.floor("15min"))
                exact = subset[aligned]
                unique = exact[~exact.run_start_time.duplicated(keep=False)]
                valid_values = [v for v in unique.customers_out
                                if pd.notna(v) and np.isfinite(v) and v >= 0 and float(v).is_integer()]
                self.assertEqual(actual[horizon]["usable"][hour], len(valid_values))
                self.assertEqual(actual[horizon]["off_grid"][hour], int((~aligned).sum()))


if __name__ == "__main__":
    unittest.main()
