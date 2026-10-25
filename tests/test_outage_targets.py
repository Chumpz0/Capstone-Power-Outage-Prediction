"""Target boundary, missingness, aggregation and exclusion counterexamples."""
import unittest
import numpy as np
import pandas as pd
from src.outage_targets import THRESHOLDS, county_targets, labels, summarize_targets


class TargetTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'run_start_time': pd.date_range('2023-01-01', periods=61, freq='15min'),
                             'customers_out': np.zeros(61)})

    def targets(self, frame, **kwargs):
        return county_targets(frame, start='2023-01-01', hours=3, envelope_start='2023-01-01',
                              envelope_end=kwargs.pop('envelope_end', '2024-01-01'),
                              housing_units=kwargs.pop('housing_units', 100000), **kwargs)

    def test_strict_thresholds_and_endpoints(self):
        for threshold in THRESHOLDS:
            frame = self.frame()
            frame.loc[0, 'customers_out'] = threshold + 10000  # origin excluded
            frame.loc[24, 'customers_out'] = threshold
            frame.loc[48, 'customers_out'] = threshold + 1
            w = self.targets(frame)
            self.assertEqual(labels(w[6], 6, threshold)[0], 0)
            self.assertEqual(labels(w[12], 12, threshold)[0], 1)
            self.assertEqual(w[6]['recorded_peak'][0], threshold)

    def test_missing_and_partial_exceedance(self):
        frame = self.frame()
        frame.customers_out = np.nan
        w = self.targets(frame)[6]
        self.assertTrue(np.isnan(labels(w, 6, 5000)[0]))
        frame.loc[1, 'customers_out'] = 0
        self.assertTrue(np.isnan(labels(self.targets(frame)[6], 6, 5000)[0]))
        frame.loc[1, 'customers_out'] = 5001
        w = self.targets(frame)[6]
        self.assertEqual(labels(w, 6, 5000)[0], 1)
        self.assertEqual(summarize_targets(w, 6, 5000)['partial_positive_unflagged'], 1)

    def test_hourly_median_suppresses_isolated_peak(self):
        frame = self.frame()
        frame.loc[4, 'customers_out'] = 6000  # 01:00 belongs to first future hour
        w = self.targets(frame)[6]
        self.assertEqual(w['recorded_peak'][0], 6000)
        self.assertEqual(w['median_peak'][0], 0)
        self.assertEqual(w['recorded_peak'][1], 0)
        frame.loc[1:4, 'customers_out'] = [4000, 6000, 6000, 6000]
        self.assertEqual(self.targets(frame)[6]['median_peak'][0], 6000)

    def test_proxy_removal_cannot_create_reliable_negative(self):
        frame = self.frame()
        frame.loc[1, 'customers_out'] = 7000
        w = self.targets(frame, housing_units=5000)[6]
        self.assertEqual(w['recorded_peak'][0], 7000)  # uncapped
        self.assertEqual(w['excluded_peak'][0], 0)
        self.assertTrue(np.isnan(labels(w, 6, 5000, exclude_proxy=True)[0]))
        self.assertEqual(summarize_targets(w, 6, 5000)['proxy_uncertain_complete'], 1)

    def test_duplicates_offgrid_invalid_and_cutoff(self):
        frame = self.frame()
        frame.loc[1, 'customers_out'] = 9000
        frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
        frame.loc[2, 'customers_out'] = -3
        extra = pd.DataFrame({'run_start_time': [pd.Timestamp('2023-01-01 00:16')], 'customers_out': [8000]})
        frame = pd.concat([frame, extra], ignore_index=True)
        w = self.targets(frame, envelope_end='2023-01-01 12:00')
        self.assertEqual(w[6]['recorded_peak'][0], 0)
        self.assertTrue(w[6]['flagged'][0])
        self.assertTrue(np.isnan(w[12]['recorded_peak']).all())

    def test_overlapping_windows_against_timestamp_reference(self):
        rng = np.random.default_rng(19)
        frame = self.frame()
        frame.customers_out = rng.integers(0, 15000, len(frame)).astype(float)
        frame.loc[[3, 8, 16], 'customers_out'] = np.nan
        for h, w in self.targets(frame).items():
            for i in range(3):
                t = pd.Timestamp('2023-01-01') + pd.Timedelta(hours=i)
                selected = frame[frame.run_start_time.gt(t) & frame.run_start_time.le(t + pd.Timedelta(hours=h))]
                self.assertEqual(w['recorded_peak'][i], selected.customers_out.max())
                medians = []
                for j in range(h):
                    a = t + pd.Timedelta(hours=j)
                    medians.append(selected.loc[selected.run_start_time.gt(a) & selected.run_start_time.le(a + pd.Timedelta(hours=1)), 'customers_out'].median())
                self.assertEqual(w['median_peak'][i], max(medians))

    def test_short_chunk_keeps_current_year_housing(self):
        frame = self.frame()
        frame.customers_out = 7000.
        windows = self.targets(frame, housing_units=10000, next_housing_units=5000)
        for horizon, window in windows.items():
            self.assertEqual(window['excluded_usable'][0], horizon * 4)
            self.assertFalse(window['flagged'][0])

    def test_chunk_crossing_midnight_switches_at_new_year(self):
        frame = pd.DataFrame({
            'run_start_time': pd.date_range('2022-12-31 18:00', periods=49, freq='15min'),
            'customers_out': np.full(49, 7000.)})
        windows = county_targets(
            frame, start='2022-12-31 18:00', hours=2,
            envelope_start='2022-01-01', envelope_end='2024-01-01',
            housing_units=10000, next_housing_units=5000)
        self.assertEqual(windows[6]['excluded_usable'][0], 23)
        self.assertEqual(windows[6]['proxy_flag'][0], 1)
        self.assertEqual(windows[6]['excluded_usable'][1], 19)

    def test_year_buffer_uses_next_year_proxy_without_crossing_cutoff(self):
        frame = pd.DataFrame({'run_start_time': pd.date_range('2022-12-31 18:00', periods=49, freq='15min'),
                              'customers_out': np.full(49, 7000.)})
        windows = county_targets(frame, start='2022-12-31 18:00', hours=6,
                                 envelope_start='2022-01-01', envelope_end='2024-01-01',
                                 housing_units=10000, next_housing_units=5000)
        # The midnight endpoint uses next-year housing and is excluded only in sensitivity.
        self.assertEqual(windows[6]['recorded_peak'][0], 7000)
        self.assertEqual(windows[6]['excluded_usable'][0], 23)
        self.assertEqual(windows[12]['excluded_usable'][0], 23)
        self.assertEqual(labels(windows[12], 12, 5000, exclude_proxy=True)[0], 1)


if __name__ == '__main__':
    unittest.main()
