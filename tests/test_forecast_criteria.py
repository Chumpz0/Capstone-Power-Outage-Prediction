import unittest
from types import SimpleNamespace
import pandas as pd
from src.forecast_criteria import split_origins
from scripts.check_geography_criteria import supported_candidate


class CriteriaTests(unittest.TestCase):
    def test_exclusive_boundaries_for_both_horizons(self):
        for h in (6, 12):
            for boundary, before, after in [("2024-01-01", "train", "validation"), ("2025-01-01", "validation", "test")]:
                b = pd.Timestamp(boundary)
                times = [b-pd.Timedelta(hours=h+1), b-pd.Timedelta(hours=h), b-pd.Timedelta(hours=1), b]
                self.assertEqual(split_origins(times,h).tolist(), [before,"excluded","excluded",after])

    def test_observation_start_and_final_cutoff(self):
        times = pd.to_datetime(["2014-11-01 03:00", "2014-11-01 04:00", "2025-12-31 18:00", "2026-01-01 00:00"])
        self.assertEqual(split_origins(times,6).tolist(), ["excluded","train","excluded","excluded"])
        with self.assertRaises(ValueError):
            split_origins(times,24)

    def test_mapping_needs_independent_name_and_unique_county_support(self):
        row = SimpleNamespace(state_abbr="GA",cz_fips="002",cz_name="WALKER")
        lookup = pd.DataFrame([dict(state="GA",zone="002",zone_name="Walker",county="Walker",fips_code="13295")])
        names = {"13295":"WALKER"}
        self.assertEqual(supported_candidate(row,lookup,names),"13295")
        self.assertIsNone(supported_candidate(row,pd.concat([lookup,lookup]),names))
        self.assertIsNone(supported_candidate(row,lookup,{"13295":"OTHER"}))
        row.cz_name="SOMETHING ELSE"
        self.assertIsNone(supported_candidate(row,lookup,names))


if __name__ == "__main__":
    unittest.main()
