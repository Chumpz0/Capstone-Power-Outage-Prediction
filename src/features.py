# One definition of the modeling-table columns, shared by every model.
# FEATURES mirrors notebook 04 (the 36 columns written to test_features.parquet); check_columns()
# fails loudly if the two ever drift apart.

ID_COLS = ["fips", "episode_id", "state_abbr", "fema_region", "year", "t0"]
TARGET = "target"

FEATURES = [
    # storm (whole-episode summary)
    "n_events", "n_tstm_wind", "n_hail", "n_tornado", "n_high_wind", "n_flood", "n_winter", "n_ice_storm",
    "n_tropical", "n_cold", "n_lightning", "n_wildfire", "n_other",
    "n_county_coded", "n_zone_coded", "share_zone_coded", "any_partial_county",
    "max_wind_kts", "max_hail_in", "max_hurricane_cat", "county_episode_hours",
    "episode_counties", "log_episode_counties", "episode_states", "episode_events",
    # time
    "month", "local_hour",
    # county
    "log_customers", "log_pop", "dqi", "pct_customers_tracked",
    # conditions before the storm
    "pre_peak_24h", "pct_out_at_start",
    # county history (training years only)
    "hist_episodes", "hist_outage_rate", "hist_mean_peak",
]

# Features left out of the models, with the reason (user decisions)
EXCLUDED = {
    "dqi": "year-mapped EAGLE-I quality score; PSI 3.7 between train and 2024/2025 (06 EDA). Decided 2026-10-25.",
}
MODEL_FEATURES = [f for f in FEATURES if f not in EXCLUDED]

# Feature groups for the logistic-regression preprocessing (see src/models.py)
COUNT_FEATURES = [  # skewed counts / durations -> log1p
    "n_events", "n_tstm_wind", "n_hail", "n_tornado", "n_high_wind", "n_flood", "n_winter", "n_ice_storm",
    "n_tropical", "n_cold", "n_lightning", "n_wildfire", "n_other", "n_county_coded", "n_zone_coded",
    "county_episode_hours", "episode_counties", "episode_states", "episode_events", "hist_episodes",
]
REPORT_MAGNITUDES = ["max_wind_kts", "max_hail_in", "max_hurricane_cat"]   # empty = no such report -> 0 + flag
UNOBSERVED_FEATURES = ["pre_peak_24h", "pct_out_at_start", "log_pop"]      # empty = unknown -> median + flag
CYCLIC_FEATURES = ["local_hour"]                                            # -> sin / cos
ONE_HOT_FEATURES = ["month"]


def plain_features(features=None):
    """Model features that need no special handling: scaled as they are."""
    features = MODEL_FEATURES if features is None else features
    special = set(COUNT_FEATURES + REPORT_MAGNITUDES + UNOBSERVED_FEATURES + CYCLIC_FEATURES + ONE_HOT_FEATURES)
    return [f for f in features if f not in special]


def check_columns(frame):
    """The frame holds every feature (and no retro_ column among them)."""
    missing = [f for f in FEATURES if f not in frame.columns]
    assert not missing, f"model table is missing features: {missing}"
    assert not any(f.startswith("retro_") for f in FEATURES)
