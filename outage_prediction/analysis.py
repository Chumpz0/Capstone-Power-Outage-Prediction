"""Reusable audit and exploratory tables; no automatic exports or test-label access."""

from functools import partial
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import (
    ProjectPaths,
    TRAIN_FIRST,
    TRAIN_LAST,
    WINDOW_HOURS,
    FAMILIES,
    require_checks,
)
from .dataset import connection, attach_prepared, query, load_raw
from .train import load_model_data

SIG_PCT = 0.01  # Descriptive event-level EDA; the modeling target remains 5%.

FEATURE_DICTIONARY = {
    # id / split
    "fips": ("id", "text", "5-digit county FIPS code (Census 2023)"),
    "episode_id": ("id", "integer", "NOAA Storm Events EPISODE_ID (one storm system)"),
    "state_abbr": ("id", "text", "State postal code (lower 48 + DC)"),
    "fema_region": ("id", "1-10", "FEMA region of the county"),
    "year": (
        "id",
        "year",
        "UTC year of t0; sets the split (train 2017-2023, validation 2024, test 2025)",
    ),
    "t0": ("id", "UTC time", "Begin time of the episode's first report in this county"),
    # storm
    "n_events": (
        "storm",
        "count",
        "Storm reports in this county during the episode (drought/heat excluded)",
    ),
    "n_tstm_wind": ("storm", "count", "Reports of type Thunderstorm Wind"),
    "n_hail": ("storm", "count", "Reports of type Hail"),
    "n_tornado": ("storm", "count", "Reports of type Tornado or Funnel Cloud"),
    "n_high_wind": ("storm", "count", "Reports of type High Wind or Strong Wind"),
    "n_flood": (
        "storm",
        "count",
        "Flash Flood, Flood, Heavy Rain, Coastal/Lakeshore Flood, Storm Surge/Tide, Debris Flow",
    ),
    "n_winter": (
        "storm",
        "count",
        "Winter Weather, Winter Storm, Heavy Snow, Blizzard, Lake-Effect Snow, Sleet",
    ),
    "n_ice_storm": ("storm", "count", "Reports of type Ice Storm"),
    "n_tropical": (
        "storm",
        "count",
        "Tropical Storm, Hurricane (Typhoon), Tropical Depression",
    ),
    "n_cold": (
        "storm",
        "count",
        "Extreme Cold/Wind Chill, Cold/Wind Chill, Frost/Freeze",
    ),
    "n_lightning": ("storm", "count", "Reports of type Lightning"),
    "n_wildfire": ("storm", "count", "Reports of type Wildfire"),
    "n_other": (
        "storm",
        "count",
        "Any other event type (e.g. Dense Fog, Rip Current, Dust Storm)",
    ),
    "n_county_coded": (
        "storm",
        "count",
        "Reports NOAA located by county (CZ_TYPE = C)",
    ),
    "n_zone_coded": (
        "storm",
        "count",
        "Reports issued for an NWS forecast zone, mapped to counties with the dated crosswalk",
    ),
    "share_zone_coded": ("storm", "0-1", "n_zone_coded / n_events"),
    "any_partial_county": (
        "storm",
        "boolean",
        "At least one zone report whose zone covers only part of the county",
    ),
    "max_wind_kts": (
        "storm",
        "knots",
        "Highest wind magnitude among the reports (measured or estimated); empty if none reported",
    ),
    "max_hail_in": (
        "storm",
        "inches",
        "Largest hail size among Hail reports; empty if no hail",
    ),
    "max_hurricane_cat": (
        "storm",
        "0-5",
        "Highest Saffir-Simpson category reported; almost always empty",
    ),
    "county_episode_hours": (
        "storm",
        "hours",
        "First report begin to last report end in this county",
    ),
    "episode_counties": (
        "storm",
        "count",
        "Counties touched by the whole episode (storm-system size)",
    ),
    "log_episode_counties": ("storm", "log10", "log10(episode_counties)"),
    "episode_states": ("storm", "count", "States touched by the whole episode"),
    "episode_events": (
        "storm",
        "count",
        "Distinct reports in the whole episode, all counties",
    ),
    # time
    "month": ("time", "1-12", "Month of t0 (UTC)"),
    "local_hour": (
        "time",
        "0-23",
        "Hour of t0 in local standard time (state's main time zone)",
    ),
    # county
    "log_customers": (
        "county",
        "log10",
        "log10 of customers served: EAGLE-I MCC, or Census population x 0.508 where MCC is broken (counties failing the documented denominator checks)",
    ),
    "log_pop": (
        "county",
        "log10",
        "log10 of Census 2023 county population; empty for Connecticut's 8 old counties",
    ),
    "dqi": (
        "county",
        "0-100",
        "EAGLE-I data quality index of the FEMA region (2018-2022 values; 2017 uses 2018, 2023+ use 2022)",
    ),
    "pct_customers_tracked": (
        "county",
        "0-1",
        "Share of the state's customers EAGLE-I tracks (same year mapping as dqi)",
    ),
    # pre-storm
    "pre_peak_24h": (
        "pre-storm",
        "0-1",
        "Worst share of customers out in the 24 h before the start hour; empty if the day before wasn't observed",
    ),
    "pct_out_at_start": (
        "pre-storm",
        "0-1",
        "Share of customers out in the hour before the start hour; empty if the day before wasn't observed",
    ),
    # history
    "hist_episodes": (
        "history",
        "count",
        "County-episodes for this county in the training years (training rows: other training years only)",
    ),
    "hist_outage_rate": (
        "history",
        "0-1",
        "County's other-training-year outage rate (full-training prior), shrunk toward the national training rate: (pos + 20 x p0) / (n + 20)",
    ),
    "hist_mean_peak": (
        "history",
        "0-1",
        "County's other-training-year mean peak (full-training prior) share out, shrunk the same way",
    ),
    # label + analysis-only
    "target": (
        "target",
        "0/1",
        "1 if peak share out in hours 0-23 from the start hour is >= 5% of customers",
    ),
    "retro_peak_24h": (
        "retro (leakage)",
        "0-1",
        "Peak share of customers out in the 24 h window (defines the target)",
    ),
    "retro_peak_customers_out": (
        "retro (leakage)",
        "customers",
        "Peak customers out in the 24 h window",
    ),
    "retro_hour_of_peak": (
        "retro (leakage)",
        "0-23",
        "Hour of the peak after the start hour (earliest if tied)",
    ),
}


def as_num(series):
    """Convert numeric/nullable/bool features to floats without filling missing observations."""
    return pd.to_numeric(
        series.astype("float64") if series.dtype == bool else series, errors="coerce"
    ).astype(float)


def load_analysis_data(paths=None):
    """Load train/validation outcomes and test features; reject outcomes in test features."""
    paths = paths or ProjectPaths()
    train, validation, _ = load_model_data(paths)
    test = pd.read_parquet(
        paths.resolve("processed/model_table/test_features.parquet").path
    )
    if any(c == "target" or c.startswith("retro_") for c in test):
        raise ValueError("Test feature file must not contain outcomes")
    return train, validation, test


def audit_sources(paths=None):
    """Inspect source schemas and training-data quality, without opening raw/test outcomes.

    The inventory includes local file/directory presence for all source years;
    outcome nulls, ranges, duplicates and family checks use training rows only.
    """
    paths = paths or ProjectPaths()
    inventory = []
    for logical in [
        "data/02_combined/eaglei_outages_2014_2025.csv",
        "data/02_combined/storm_events_details_all_years.csv",
        "processed/outages_hourly",
        "processed/counties.parquet",
        "processed/model_table/train.parquet",
        "processed/model_table/validation.parquet",
        "processed/model_table/test_features.parquet",
    ]:
        candidates = paths.candidates(logical)
        local = next((p for p in candidates if p.exists()), None)
        inventory.append(
            {
                "source": logical,
                "local": local is not None,
                "location": str(local)
                if local
                else "pinned remote (not read for inventory)",
                "bytes": local.stat().st_size if local and local.is_file() else np.nan,
            }
        )
    train = pd.read_parquet(paths.resolve("processed/model_table/train.parquet").path)
    require_checks(
        {
            "training years": train.year.between(TRAIN_FIRST, TRAIN_LAST).all(),
            "county/episode keys": not train.duplicated(["fips", "episode_id"]).any(),
            "binary labels": train.target.notna().all()
            and train.target.isin([0, 1]).all(),
        },
        name="training audit",
    )
    schema = pd.DataFrame(
        {
            "column": train.columns,
            "dtype": [str(x) for x in train.dtypes],
            "missing_pct": 100 * train.isna().mean().to_numpy(),
        }
    )
    return {
        "source_inventory": pd.DataFrame(inventory),
        "training_schema": schema,
        "training_years": train.groupby("year").agg(
            rows=("target", "size"), positive_rate=("target", "mean")
        ),
    }


def audit_raw(paths=None):
    """Optionally audit large raw sources without analyzing held-out outcomes.

    This explicitly builds/reuses the content-identified raw cache. All-year
    inventory counts use source years only. Outage quality and storm-type
    summaries are restricted to training years before aggregation. Unlike the
    historical raw notebook, no 2025 storm/outage signal is computed.
    """
    paths = paths or ProjectPaths()
    with connection(paths, cached=True) as con:
        load_raw(con, paths)
        years = f"BETWEEN {TRAIN_FIRST} AND {TRAIN_LAST}"
        coverage = con.execute("""
            SELECT 'outages' AS source, SOURCE_YEAR AS year, count(*) AS rows
            FROM eaglei_raw GROUP BY SOURCE_YEAR
            UNION ALL
            SELECT 'storms' AS source, SOURCE_YEAR AS year, count(*) AS rows
            FROM storms_raw GROUP BY SOURCE_YEAR
            ORDER BY source, year
        """).df()
        outage_quality = con.execute(f"""
            SELECT SOURCE_YEAR AS year, count(*) AS rows,
                count(DISTINCT fips_code) AS counties,
                sum((customers_out IS NULL)::INT) AS missing_customers_out,
                sum((TRY_CAST(customers_out AS BIGINT) = 0)::INT) AS zero_rows,
                sum((TRY_CAST(customers_out AS BIGINT) < 0)::INT) AS negative_rows,
                sum((customers_out IS NOT NULL AND TRY_CAST(customers_out AS BIGINT) IS NULL)::INT) AS invalid_counts,
                sum((TRY_CAST(run_start_time AS TIMESTAMP) IS NULL)::INT) AS invalid_times,
                sum((NOT regexp_full_match(fips_code, '[0-9]{{5}}'))::INT) AS invalid_fips,
                count(*) - count(DISTINCT (fips_code, run_start_time)) AS duplicate_keys
            FROM eaglei_raw WHERE TRY_CAST(SOURCE_YEAR AS INT) {years}
            GROUP BY SOURCE_YEAR ORDER BY SOURCE_YEAR
        """).df()
        storm_quality = con.execute(f"""
            SELECT SOURCE_YEAR AS year, count(*) AS rows,
                count(*) - count(DISTINCT EVENT_ID) AS duplicate_event_ids,
                sum((TRY_STRPTIME(BEGIN_DATE_TIME, '%d-%b-%y %H:%M:%S') IS NULL)::INT) AS invalid_begin,
                sum((TRY_STRPTIME(END_DATE_TIME, '%d-%b-%y %H:%M:%S') IS NULL)::INT) AS invalid_end
            FROM storms_raw WHERE TRY_CAST(SOURCE_YEAR AS INT) {years}
            GROUP BY SOURCE_YEAR ORDER BY SOURCE_YEAR
        """).df()
        storm_types = con.execute(f"""
            SELECT EVENT_TYPE AS event_type, count(*) AS rows,
                avg((CZ_TYPE = 'C')::INT) AS county_coded_share
            FROM storms_raw WHERE TRY_CAST(SOURCE_YEAR AS INT) {years}
            GROUP BY EVENT_TYPE ORDER BY rows DESC
        """).df()
    return {
        "source_year_counts": coverage,
        "training_outage_quality": outage_quality,
        "training_storm_quality": storm_quality,
        "training_storm_types": storm_types,
    }


def event_county_days(con):
    """Return training-year significant outage days and state exposure-adjusted summaries."""
    q = partial(query, con)
    # A county-day counts when the worst hour that day had >= 1% of customers out.
    # Coverage eligibility comes from the prepared observation table.
    con.execute(f"""
    CREATE TEMP TABLE sig_days AS
    SELECT h.fips, CAST(h.hour_utc AS DATE) AS day, max(h.max_pct_out) AS peak_pct
    FROM hourly h
    WHERE h.year BETWEEN {TRAIN_FIRST} AND {TRAIN_LAST}
    GROUP BY 1, 2
    HAVING max(h.max_pct_out) >= {SIG_PCT}
    """)

    state_days = q(f"""
    SELECT o.state_abbr                                     AS state,
           count(s.fips)                                    AS sig_outage_days,
           sum(o.observed::INT)                             AS observed_county_days,
           365.0 * count(s.fips) / sum(o.observed::INT)     AS sig_days_per_county_year
    FROM observed o
    LEFT JOIN sig_days s ON s.fips = o.fips AND s.day = o.day
    WHERE o.year BETWEEN {TRAIN_FIRST} AND {TRAIN_LAST} AND o.observed
    GROUP BY 1 HAVING sum(o.observed::INT) > 0
    ORDER BY sig_outage_days DESC
    """)
    # For every storm-county row we take the worst hour in the 48 h after the storm starts.
    # Baseline = the same county and hours exactly one week earlier (a "normal" week).
    # The hourly table only has hours with an outage, so a missing hour means 0 out
    # (rows are limited to counties observed on the storm's start day).
    return state_days


def event_storm_peaks(con):
    """Return training event-level outage peaks and same-hours/prior-week comparison."""
    q = partial(query, con)
    con.execute(f"""
    CREATE TEMP TABLE sc AS
    SELECT row_number() OVER () AS rid, event_id, fips, event_type, magnitude, magnitude_type,
           mapped_from, partial_county, date_trunc('hour', begin_utc) AS start_hr, month(begin_utc) AS month
    FROM storm_county
    WHERE year BETWEEN {TRAIN_FIRST} AND {TRAIN_LAST}
      AND begin_utc < TIMESTAMP '{TRAIN_LAST}-12-30'     -- keep the 48 h window inside the training years
      AND fips_known AND county_observed_on_begin_day
    """)
    peaks = q(f"""
    WITH hours AS (
        SELECT rid, fips, start_hr + to_hours(h) AS hr, h
        FROM sc CROSS JOIN range(0, 48) t(h)
    ),
    after AS (
        SELECT x.rid,
               max(o.max_pct_out) FILTER (WHERE x.h < {WINDOW_HOURS}) AS peak_24h,
               max(o.max_pct_out)                                     AS peak_48h,
               arg_max(x.h, o.max_pct_out)                            AS hour_of_peak
        FROM hours x JOIN hourly o ON o.fips = x.fips AND o.hour_utc = x.hr
        WHERE o.year BETWEEN {TRAIN_FIRST} AND {TRAIN_LAST}
        GROUP BY x.rid
    ),
    before AS (
        SELECT x.rid, max(o.max_pct_out) AS baseline_24h
        FROM hours x JOIN hourly o ON o.fips = x.fips AND o.hour_utc = x.hr - INTERVAL 7 DAY
        WHERE x.h < {WINDOW_HOURS} AND o.year BETWEEN {TRAIN_FIRST} AND {TRAIN_LAST}
        GROUP BY x.rid
    )
    SELECT sc.*,
           coalesce(a.peak_24h, 0)     AS peak_24h,
           coalesce(a.peak_48h, 0)     AS peak_48h,
           a.hour_of_peak,
           coalesce(b.baseline_24h, 0) AS baseline_24h
    FROM sc LEFT JOIN after a USING (rid) LEFT JOIN before b USING (rid)
    """)
    peaks["outage"] = peaks["peak_24h"] >= SIG_PCT
    peaks["baseline_outage"] = peaks["baseline_24h"] >= SIG_PCT

    by_type = (
        peaks.groupby("event_type")
        .agg(
            rows=("rid", "count"),
            outages=("outage", "sum"),
            outage_rate=("outage", "mean"),
            baseline_rate=("baseline_outage", "mean"),
            median_peak=("peak_24h", "median"),
        )
        .sort_values("outages", ascending=False)
    )
    return peaks, by_type


def event_tables(paths=None):
    """Compute original event-level EDA using 2017-2023 outcomes only."""
    with connection(paths) as con:
        attach_prepared(con, paths or ProjectPaths())
        state_days = event_county_days(con)
        peaks, by_type = event_storm_peaks(con)
        q = partial(query, con)
        monthly = q(
            "SELECT month(day) AS month, count(*) / 7.0 AS sig_days_per_year FROM sig_days GROUP BY 1 ORDER BY 1"
        )
        wind = peaks[
            peaks["event_type"].isin(["Thunderstorm Wind", "High Wind"])
            & peaks["magnitude"].notna()
        ].copy()
        wind["speed_bin"] = pd.cut(
            wind.magnitude,
            bins=[0, 50, 55, 60, 65, 70, 80, 200],
            labels=["< 50", "50-54", "55-59", "60-64", "65-69", "70-79", "80+"],
            right=False,
        )
        wind_rate = wind.groupby("speed_bin", observed=True).agg(
            rows=("rid", "count"), outage_rate=("outage", "mean")
        )
        return {
            "state_days": state_days,
            "peaks": peaks,
            "by_type": by_type,
            "wind_rate": wind_rate,
            "monthly": monthly,
        }


def dataset_overview(train, validation, test):
    """Return split sizes and yearly prevalence; test prevalence stays missing."""
    frames = [("train", train), ("validation", validation), ("test", test)]
    overview = pd.DataFrame(
        [
            {
                "split": name,
                "years": f"{d.year.min()}-{d.year.max()}",
                "rows": len(d),
                "counties": d.fips.nunique(),
                "episodes": d.episode_id.nunique(),
                "states": d.state_abbr.nunique(),
                "positive_rate": d.target.mean() if "target" in d else np.nan,
            }
            for name, d in frames
        ]
    )
    by_year = (
        pd.concat([train, validation, test])[["year"]]
        .value_counts()
        .sort_index()
        .rename("rows")
        .reset_index()
    )
    by_year["positive_rate"] = by_year.year.map(
        pd.concat([train, validation]).groupby("year").target.mean()
    )
    return overview, by_year


def feature_dictionary(train, validation, test):
    """Return canonical field descriptions and split missingness, training statistics only."""
    if set(FEATURE_DICTIONARY) != set(train.columns):
        raise ValueError("Feature dictionary does not match the model-table schema")
    features = [
        c
        for c in test
        if c not in {"fips", "episode_id", "state_abbr", "fema_region", "year", "t0"}
    ]
    rows = []
    for col, (group, unit, desc) in FEATURE_DICTIONARY.items():
        s = train[col]
        row = {
            "column": col,
            "group": group,
            "unit": unit,
            "dtype": str(s.dtype),
            "description": desc,
            "model_input": col in features,
            "missing_pct_train": 100 * s.isna().mean(),
            "missing_pct_val": 100 * validation[col].isna().mean(),
            "missing_pct_test": 100 * test[col].isna().mean()
            if col in test
            else np.nan,
        }
        if col in features or col == "target" or col.startswith("retro_"):
            x = as_num(s)
            row.update(
                mean=x.mean(),
                std=x.std(),
                min=x.min(),
                p25=x.quantile(0.25),
                median=x.median(),
                p75=x.quantile(0.75),
                p99=x.quantile(0.99),
                max=x.max(),
                share_zero=(x == 0).mean(),
            )
        else:
            row["distinct"] = s.nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def feature_signal(train, features):
    """Score each observed training feature alone; direction is descriptive, not model performance."""
    rows = []
    for col in features:
        x = as_num(train[col])
        ok = x.notna()
        auc = (
            roc_auc_score(train.loc[ok, "target"], x[ok])
            if x[ok].nunique() > 1 and train.loc[ok, "target"].nunique() == 2
            else np.nan
        )
        rows.append(
            {
                "feature": col,
                "group": FEATURE_DICTIONARY[col][0],
                "auc": auc,
                "strength": abs(auc - 0.5) + 0.5,
                "direction": "higher -> more outages"
                if auc >= 0.5
                else "higher -> fewer outages",
                "coverage_pct": 100 * ok.mean(),
                "spearman": x.corr(train.target, method="spearman"),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("strength", ascending=False)
        .reset_index(drop=True)
    )


CURVES = [
    "hist_outage_rate",
    "hist_mean_peak",
    "pre_peak_24h",
    "pct_out_at_start",
    "n_events",
    "county_episode_hours",
    "n_county_coded",
    "share_zone_coded",
    "max_wind_kts",
    "max_hail_in",
    "episode_counties",
    "log_customers",
    "pct_customers_tracked",
    "dqi",
    "month",
    "local_hour",
]


def feature_bins(train):
    """Return original quantile/value-bin outage rates; ties remain together."""
    rows = []
    for col in CURVES:
        x, y = as_num(train[col]), train.target
        ok = x.notna()
        if not ok.any():
            continue
        if col in ("month", "local_hour"):
            bins = x[ok]
        elif col == "share_zone_coded":
            bins = pd.cut(x[ok], [-0.001, 0, 0.5, 0.999, 1])
        else:
            bins = (
                pd.qcut(x[ok], 10, duplicates="drop") if x[ok].nunique() > 10 else x[ok]
            )
        g = (
            pd.DataFrame({"x": x[ok], "y": y[ok], "b": bins})
            .groupby("b", observed=True)
            .agg(x=("x", "median"), rate=("y", "mean"), rows=("y", "size"))
        )
        rows.extend({"feature": col, **r} for r in g.to_dict("records"))
    return pd.DataFrame(rows)


def family_profiles(train):
    """Return training storm-family, mapping, and preceding-hour outage profiles."""
    fam = pd.DataFrame(
        [
            {
                "family": f,
                "rows": int((train[f"n_{f}"] > 0).sum()),
                "rate": train.loc[train[f"n_{f}"] > 0, "target"].mean(),
                "rate_only_family": train.loc[
                    (train[f"n_{f}"] > 0)
                    & (train[[f"n_{o}" for o in FAMILIES if o != f]].sum(axis=1) == 0),
                    "target",
                ].mean(),
            }
            for f in FAMILIES
        ]
    ).sort_values("rate")
    coding = np.select(
        [train.share_zone_coded.eq(0), train.share_zone_coded.eq(1)],
        ["county-coded only", "zone-coded only"],
        "mixed",
    )
    loc = train.groupby(coding).target.agg(rate="mean", rows="size")
    loc.loc["zone, partial county"] = [
        train.loc[train.any_partial_county, "target"].mean(),
        int(train.any_partial_county.sum()),
    ]
    pre = pd.cut(
        train.pct_out_at_start,
        [-0.001, 0.001, 0.01, 0.05, 1.0],
        labels=["< 0.1%", "0.1-1%", "1-5%", ">= 5%"],
    )
    pre_t = train.groupby(pre, observed=True).target.agg(rate="mean", rows="size")
    pre_t.loc["unknown (day before unobserved)"] = [
        train.loc[train.pct_out_at_start.isna(), "target"].mean(),
        int(train.pct_out_at_start.isna().sum()),
    ]
    return fam, loc, pre_t


def psi(ref, new, bins=10):
    """Compare observed distributions; return NaN when either has no observations.

    Missingness is reported alongside PSI, not interpreted as a measured shift.
    Training quantiles define continuous bins; small discrete domains use shares.
    """
    ref, new = ref.dropna(), new.dropna()
    if ref.empty or new.empty:
        return np.nan
    if ref.nunique() <= bins:
        cats = sorted(set(ref.unique()) | set(new.unique()))
        p = ref.value_counts(normalize=True).reindex(cats, fill_value=0)
        q = new.value_counts(normalize=True).reindex(cats, fill_value=0)
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        p = pd.Series(np.histogram(ref, edges)[0] / len(ref))
        q = pd.Series(np.histogram(new, edges)[0] / len(new))
    p, q = p.clip(lower=1e-4), q.clip(lower=1e-4)
    return float(((q - p) * np.log(q / p)).sum())


def feature_drift(train, validation, test, features):
    """Return split distribution shifts plus missing/observed counts and availability status."""
    rows = []
    for col in features:
        values = {
            name: as_num(frame[col])
            for name, frame in [
                ("train", train),
                ("validation", validation),
                ("test", test),
            ]
        }
        row = {"feature": col, "group": FEATURE_DICTIONARY[col][0]}
        for name, v in values.items():
            row.update(
                {
                    f"mean_{name}": v.mean(),
                    f"observed_{name}": int(v.notna().sum()),
                    f"missing_{name}": int(v.isna().sum()),
                }
            )
            if name != "train":
                value = psi(values["train"], v)
                row.update(
                    {
                        f"psi_{name}": value,
                        f"status_{name}": "available"
                        if np.isfinite(value)
                        else "no observed distribution",
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("psi_test", ascending=False)


def model_tables(train, validation, test):
    """Build ML-ready EDA tables; target relationships always use training rows only."""
    ids = {"fips", "episode_id", "state_abbr", "fema_region", "year", "t0"}
    features = [c for c in test if c not in ids]
    require_checks(
        {
            "no test outcomes": not any(
                c == "target" or c.startswith("retro_") for c in test
            ),
            "split order": train.year.max()
            < validation.year.min()
            <= validation.year.max()
            < test.year.min(),
            "unique keys": not pd.concat([train, validation, test])[
                ["fips", "episode_id"]
            ]
            .duplicated()
            .any(),
            "training target definition": train.target.eq(
                (train.retro_peak_24h >= 0.05).astype(int)
            ).all(),
        },
        name="EDA inputs",
    )
    overview, by_year = dataset_overview(train, validation, test)
    fam, loc, pre = family_profiles(train)
    numeric = pd.DataFrame({c: as_num(train[c]) for c in features})
    corr = numeric.corr(method="spearman")
    pairs = (
        corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1))
        .stack()
        .rename("spearman")
        .reset_index()
    )
    pairs.columns = ["feature_a", "feature_b", "spearman"]
    pairs = pairs.reindex(pairs.spearman.abs().sort_values(ascending=False).index)
    missing = pd.DataFrame(
        [
            {
                "feature": c,
                "missing_pct": 100 * train[c].isna().mean(),
                "rate_when_missing": train.loc[train[c].isna(), "target"].mean(),
                "rate_when_present": train.loc[train[c].notna(), "target"].mean(),
            }
            for c in features
            if train[c].isna().any()
        ]
    )
    if not missing.empty:
        missing = missing.sort_values("missing_pct", ascending=False)
    return {
        "dataset_overview": overview,
        "rows_by_year": by_year,
        "feature_dictionary": feature_dictionary(train, validation, test),
        "univariate_signal": feature_signal(train, features),
        "rate_by_feature_bins": feature_bins(train),
        "rate_by_storm_family": fam,
        "rate_by_storm_coding": loc,
        "rate_by_pre_storm_outage": pre,
        "missingness_vs_target": missing,
        "feature_correlation": corr,
        "top_correlated_pairs": pairs.head(40),
        "feature_drift_psi": feature_drift(train, validation, test, features),
    }
