"""Prepare validated outage, observation and storm tables; reports are opt-in.

County monitoring is a coverage heuristic, not continuous-observation proof.
Reference fallbacks and Connecticut harmonization preserve the original method.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import json
from pathlib import Path

import duckdb
import pandas as pd

from .config import (
    ProjectPaths,
    TRAIN_FIRST as FIRST_YEAR,
    TEST_YEAR as LAST_YEAR,
    MIN_DAILY_SNAPSHOT_PCT,
    FEMA,
    staged_directory,
    require_checks,
    export_results,
    sha256,
)


def query(con, sql):
    """Run SQL on the caller-owned connection and return a DataFrame."""
    return con.execute(sql).df()


@contextmanager
def connection(paths=None, *, cached=False, memory_limit="16GB"):
    """Own a DuckDB connection; configure remote credentials only for explicit remote reads."""
    paths = paths or ProjectPaths()
    if cached:
        paths.cache.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(paths.cache / "raw.duckdb") if cached else ":memory:")
    try:
        con.execute("SET memory_limit = ?", [memory_limit])
        yield con
    finally:
        con.close()


def reader_path(con, source):
    """Enable credential-chain access only when a resolved source is remote."""
    if source.revision:
        con.execute(
            "CREATE OR REPLACE SECRET hf (TYPE huggingface, PROVIDER credential_chain)"
        )
    return source.path.replace("'", "''")


def attach_prepared(con, paths):
    """Register cleaned source views without loading labels or fitting models."""
    names = {
        "hourly": "outages_hourly",
        "observed": "county_observed_daily.parquet",
        "storm_county": "storm_county_events.parquet",
        "counties": "counties.parquet",
    }
    for name, filename in names.items():
        source = paths.resolve("processed/" + filename)
        path = reader_path(con, source)
        if name == "hourly":
            path += "/*/*.parquet"
        con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}', hive_partitioning=true)"
        )


def load_raw(con, paths):
    """Reuse raw caches only when content identities and transformation version match.

    A legacy or changed cache is rebuilt explicitly during preparation; notebook
    imports never invoke this operation. All raw columns stay text.
    """
    inputs = {}
    for name, csv_name, parquet_name in [
        ("eaglei_raw", "eaglei_outages_2014_2025.csv", "eaglei_outages/*/*.parquet"),
        (
            "storms_raw",
            "storm_events_details_all_years.csv",
            "storm_events_details_all_years.parquet",
        ),
    ]:
        logical = "data/02_combined/" + csv_name
        local = next((p for p in paths.candidates(logical) if p.is_file()), None)
        if local:
            inputs[name] = (
                str(local),
                {"path": str(local), "sha256": sha256(local)},
                True,
            )
        else:
            source = paths.resolve("data/02_combined/" + parquet_name.split("/*")[0])
            path = reader_path(con, source) + (
                "/*/*.parquet" if "/*" in parquet_name else ""
            )
            if source.revision:
                identity = source.metadata()
            else:
                files = (
                    sorted(Path(source.path).rglob("*.parquet"))
                    if Path(source.path).is_dir()
                    else [Path(source.path)]
                )
                identity = {"files": {str(p): sha256(p) for p in files}}
            inputs[name] = (path, identity, False)
    expected = {"version": 1, "sources": {k: v[1] for k, v in inputs.items()}}
    manifest = paths.cache / "raw_manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    existing = set(con.execute("SHOW TABLES").df()["name"])
    if previous != expected or not set(inputs) <= existing:
        con.execute("BEGIN TRANSACTION")
        try:
            for name, (path, _, csv) in inputs.items():
                path = path.replace("'", "''")
                reader = (
                    f"read_csv('{path}', all_varchar=true)"
                    if csv
                    else f"read_parquet('{path}', hive_partitioning=false)"
                )
                con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {reader}")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        paths.cache.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(expected, indent=2), encoding="utf-8")


def load_references(paths):
    """Load small geography, denominator and quality references; return named DataFrames."""
    src = lambda rel: paths.resolve(rel).path
    CENSUS_CSV = src("data/01_raw/census_counties/co-est2023-alldata.csv")
    CROSSWALK = src("data/reference/nws_zone_county_versions.parquet")
    CT_WEIGHTS = src("data/reference/ct_region_to_county_weights.csv")
    MCC_CSV = src("data/reference/eaglei_archive/MCC.csv")
    DQI_CSV = src("data/reference/eaglei_archive/DQI.csv")
    COVERAGE_CSV = src("data/reference/eaglei_archive/coverage_history.csv")
    # Census counties (SUMLEV 050 = county rows)
    census = pd.read_csv(
        CENSUS_CSV,
        dtype=str,
        encoding="latin-1",
        usecols=["SUMLEV", "STATE", "COUNTY", "STNAME", "CTYNAME", "POPESTIMATE2023"],
    )
    census = census[census["SUMLEV"] == "050"].copy()
    census["fips"] = census["STATE"] + census["COUNTY"]
    census["pop2023"] = census["POPESTIMATE2023"].astype(int)

    # MCC = EAGLE-I modeled customer count per county (our denominator)
    mcc = pd.read_csv(MCC_CSV, dtype=str, encoding="utf-8-sig")
    mcc["fips"] = mcc["County_FIPS"].str.zfill(5)
    mcc["mcc_customers"] = mcc["Customers"].astype(int)

    # Connecticut planning region -> old county population weights
    ct_weights = pd.read_csv(CT_WEIGHTS, dtype={"region_fips": str, "county_fips": str})

    # NWS zone -> county crosswalk; let the oldest version also cover dates before it
    crosswalk = pd.read_parquet(CROSSWALK)
    crosswalk = crosswalk[crosswalk["county_fips"].str.len() == 5]
    crosswalk["effective_date"] = pd.to_datetime(crosswalk["effective_date"])
    crosswalk["next_effective_date"] = pd.to_datetime(crosswalk["next_effective_date"])
    crosswalk.loc[
        crosswalk["effective_date"] == crosswalk["effective_date"].min(),
        "effective_date",
    ] = pd.Timestamp("2000-01-01")

    # State FIPS -> postal code (from the crosswalk) and FEMA region
    abbr_to_region = {a: r for r, states in FEMA.items() for a in states}
    state_abbr = (
        crosswalk[crosswalk["state"] != ""]
        .assign(state_fips=lambda d: d["county_fips"].str[:2])
        .groupby("state_fips")["state"]
        .agg(lambda s: s.mode()[0])
    )
    regions = pd.DataFrame(
        {"state_fips": state_abbr.index, "state_abbr": state_abbr.values}
    )
    regions["fema_region"] = regions["state_abbr"].map(abbr_to_region).astype("Int64")

    # Quality references: % of customers tracked (by state) and DQI (by FEMA region), 2018-2022
    coverage = pd.read_csv(COVERAGE_CSV)
    coverage["ref_year"] = pd.to_datetime(coverage["year"], format="%m/%d/%y").dt.year
    coverage = coverage.rename(
        columns={"state": "state_abbr", "max_pct_covered": "pct_customers_tracked"}
    )
    dqi = pd.read_csv(DQI_CSV).rename(
        columns={"fema": "fema_region", "year": "ref_year", "DQI": "dqi"}
    )
    return dict(
        census=census,
        mcc=mcc,
        ct_weights=ct_weights,
        crosswalk=crosswalk,
        regions=regions,
        coverage=coverage,
        dqi=dqi,
    )


def build_counties(census, mcc, ct_weights, regions):
    """Return county denominators, using MCC or the documented Census fallback."""
    # Denominator = MCC customers; if a county has no MCC value, estimate it from Census
    # population x the median customers-per-person ratio.
    counties = mcc[["fips", "mcc_customers"]].merge(
        census[["fips", "STNAME", "CTYNAME", "pop2023"]], on="fips", how="outer"
    )
    counties = counties[
        ~counties["fips"].isin(ct_weights["region_fips"])
    ]  # CT uses the old 8 counties
    counties = counties[
        counties["fips"].str.fullmatch(r"\d{5}")
    ]  # drop MCC's "Grand Total" row
    ratio = (counties["mcc_customers"] / counties["pop2023"]).median()
    counties["state_fips"] = counties["fips"].str[:2]

    # Some states' MCC totals are not believable (e.g. NC: 0.17 customers per person vs ~0.5 nationally).
    # If a state's MCC customers-per-person is below half the national median, use the Census estimate there.
    state_ratio = (
        counties.dropna(subset=["mcc_customers", "pop2023"])
        .groupby("state_fips")[["mcc_customers", "pop2023"]]
        .sum()
        .eval("mcc_customers / pop2023")
    )
    bad_mcc_states = state_ratio[state_ratio < 0.5 * ratio].index.tolist()
    use_mcc = counties["mcc_customers"].notna() & ~counties["state_fips"].isin(
        bad_mcc_states
    )

    # Single counties can be just as wrong (e.g. Buena Vista city VA: 1 customer for 6,566 people;
    # McDonald County MO: 178 K customers for 24 K people). Use the Census estimate when a county's
    # customers-per-person is below 0.1 or above 5 (provisional cut-offs, ~5x off the national median).
    county_ratio = counties["mcc_customers"] / counties["pop2023"]
    bad_mcc_counties = county_ratio.lt(0.1) | county_ratio.gt(5)
    use_mcc &= ~bad_mcc_counties

    counties["denominator"] = counties["mcc_customers"].where(
        use_mcc, (counties["pop2023"] * ratio).round()
    )
    counties["denominator_source"] = use_mcc.map(
        {True: "mcc", False: "census_estimate"}
    )
    counties = counties.merge(regions, on="state_fips", how="left")
    counties = counties.rename(
        columns={"STNAME": "state_name", "CTYNAME": "county_name"}
    )
    counties = counties[
        [
            "fips",
            "state_fips",
            "state_abbr",
            "fema_region",
            "county_name",
            "state_name",
            "pop2023",
            "mcc_customers",
            "denominator",
            "denominator_source",
        ]
    ]
    return counties


def clean_outages(con, output):
    """Write cleaned, deduplicated positive snapshots to a fresh staging directory."""
    OUT = Path(output)
    # A. EAGLE-I OUTAGES (15-minute)
    # Timestamps are snapped to the nearest 15 minutes (add 7.5 min, then round down).
    con.execute(f"""
    CREATE OR REPLACE VIEW eaglei_typed AS
    SELECT
        fips_code                                   AS fips,
        CASE WHEN state = 'US Virgin Islands'
             THEN 'United States Virgin Islands' ELSE state END AS state,
        TRY_CAST(customers_out AS BIGINT)           AS customers_out,
        CAST(run_start_time AS TIMESTAMP)           AS raw_ts,
        time_bucket(INTERVAL 15 MINUTE, CAST(run_start_time AS TIMESTAMP) + INTERVAL 450 SECOND) AS ts
    FROM eaglei_raw
    WHERE CAST(SOURCE_YEAR AS INT) BETWEEN {FIRST_YEAR} AND {LAST_YEAR}
    """)
    # Counted BEFORE nulls/zeros are dropped: any row (even 0 or null) shows EAGLE-I saw the county.
    con.execute("""
    CREATE OR REPLACE TEMP TABLE county_day_raw AS
    SELECT fips, CAST(ts AS DATE) AS day,
           count(*)              AS rows_any,
           count(customers_out)  AS rows_non_null,
           sum((customers_out IS NULL)::INT) AS rows_null
    FROM eaglei_typed GROUP BY ALL
    """)
    con.execute("""
    CREATE OR REPLACE TEMP TABLE national_day AS
    SELECT CAST(ts AS DATE) AS day, count(DISTINCT ts) / 96 AS snapshot_pct
    FROM eaglei_typed GROUP BY 1
    """)
    # 1. keep positive readings only (drops nulls and zeros)
    # 2. snapped duplicates of the same county + time -> keep the larger reading
    # 3. CT planning regions (2025) -> old counties by population weight, summed per county + time
    # 4. join the denominator, cap at it, flag capped rows, compute pct_out
    con.execute(f"""
    COPY (
        WITH positive AS (
            SELECT fips, ts, max(customers_out) AS customers_out, any_value(state) AS state
            FROM eaglei_typed WHERE customers_out > 0
            GROUP BY fips, ts
        ),
        ct_mapped AS (
            SELECT coalesce(w.county_fips, p.fips) AS fips, p.ts, p.state,
                   p.customers_out * coalesce(w.weight, 1) AS customers_out
            FROM positive p LEFT JOIN ct_weights w ON w.region_fips = p.fips
        ),
        combined AS (
            SELECT fips, ts, any_value(state) AS state, CAST(round(sum(customers_out)) AS BIGINT) AS customers_out
            FROM ct_mapped GROUP BY fips, ts
        )
        SELECT c.fips, c.state, c.ts,
               year(c.ts)                                            AS year,
               least(c.customers_out, d.denominator)::BIGINT          AS customers_out,
               coalesce(c.customers_out > d.denominator, FALSE)      AS over_mcc,
               c.customers_out                                       AS customers_out_raw,
               d.denominator                                         AS mcc_customers,
               d.denominator_source,
               least(c.customers_out, d.denominator) / d.denominator AS pct_out
        FROM combined c LEFT JOIN counties d USING (fips)
        WHERE c.customers_out > 0 AND year(c.ts) BETWEEN {FIRST_YEAR} AND {LAST_YEAR}
    ) TO '{(OUT / "outages_15min").as_posix()}' (FORMAT parquet, PARTITION_BY (year))
    """)
    con.execute(f"""
    CREATE OR REPLACE VIEW outages_15min AS
    SELECT * FROM read_parquet('{(OUT / "outages_15min").as_posix()}/*/*.parquet', hive_partitioning = true)
    """)


def build_hourly(con, output):
    """Aggregate positive 15-minute snapshots to hourly maxima in staging."""
    OUT = Path(output)
    # B. HOURLY OUTAGES -> processed/outages_hourly/
    con.execute(f"""
    COPY (
        SELECT fips, year,
               date_trunc('hour', ts)   AS hour_utc,
               max(customers_out)       AS max_customers_out,
               max(pct_out)             AS max_pct_out,
               any_value(mcc_customers) AS mcc_customers,
               bool_or(over_mcc)        AS over_mcc,
               count(*)                 AS snapshots_with_outage   -- 1 to 4
        FROM outages_15min
        GROUP BY fips, year, hour_utc
    ) TO '{(OUT / "outages_hourly").as_posix()}' (FORMAT parquet, PARTITION_BY (year))
    """)
    con.execute(f"""
    CREATE OR REPLACE VIEW outages_hourly AS
    SELECT * FROM read_parquet('{(OUT / "outages_hourly").as_posix()}/*/*.parquet', hive_partitioning = true)
    """)


def build_observation_flags(con, output):
    """Write daily coverage flags; missing rows are zero only under this heuristic."""
    OUT = Path(output)
    q = partial(query, con)
    # C. COUNTY OBSERVATION FLAG (daily) -> processed/county_observed_daily.parquet
    # observed = TRUE when:
    #   - the county has any EAGLE-I row (incl. zero/null) in that calendar year
    #   - EAGLE-I collected >= 90% of the day's snapshots nationally
    #   - the county's readings that day are not ALL null
    # Days are UTC days. Quality references exist for 2018-2022 only, so other years
    # use the nearest year (quality_ref_year says which).
    con.execute(f"""
    COPY (
        WITH county_day AS (   -- CT planning-region activity counts toward each old county it overlaps
            SELECT coalesce(w.county_fips, r.fips) AS fips, r.day,
                   sum(r.rows_any) AS rows_any, sum(r.rows_non_null) AS rows_non_null, sum(r.rows_null) AS rows_null
            FROM county_day_raw r LEFT JOIN ct_weights w ON w.region_fips = r.fips
            GROUP BY ALL
        ),
        county_year AS (SELECT DISTINCT fips, year(day) AS year FROM county_day),
        days AS (
            SELECT CAST(unnest(generate_series(DATE '{FIRST_YEAR}-01-01', DATE '{LAST_YEAR}-12-31', INTERVAL 1 DAY)) AS DATE) AS day
        ),
        grid AS (SELECT c.fips, c.state_abbr, c.fema_region, d.day FROM counties c CROSS JOIN days d)
        SELECT g.fips, g.day, year(g.day) AS year, g.state_abbr, g.fema_region,
               cy.fips IS NOT NULL                          AS reported_in_year,
               coalesce(n.snapshot_pct, 0)                  AS national_snapshot_pct,
               coalesce(cd.rows_any, 0)                     AS rows_any,
               coalesce(cd.rows_null, 0)                    AS null_readings,
               coalesce(cd.rows_null, 0) > 0 AND coalesce(cd.rows_non_null, 0) = 0 AS all_readings_null,
               cy.fips IS NOT NULL
                 AND coalesce(n.snapshot_pct, 0) >= {MIN_DAILY_SNAPSHOT_PCT}
                 AND NOT (coalesce(cd.rows_null, 0) > 0 AND coalesce(cd.rows_non_null, 0) = 0) AS observed,
               greatest(2018, least(2022, year(g.day)))     AS quality_ref_year,
               cov.pct_customers_tracked,
               dq.dqi
        FROM grid g
        LEFT JOIN county_year cy  ON cy.fips = g.fips AND cy.year = year(g.day)
        LEFT JOIN national_day n  ON n.day = g.day
        LEFT JOIN county_day cd   ON cd.fips = g.fips AND cd.day = g.day
        LEFT JOIN coverage cov    ON cov.state_abbr = g.state_abbr AND cov.ref_year = greatest(2018, least(2022, year(g.day)))
        LEFT JOIN dqi dq          ON dq.fema_region = g.fema_region AND dq.ref_year = greatest(2018, least(2022, year(g.day)))
        ORDER BY g.fips, g.day
    ) TO '{(OUT / "county_observed_daily.parquet").as_posix()}' (FORMAT parquet)
    """)
    con.execute(
        f"CREATE OR REPLACE VIEW observed_daily AS SELECT * FROM read_parquet('{(OUT / 'county_observed_daily.parquet').as_posix()}')"
    )

    observed_summary = q("""
    SELECT year,
           count(DISTINCT fips)                                AS counties,
           round(100 * avg(reported_in_year::INT), 1)          AS pct_county_days_in_reporting_year,
           sum((national_snapshot_pct < 0.9)::INT) / count(DISTINCT fips) AS national_gap_days,
           sum(all_readings_null::INT)                         AS county_days_all_null,
           round(100 * avg(observed::INT), 1)                  AS pct_county_days_observed
    FROM observed_daily GROUP BY year ORDER BY year
    """)
    return observed_summary


def clean_storms(con, output):
    """Normalize NOAA timestamps and geography, preserving original source filters."""
    OUT = Path(output)
    # D. STORM EVENTS -> processed/storm_events_clean.parquet
    # Kept columns are listed explicitly; everything else is dropped (printed below).
    # UTC = local standard time minus the offset in CZ_TIMEZONE (e.g. 'CST-6' -> +6 h).
    # NOAA uses its own state codes for territories; convert them to Census codes:
    # 96 Virgin Islands -> 78, 97 American Samoa -> 60, 98 Guam -> 66, 99 Puerto Rico -> 72
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE storms_clean AS
    WITH coded AS (
        SELECT *,
               CASE lpad(STATE_FIPS, 2, '0')
                   WHEN '96' THEN '78' WHEN '97' THEN '60' WHEN '98' THEN '66' WHEN '99' THEN '72'
                   ELSE lpad(STATE_FIPS, 2, '0') END AS census_state_fips
        FROM storms_raw
    ),
    typed AS (
        SELECT
            CAST(EVENT_ID AS BIGINT)                          AS event_id,
            CAST(EPISODE_ID AS BIGINT)                        AS episode_id,
            EVENT_TYPE                                        AS event_type,
            CZ_TYPE                                           AS cz_type,
            census_state_fips                                 AS state_fips,
            lpad(CZ_FIPS, 3, '0')                             AS cz_fips,
            census_state_fips || lpad(CZ_FIPS, 3, '0')        AS cz_code,
            CZ_NAME                                           AS cz_name,
            WFO                                               AS wfo,
            STRPTIME(BEGIN_DATE_TIME, '%d-%b-%y %H:%M:%S')
                - to_hours(CAST(regexp_extract(CZ_TIMEZONE, '(-?[0-9]+)$', 1) AS INT)) AS begin_utc,
            STRPTIME(END_DATE_TIME, '%d-%b-%y %H:%M:%S')
                - to_hours(CAST(regexp_extract(CZ_TIMEZONE, '(-?[0-9]+)$', 1) AS INT)) AS end_utc,
            TRY_CAST(MAGNITUDE AS DOUBLE)                     AS magnitude,
            MAGNITUDE_TYPE                                    AS magnitude_type,
            TRY_CAST(CATEGORY AS INT)                         AS hurricane_category,
            FLOOD_CAUSE                                       AS flood_cause,
            TRY_CAST(BEGIN_LAT AS DOUBLE)                     AS begin_lat,
            TRY_CAST(BEGIN_LON AS DOUBLE)                     AS begin_lon,
            TRY_CAST(END_LAT AS DOUBLE)                       AS end_lat,
            TRY_CAST(END_LON AS DOUBLE)                       AS end_lon
        FROM coded
        WHERE EVENT_TYPE NOT LIKE 'Marine%' AND EVENT_TYPE <> 'Waterspout'   -- over water, no county
    )
    SELECT *,
           year(begin_utc)                                       AS year,
           date_diff('minute', begin_utc, end_utc) / 60.0        AS duration_hours
    FROM typed
    WHERE year(begin_utc) BETWEEN {FIRST_YEAR} AND {LAST_YEAR}
    """)
    con.execute(
        f"COPY storms_clean TO '{(OUT / 'storm_events_clean.parquet').as_posix()}' (FORMAT parquet)"
    )


def map_storm_counties(con, output):
    """Expand dated forecast zones into counties and preserve mapping diagnostics."""
    OUT = Path(output)
    q = partial(query, con)
    # E. STORM x COUNTY -> processed/storm_county_events.parquet
    # County-coded events map directly. Zone-coded events go through the crosswalk
    # version in effect on the event's (UTC) begin date.
    con.execute(f"""
    COPY (
        WITH county_coded AS (
            SELECT event_id, cz_code AS fips, FALSE AS partial_county, 1 AS counties_in_zone, 'county' AS mapped_from
            FROM storms_clean WHERE cz_type = 'C'
        ),
        zone_coded AS (
            SELECT s.event_id, x.county_fips AS fips,
                   bool_or(x.partial_county) AS partial_county,
                   max(x.counties_in_zone)   AS counties_in_zone,
                   'zone'                    AS mapped_from
            FROM storms_clean s
            JOIN regions r USING (state_fips)
            JOIN crosswalk x
              ON x.state_zone = r.state_abbr || s.cz_fips
             AND s.begin_utc >= x.effective_date
             AND (x.next_effective_date IS NULL OR s.begin_utc < x.next_effective_date)
            WHERE s.cz_type = 'Z'
            GROUP BY s.event_id, x.county_fips
        ),
        mapped AS (SELECT * FROM county_coded UNION ALL SELECT * FROM zone_coded)
        SELECT s.event_id, s.episode_id, s.year, m.fips, s.event_type, s.cz_type, m.mapped_from,
               m.partial_county, m.counties_in_zone,
               s.begin_utc, s.end_utc, s.duration_hours,
               s.magnitude, s.magnitude_type, s.hurricane_category, s.flood_cause,
               s.begin_lat, s.begin_lon,
               c.denominator          AS mcc_customers,
               c.fips IS NOT NULL     AS fips_known,
               coalesce(o.observed, FALSE) AS county_observed_on_begin_day
        FROM mapped m
        JOIN storms_clean s USING (event_id)
        LEFT JOIN counties c USING (fips)
        LEFT JOIN observed_daily o ON o.fips = m.fips AND o.day = CAST(s.begin_utc AS DATE)
        ORDER BY s.begin_utc, m.fips
    ) TO '{(OUT / "storm_county_events.parquet").as_posix()}' (FORMAT parquet)
    """)
    con.execute(
        f"CREATE OR REPLACE VIEW storm_county AS SELECT * FROM read_parquet('{(OUT / 'storm_county_events.parquet').as_posix()}')"
    )

    storm_county_summary = q("""
    SELECT year,
           count(*)                                         AS event_county_rows,
           count(DISTINCT event_id)                         AS events,
           round(100 * avg((mapped_from = 'zone')::INT), 1) AS pct_from_zones,
           round(100 * avg(partial_county::INT), 1)         AS pct_partial_county,
           round(100 * avg(fips_known::INT), 2)             AS pct_fips_known,
           round(100 * avg(county_observed_on_begin_day::INT), 1) AS pct_county_observed
    FROM storm_county GROUP BY year ORDER BY year
    """)
    return storm_county_summary


def preparation_checks(con):
    """Return invariant violation counts and advisory unknown-geography counts."""
    q = partial(query, con)
    # F. OUTPUT CHECKS
    checks = {
        "outages_15min: duplicate fips+ts": q(
            "SELECT count(*) - count(DISTINCT (fips, ts)) FROM outages_15min"
        ).iat[0, 0],
        "outages_15min: rows with customers_out <= 0": q(
            "SELECT count(*) FROM outages_15min WHERE customers_out <= 0"
        ).iat[0, 0],
        "outages_15min: rows with pct_out > 1": q(
            "SELECT count(*) FROM outages_15min WHERE pct_out > 1"
        ).iat[0, 0],
        "outages_15min: rows off the 15-min grid": q(
            "SELECT count(*) FROM outages_15min WHERE minute(ts) % 15 <> 0 OR second(ts) <> 0"
        ).iat[0, 0],
        "outages_15min: fips not in county lookup": q(
            "SELECT count(DISTINCT fips) FROM outages_15min ANTI JOIN counties USING (fips)"
        ).iat[0, 0],
        "outages_15min: CT planning-region codes left": q(
            "SELECT count(*) FROM outages_15min WHERE fips BETWEEN '09100' AND '09199'"
        ).iat[0, 0],
        "outages_hourly: duplicate fips+hour": q(
            "SELECT count(*) - count(DISTINCT (fips, hour_utc)) FROM outages_hourly"
        ).iat[0, 0],
        "observed_daily: duplicate fips+day": q(
            "SELECT count(*) - count(DISTINCT (fips, day)) FROM observed_daily"
        ).iat[0, 0],
        "storm_events_clean: duplicate event_id": q(
            "SELECT count(*) - count(DISTINCT event_id) FROM storms_clean"
        ).iat[0, 0],
        "storm_events_clean: end before begin": q(
            "SELECT count(*) FROM storms_clean WHERE end_utc < begin_utc"
        ).iat[0, 0],
        "storm_county: duplicate event_id+fips": q(
            "SELECT count(*) - count(DISTINCT (event_id, fips)) FROM storm_county"
        ).iat[0, 0],
        "storm_county: fips not in county lookup": q(
            "SELECT count(DISTINCT fips) FROM storm_county WHERE NOT fips_known"
        ).iat[0, 0],
    }
    return checks


def prepare(paths=None, *, export_dir=None, memory_limit="16GB"):
    """Build a complete interim generation; reject invalid outputs before replacing old data."""
    paths = paths or ProjectPaths()
    with connection(paths, cached=True, memory_limit=memory_limit) as con:
        load_raw(con, paths)
        refs = load_references(paths)
        counties = build_counties(
            refs["census"], refs["mcc"], refs["ct_weights"], refs["regions"]
        )
        for name, frame in {**refs, "counties": counties}.items():
            con.register(name, frame)
        with staged_directory(paths.interim) as stage:
            counties.to_parquet(stage / "counties.parquet", index=False)
            clean_outages(con, stage)
            build_hourly(con, stage)
            observed = build_observation_flags(con, stage)
            clean_storms(con, stage)
            storms = map_storm_counties(con, stage)
            checks = preparation_checks(con)
            # Unknown storm codes are retained as diagnostics until feature-scope filtering.
            required = {
                k: v == 0
                for k, v in checks.items()
                if k != "storm_county: fips not in county lookup"
            }
            require_checks(required, name="preparation")
            (stage / "build.json").write_text(
                json.dumps({"checks": checks}, default=int, indent=2)
            )
    tables = {
        "checks": pd.DataFrame(list(checks.items()), columns=["check", "violations"]),
        "observed_daily": observed,
        "storm_counties": storms,
    }
    export_results(tables, directory=export_dir)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare"])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--memory-limit", default="16GB")
    args = parser.parse_args()
    paths = ProjectPaths(
        offline=args.offline, **({"revision": args.revision} if args.revision else {})
    )
    print(
        prepare(paths, export_dir=args.export_dir, memory_limit=args.memory_limit)[
            "checks"
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
