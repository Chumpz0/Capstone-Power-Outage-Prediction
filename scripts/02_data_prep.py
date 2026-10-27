"""Standalone Python version of notebooks/02_data_prep.ipynb.

Run with the project virtual environment. See scripts/README.md for inputs,
outputs, execution order, and differences from the notebook.
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # Save figures without requiring a notebook or GUI.


    # Source: 02_data_prep.ipynb, code cell 1
    # =============================================================================
    # 02 - DATA PREPARATION: clean EAGLE-I outages + NOAA Storm Events (2017-2025)
    # Summary of results lives in reports/02_data_prep_summary.md
    # data/ is read-only. Cleaned outputs go to processed/, check tables to reports/prep_tables/
    # Split: train 2017-2023, validation 2024, test 2025. All US counties.
    # =============================================================================

    # Source: 02_data_prep.ipynb, code cell 2
    # ---- 0. Setup ---------------------------------------------------------------
    import shutil
    from pathlib import Path

    import duckdb
    import pandas as pd

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    WORK = ROOT / "work"
    OUT = ROOT / "processed"
    TABLES = ROOT / "reports" / "prep_tables"
    for folder in [WORK, WORK / "tmp", OUT, TABLES]:
        folder.mkdir(parents=True, exist_ok=True)

    # Inputs: the local file when it is in the repo, otherwise streamed from the Hugging Face dataset
    HF_DATASET = "hf://datasets/DSCI-Capstone/county-outage-data"


    def src(rel):
        """Local path if the file is in the repo, otherwise its hf:// URL."""
        local = ROOT / rel
        return local.as_posix() if local.exists() else f"{HF_DATASET}/{rel}"

    # The two big CSVs are stored as Parquet on Hugging Face (every column still text). The CSV is used when present.
    EAGLEI_CSV = DATA / "02_combined" / "eaglei_outages_2014_2025.csv"
    STORMS_CSV = DATA / "02_combined" / "storm_events_details_all_years.csv"
    EAGLEI_PARQUET = src("data/02_combined/eaglei_outages") + "/*/*.parquet"
    STORMS_PARQUET = src("data/02_combined/storm_events_details_all_years.parquet")
    CENSUS_CSV = src("data/01_raw/census_counties/co-est2023-alldata.csv")
    CROSSWALK = src("data/reference/nws_zone_county_versions.parquet")
    CT_WEIGHTS = src("data/reference/ct_region_to_county_weights.csv")
    MCC_CSV = src("data/reference/eaglei_archive/MCC.csv")
    DQI_CSV = src("data/reference/eaglei_archive/DQI.csv")
    COVERAGE_CSV = src("data/reference/eaglei_archive/coverage_history.csv")

    FIRST_YEAR, LAST_YEAR = 2017, 2025
    MIN_DAILY_SNAPSHOT_PCT = 0.90  # a day with < 90% of its 96 snapshots counts as a collection gap

    # Raw CSVs are loaded once into DuckDB (same working file as notebook 01)
    con = duckdb.connect(str(WORK / "audit.duckdb"))
    con.execute(f"SET temp_directory = '{(WORK / 'tmp').as_posix()}'")
    con.execute("SET memory_limit = '16GB'")

    try:  # the dataset is private: streaming uses the token saved by `hf auth login` (not stored in any file here)
        con.execute("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")
    except duckdb.IOException:
        print("No Hugging Face login found. Files not in the repo can't be streamed until you run `hf auth login`.")


    def q(sql):
        """Run a SQL query and return a pandas DataFrame."""
        return con.execute(sql).df()


    tables = set(q("SHOW TABLES")["name"])
    if "eaglei_raw" not in tables:  # takes ~2 minutes the first time
        if EAGLEI_CSV.exists():
            con.execute(f"CREATE TABLE eaglei_raw AS SELECT * FROM read_csv('{EAGLEI_CSV.as_posix()}', all_varchar=true)")
        else:  # hive_partitioning off so SOURCE_YEAR stays text, as in the CSV
            con.execute(f"CREATE TABLE eaglei_raw AS SELECT * FROM read_parquet('{EAGLEI_PARQUET}', hive_partitioning=false)")
    if "storms_raw" not in tables:
        if STORMS_CSV.exists():
            con.execute(f"CREATE TABLE storms_raw AS SELECT * FROM read_csv('{STORMS_CSV.as_posix()}', all_varchar=true)")
        else:
            con.execute(f"CREATE TABLE storms_raw AS SELECT * FROM read_parquet('{STORMS_PARQUET}')")


    def fresh_dir(path):
        """Delete an output folder from a previous run so results are not mixed."""
        path = path.resolve()
        allowed = {ROOT / "processed" / "outages_15min", ROOT / "processed" / "outages_hourly"}
        if path not in allowed:
            raise ValueError(f"Refusing to replace an unexpected output directory: {path}")
        if path.exists():
            shutil.rmtree(path)
        return path.as_posix()

    # Source: 02_data_prep.ipynb, code cell 3
    # ---- 0b. Reference tables ----------------------------------------------------
    # Census counties (SUMLEV 050 = county rows)
    census = pd.read_csv(CENSUS_CSV, dtype=str, encoding="latin-1",
                         usecols=["SUMLEV", "STATE", "COUNTY", "STNAME", "CTYNAME", "POPESTIMATE2023"])
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
    crosswalk.loc[crosswalk["effective_date"] == crosswalk["effective_date"].min(), "effective_date"] = pd.Timestamp("2000-01-01")

    # State FIPS -> postal code (from the crosswalk) and FEMA region
    FEMA = {
        1: ["CT", "ME", "MA", "NH", "RI", "VT"], 2: ["NJ", "NY", "PR", "VI"],
        3: ["DE", "MD", "PA", "VA", "DC", "WV"], 4: ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"],
        5: ["IL", "IN", "MI", "MN", "OH", "WI"], 6: ["AR", "LA", "NM", "OK", "TX"],
        7: ["IA", "KS", "MO", "NE"], 8: ["CO", "MT", "ND", "SD", "UT", "WY"],
        9: ["AZ", "CA", "HI", "NV", "GU", "AS", "MP"], 10: ["AK", "ID", "OR", "WA"],
    }
    abbr_to_region = {a: r for r, states in FEMA.items() for a in states}
    state_abbr = (crosswalk[crosswalk["state"] != ""]
                  .assign(state_fips=lambda d: d["county_fips"].str[:2])
                  .groupby("state_fips")["state"].agg(lambda s: s.mode()[0]))
    regions = pd.DataFrame({"state_fips": state_abbr.index, "state_abbr": state_abbr.values})
    regions["fema_region"] = regions["state_abbr"].map(abbr_to_region).astype("Int64")

    # Quality references: % of customers tracked (by state) and DQI (by FEMA region), 2018-2022
    coverage = pd.read_csv(COVERAGE_CSV)
    coverage["ref_year"] = pd.to_datetime(coverage["year"], format="%m/%d/%y").dt.year
    coverage = coverage.rename(columns={"state": "state_abbr", "max_pct_covered": "pct_customers_tracked"})
    dqi = pd.read_csv(DQI_CSV).rename(columns={"fema": "fema_region", "year": "ref_year", "DQI": "dqi"})

    print(len(census), "Census counties |", len(mcc), "MCC counties |", len(ct_weights), "CT weight rows")

    # Source: 02_data_prep.ipynb, code cell 4
    # ---- 0c. County lookup + denominator -----------------------------------------
    # Denominator = MCC customers; if a county has no MCC value, estimate it from Census
    # population x the median customers-per-person ratio.
    counties = mcc[["fips", "mcc_customers"]].merge(
        census[["fips", "STNAME", "CTYNAME", "pop2023"]], on="fips", how="outer")
    counties = counties[~counties["fips"].isin(ct_weights["region_fips"])]  # CT uses the old 8 counties
    counties = counties[counties["fips"].str.fullmatch(r"\d{5}")]           # drop MCC's "Grand Total" row
    ratio = (counties["mcc_customers"] / counties["pop2023"]).median()
    counties["state_fips"] = counties["fips"].str[:2]

    # Some states' MCC totals are not believable (e.g. NC: 0.17 customers per person vs ~0.5 nationally).
    # If a state's MCC customers-per-person is below half the national median, use the Census estimate there.
    state_ratio = (counties.dropna(subset=["mcc_customers", "pop2023"]).groupby("state_fips")[["mcc_customers", "pop2023"]].sum()
                   .eval("mcc_customers / pop2023"))
    bad_mcc_states = state_ratio[state_ratio < 0.5 * ratio].index.tolist()
    print("states with unreliable MCC (Census estimate used):", bad_mcc_states, state_ratio[bad_mcc_states].round(3).tolist())
    use_mcc = counties["mcc_customers"].notna() & ~counties["state_fips"].isin(bad_mcc_states)

    # Single counties can be just as wrong (e.g. Buena Vista city VA: 1 customer for 6,566 people;
    # McDonald County MO: 178 K customers for 24 K people). Use the Census estimate when a county's
    # customers-per-person is below 0.1 or above 5 (provisional cut-offs, ~5x off the national median).
    county_ratio = counties["mcc_customers"] / counties["pop2023"]
    bad_mcc_counties = county_ratio.lt(0.1) | county_ratio.gt(5)
    print("counties with unreliable MCC (Census estimate used):",
          counties.loc[bad_mcc_counties & use_mcc, ["STNAME", "CTYNAME"]].values.tolist())
    use_mcc &= ~bad_mcc_counties

    counties["denominator"] = counties["mcc_customers"].where(use_mcc, (counties["pop2023"] * ratio).round())
    counties["denominator_source"] = use_mcc.map({True: "mcc", False: "census_estimate"})
    counties = counties.merge(regions, on="state_fips", how="left")
    counties = counties.rename(columns={"STNAME": "state_name", "CTYNAME": "county_name"})
    counties = counties[["fips", "state_fips", "state_abbr", "fema_region", "county_name", "state_name",
                         "pop2023", "mcc_customers", "denominator", "denominator_source"]]
    counties.to_parquet(OUT / "counties.parquet", index=False)

    for name, df in {"counties": counties, "ct_weights": ct_weights, "crosswalk": crosswalk,
                     "regions": regions, "coverage": coverage, "dqi": dqi}.items():
        con.register(name, df)

    print("customers per person used for estimates:", round(ratio, 3))
    print(counties["denominator_source"].value_counts())

    # Source: 02_data_prep.ipynb, code cell 5
    # =============================================================================
    # A. EAGLE-I OUTAGES (15-minute)
    # =============================================================================
    # ---- A1. Typed view, 2017+ ---------------------------------------------------
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

    # Source: 02_data_prep.ipynb, code cell 6
    # ---- A2. Row counts before cleaning ------------------------------------------
    waterfall = q("""
    SELECT year(raw_ts)                                   AS year,
           count(*)                                       AS raw_rows,
           sum((customers_out IS NULL)::INT)              AS null_rows_dropped,
           sum((customers_out = 0)::INT)                  AS zero_rows_dropped,
           sum((customers_out > 0)::INT)                  AS positive_rows,
           sum((raw_ts <> ts AND customers_out > 0)::INT) AS positive_rows_snapped
    FROM eaglei_typed GROUP BY 1 ORDER BY 1
    """)
    print(waterfall)

    # Source: 02_data_prep.ipynb, code cell 7
    # ---- A3. Per county-day activity, kept for the observation flag (section C) --
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
    print(q("SELECT count(*) AS county_days, (SELECT count(*) FROM national_day) AS days FROM county_day_raw"))

    # Source: 02_data_prep.ipynb, code cell 8
    # ---- A4. Clean 15-minute outage table -> processed/outages_15min/ -----------
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
    ) TO '{fresh_dir(OUT / "outages_15min")}' (FORMAT parquet, PARTITION_BY (year))
    """)
    con.execute(f"""
    CREATE OR REPLACE VIEW outages_15min AS
    SELECT * FROM read_parquet('{(OUT / "outages_15min").as_posix()}/*/*.parquet', hive_partitioning = true)
    """)
    print(q("SELECT count(*) AS rows, count(DISTINCT fips) AS counties FROM outages_15min"))

    # Source: 02_data_prep.ipynb, code cell 9
    # ---- A5. Row counts after cleaning -------------------------------------------
    after = q("""
    SELECT year,
           count(*)                       AS final_rows,
           sum(over_mcc::INT)             AS capped_at_mcc,
           sum((mcc_customers IS NULL)::INT) AS rows_without_denominator,
           count(DISTINCT fips)           AS counties
    FROM outages_15min GROUP BY year ORDER BY year
    """)
    waterfall = waterfall.merge(after, on="year", how="outer")
    waterfall["rows_merged_or_shifted"] = waterfall["positive_rows"] - waterfall["final_rows"]
    waterfall.to_csv(TABLES / "outage_row_waterfall.csv", index=False)
    print(waterfall)

    # Source: 02_data_prep.ipynb, code cell 10
    # ---- A6. Connecticut check: 2025 (mixed codes) vs 2024 -----------------------
    # If the conversion double counted, 2025 customer-hours would jump far above 2024.
    print(q("""
    SELECT year, count(DISTINCT fips) AS ct_counties,
           round(sum(customers_out) / 4 / 1e6, 2) AS million_customer_hours_out
    FROM outages_15min WHERE fips LIKE '09%' AND year IN (2023, 2024, 2025)
    GROUP BY year ORDER BY year
    """))

    # Source: 02_data_prep.ipynb, code cell 11
    # ---- A7. When did EAGLE-I switch CT from old counties to planning regions? ----
    print(q("""
    SELECT fips BETWEEN '09100' AND '09199' AS planning_region_code, min(raw_ts) AS first_ts, max(raw_ts) AS last_ts
    FROM eaglei_typed WHERE fips LIKE '09%' AND year(raw_ts) = 2025 GROUP BY 1
    """))

    # Source: 02_data_prep.ipynb, code cell 12
    # =============================================================================
    # B. HOURLY OUTAGES -> processed/outages_hourly/
    # =============================================================================
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
    ) TO '{fresh_dir(OUT / "outages_hourly")}' (FORMAT parquet, PARTITION_BY (year))
    """)
    con.execute(f"""
    CREATE OR REPLACE VIEW outages_hourly AS
    SELECT * FROM read_parquet('{(OUT / "outages_hourly").as_posix()}/*/*.parquet', hive_partitioning = true)
    """)
    print(q("SELECT year, count(*) AS rows FROM outages_hourly GROUP BY year ORDER BY year"))

    # Source: 02_data_prep.ipynb, code cell 13
    # =============================================================================
    # C. COUNTY OBSERVATION FLAG (daily) -> processed/county_observed_daily.parquet
    # =============================================================================
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
    con.execute(f"CREATE OR REPLACE VIEW observed_daily AS SELECT * FROM read_parquet('{(OUT / 'county_observed_daily.parquet').as_posix()}')")

    observed_summary = q("""
    SELECT year,
           count(DISTINCT fips)                                AS counties,
           round(100 * avg(reported_in_year::INT), 1)          AS pct_county_days_in_reporting_year,
           sum((national_snapshot_pct < 0.9)::INT) / count(DISTINCT fips) AS national_gap_days,
           sum(all_readings_null::INT)                         AS county_days_all_null,
           round(100 * avg(observed::INT), 1)                  AS pct_county_days_observed
    FROM observed_daily GROUP BY year ORDER BY year
    """)
    observed_summary.to_csv(TABLES / "observed_daily_summary.csv", index=False)
    print(observed_summary)

    # Source: 02_data_prep.ipynb, code cell 14
    # ---- C2. Which dates are national collection gaps? ---------------------------
    print(q(f"SELECT day, round(snapshot_pct, 2) AS snapshot_pct FROM national_day WHERE snapshot_pct < {MIN_DAILY_SNAPSHOT_PCT} ORDER BY day"))

    # Source: 02_data_prep.ipynb, code cell 15
    # ---- C3. Counties never observed in a year -----------------------------------
    print(q("""
    SELECT year, count(DISTINCT fips) AS counties_not_reported, list(DISTINCT state_abbr ORDER BY state_abbr) AS states
    FROM observed_daily WHERE NOT reported_in_year
    GROUP BY year ORDER BY year
    """))

    # Source: 02_data_prep.ipynb, code cell 16
    # =============================================================================
    # D. STORM EVENTS -> processed/storm_events_clean.parquet
    # =============================================================================
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
    con.execute(f"COPY storms_clean TO '{(OUT / 'storm_events_clean.parquet').as_posix()}' (FORMAT parquet)")

    raw_cols = q("DESCRIBE storms_raw")["column_name"].tolist()
    USED_RAW = {"EVENT_ID", "EPISODE_ID", "EVENT_TYPE", "CZ_TYPE", "STATE_FIPS", "CZ_FIPS", "CZ_NAME", "WFO",
                "BEGIN_DATE_TIME", "END_DATE_TIME", "CZ_TIMEZONE", "MAGNITUDE", "MAGNITUDE_TYPE", "CATEGORY",
                "FLOOD_CAUSE", "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON"}
    print("dropped storm columns:", [c for c in raw_cols if c not in USED_RAW])

    storm_counts = q(f"""
    SELECT year(STRPTIME(BEGIN_DATE_TIME, '%d-%b-%y %H:%M:%S')) AS local_year, count(*) AS raw_events,
           sum((EVENT_TYPE LIKE 'Marine%' OR EVENT_TYPE = 'Waterspout')::INT) AS marine_dropped
    FROM storms_raw GROUP BY 1 HAVING local_year BETWEEN {FIRST_YEAR} AND {LAST_YEAR} ORDER BY 1
    """).merge(q("SELECT year AS local_year, count(*) AS clean_events FROM storms_clean GROUP BY 1"), on="local_year")
    print(storm_counts)

    # Source: 02_data_prep.ipynb, code cell 17
    # =============================================================================
    # E. STORM x COUNTY -> processed/storm_county_events.parquet
    # =============================================================================
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
    ) TO '{(OUT / 'storm_county_events.parquet').as_posix()}' (FORMAT parquet)
    """)
    con.execute(f"CREATE OR REPLACE VIEW storm_county AS SELECT * FROM read_parquet('{(OUT / 'storm_county_events.parquet').as_posix()}')")

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
    storm_county_summary.to_csv(TABLES / "storm_county_summary.csv", index=False)
    print(storm_county_summary)

    # Source: 02_data_prep.ipynb, code cell 18
    # ---- E2. Events that did not map to any county --------------------------------
    print(q("""
    SELECT s.event_type, count(*) AS unmapped_events
    FROM storms_clean s ANTI JOIN storm_county sc USING (event_id)
    GROUP BY 1 ORDER BY 2 DESC LIMIT 10
    """))

    # Source: 02_data_prep.ipynb, code cell 19
    # ---- E3. Event types after mapping --------------------------------------------
    print(q("""
    SELECT event_type, count(*) AS event_county_rows, count(DISTINCT event_id) AS events,
           round(100 * avg(county_observed_on_begin_day::INT), 1) AS pct_county_observed
    FROM storm_county GROUP BY 1 ORDER BY 2 DESC LIMIT 20
    """))

    # Source: 02_data_prep.ipynb, code cell 20
    # =============================================================================
    # F. OUTPUT CHECKS
    # =============================================================================
    checks = {
        "outages_15min: duplicate fips+ts":
            q("SELECT count(*) - count(DISTINCT (fips, ts)) FROM outages_15min").iat[0, 0],
        "outages_15min: rows with customers_out <= 0":
            q("SELECT count(*) FROM outages_15min WHERE customers_out <= 0").iat[0, 0],
        "outages_15min: rows with pct_out > 1":
            q("SELECT count(*) FROM outages_15min WHERE pct_out > 1").iat[0, 0],
        "outages_15min: rows off the 15-min grid":
            q("SELECT count(*) FROM outages_15min WHERE minute(ts) % 15 <> 0 OR second(ts) <> 0").iat[0, 0],
        "outages_15min: fips not in county lookup":
            q("SELECT count(DISTINCT fips) FROM outages_15min ANTI JOIN counties USING (fips)").iat[0, 0],
        "outages_15min: CT planning-region codes left":
            q("SELECT count(*) FROM outages_15min WHERE fips BETWEEN '09100' AND '09199'").iat[0, 0],
        "outages_hourly: duplicate fips+hour":
            q("SELECT count(*) - count(DISTINCT (fips, hour_utc)) FROM outages_hourly").iat[0, 0],
        "observed_daily: duplicate fips+day":
            q("SELECT count(*) - count(DISTINCT (fips, day)) FROM observed_daily").iat[0, 0],
        "storm_events_clean: duplicate event_id":
            q("SELECT count(*) - count(DISTINCT event_id) FROM storms_clean").iat[0, 0],
        "storm_events_clean: end before begin":
            q("SELECT count(*) FROM storms_clean WHERE end_utc < begin_utc").iat[0, 0],
        "storm_county: duplicate event_id+fips":
            q("SELECT count(*) - count(DISTINCT (event_id, fips)) FROM storm_county").iat[0, 0],
        "storm_county: fips not in county lookup":
            q("SELECT count(DISTINCT fips) FROM storm_county WHERE NOT fips_known").iat[0, 0],
    }
    checks = pd.DataFrame({"check": list(checks), "value": list(checks.values())})
    checks.to_csv(TABLES / "output_checks.csv", index=False)
    print(checks)

    # Source: 02_data_prep.ipynb, code cell 21
    # ---- F2. Unknown storm FIPS (not in the county lookup) ------------------------
    print(q("""
    SELECT fips, count(*) AS rows, min(year) AS first_year, max(year) AS last_year
    FROM storm_county WHERE NOT fips_known GROUP BY fips ORDER BY rows DESC LIMIT 15
    """))

    # Source: 02_data_prep.ipynb, code cell 22
    # ---- F3. Output files and sizes -----------------------------------------------
    files = pd.DataFrame([{"file": str(p.relative_to(ROOT)), "mb": round(p.stat().st_size / 1e6, 1)}
                          for p in sorted(OUT.rglob("*.parquet"))])
    files.to_csv(TABLES / "output_files.csv", index=False)
    print("total MB:", files["mb"].sum())
    print(files.groupby(files["file"].str.split(r"[\\/]").str[1])["mb"].sum())

    # Source: 02_data_prep.ipynb, code cell 23
    con.close()


if __name__ == "__main__":
    main()
