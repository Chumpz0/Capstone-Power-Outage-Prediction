"""Standalone Python version of notebooks/01_data_audit.ipynb.

Run with the project virtual environment. See scripts/README.md for inputs,
outputs, execution order, and differences from the notebook.
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # Save figures without requiring a notebook or GUI.


    # Source: 01_data_audit.ipynb, code cell 1
    # =============================================================================
    # 01 - DATA QUALITY AUDIT: EAGLE-I outages + NOAA Storm Events (2014-2025)
    # Summary of results lives in reports/01_data_audit_summary.md
    # data/ is read-only. Working copies go to work/, small result tables to reports/audit_tables/
    # =============================================================================

    # Source: 01_data_audit.ipynb, code cell 2
    # ---- 0. Setup ---------------------------------------------------------------
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 120)

    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    WORK = ROOT / "work"
    TABLES = ROOT / "reports" / "audit_tables"
    WORK.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

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
    MCC_CSV = src("data/reference/eaglei_archive/MCC.csv")
    DQI_CSV = src("data/reference/eaglei_archive/DQI.csv")
    COVERAGE_CSV = src("data/reference/eaglei_archive/coverage_history.csv")

    NULL_CUTOFF = 50  # % null above which a column becomes a drop candidate

    # The 13 GB CSV is too big for pandas, so we load it once into a DuckDB file.
    # Every column is read as text so we can see the raw values before typing them.
    con = duckdb.connect(str(WORK / "audit.duckdb"))

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

    # Source: 01_data_audit.ipynb, code cell 3
    # ---- 0b. Reference tables: Census counties, MCC customers, FEMA regions ----
    # Census county list (SUMLEV 050 = county rows). FIPS = 2-digit state + 3-digit county.
    census = pd.read_csv(CENSUS_CSV, dtype=str, encoding="latin-1",
                         usecols=["SUMLEV", "STATE", "COUNTY", "STNAME", "CTYNAME", "POPESTIMATE2023"])
    census = census[census["SUMLEV"] == "050"].copy()
    census["fips"] = census["STATE"] + census["COUNTY"]
    census["pop2023"] = census["POPESTIMATE2023"].astype(int)
    census = census[["fips", "STNAME", "CTYNAME", "pop2023"]]

    # MCC = EAGLE-I's "modeled customer count" per county (the denominator for % out)
    mcc = pd.read_csv(MCC_CSV, dtype=str, encoding="utf-8-sig")
    mcc["fips"] = mcc["County_FIPS"].str.zfill(5)
    mcc["mcc_customers"] = mcc["Customers"].astype(int)
    mcc = mcc[["fips", "mcc_customers"]]

    # NWS zone -> county crosswalk (one row per zone/county pair per NWS file version)
    crosswalk = pd.read_parquet(CROSSWALK)
    # The oldest version starts 2014-10-07; let it also cover Jan-Sep 2014
    first_version = crosswalk["effective_date"] == crosswalk["effective_date"].min()
    crosswalk.loc[first_version, "effective_date"] = pd.Timestamp("2014-01-01").date()

    # State FIPS -> postal code, taken from the crosswalk itself (no hand-typed list)
    state_abbr = (crosswalk[crosswalk["state"] != ""]
                  .assign(state_fips=lambda d: d["county_fips"].str[:2])
                  .groupby("state_fips")["state"].agg(lambda s: s.mode()[0]))

    # FEMA regions (same grouping EAGLE-I uses for its Data Quality Index).
    # Region IV is our "Southeast".
    FEMA = {
        1: ["CT", "ME", "MA", "NH", "RI", "VT"], 2: ["NJ", "NY", "PR", "VI"],
        3: ["DE", "MD", "PA", "VA", "DC", "WV"], 4: ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"],
        5: ["IL", "IN", "MI", "MN", "OH", "WI"], 6: ["AR", "LA", "NM", "OK", "TX"],
        7: ["IA", "KS", "MO", "NE"], 8: ["CO", "MT", "ND", "SD", "UT", "WY"],
        9: ["AZ", "CA", "HI", "NV", "GU", "AS", "MP"], 10: ["AK", "ID", "OR", "WA"],
    }
    abbr_to_region = {a: r for r, states in FEMA.items() for a in states}
    regions = pd.DataFrame({"state_fips": state_abbr.index, "abbr": state_abbr.values})
    regions["fema_region"] = regions["abbr"].map(abbr_to_region).astype("Int64")
    regions["is_southeast"] = regions["fema_region"] == 4

    # Register the small tables so SQL can join to them
    for name, df in {"census": census, "mcc": mcc, "crosswalk": crosswalk, "regions": regions}.items():
        con.register(name, df)

    print(len(census), "Census counties |", len(mcc), "MCC counties |", len(crosswalk), "crosswalk rows")
    print(regions.head())

    # Source: 01_data_audit.ipynb, code cell 4
    # =============================================================================
    # 1. EAGLE-I OUTAGES
    # =============================================================================
    # ---- 1a. Shape, columns and a sample -----------------------------------------
    print(q("SELECT count(*) AS rows FROM eaglei_raw"))
    print(q("DESCRIBE eaglei_raw")[["column_name", "column_type"]])
    print(q("SELECT * FROM eaglei_raw USING SAMPLE 5 ROWS"))

    # Source: 01_data_audit.ipynb, code cell 5
    # ---- 1b. Can each text column be converted to its intended type? -------------
    # A value that is present but fails conversion is a type problem (not a null).
    print(q("""
    SELECT
        count(*)                                                                   AS rows,
        sum((length(fips_code) <> 5 OR NOT regexp_full_match(fips_code, '\\d{5}'))::INT) AS bad_fips_format,
        sum((customers_out IS NOT NULL AND TRY_CAST(customers_out AS BIGINT) IS NULL)::INT) AS bad_customers_out,
        sum((total_customers IS NOT NULL AND TRY_CAST(total_customers AS BIGINT) IS NULL)::INT) AS bad_total_customers,
        sum((TRY_CAST(run_start_time AS TIMESTAMP) IS NULL)::INT)                   AS bad_timestamp
    FROM eaglei_raw
    """))

    # Source: 01_data_audit.ipynb, code cell 6
    # ---- 1c. Typed view used for the rest of the notebook ------------------------
    con.execute("""
    CREATE OR REPLACE VIEW eaglei AS
    SELECT
        CAST(SOURCE_YEAR AS INT)               AS year,
        fips_code                              AS fips,
        county,
        state,
        TRY_CAST(customers_out AS BIGINT)      AS customers_out,
        TRY_CAST(total_customers AS BIGINT)    AS total_customers,
        CAST(run_start_time AS TIMESTAMP)      AS ts          -- EAGLE-I timestamps are UTC
    FROM eaglei_raw
    """)

    # Source: 01_data_audit.ipynb, code cell 7
    # ---- 1d. Null % per column, by year ------------------------------------------
    eaglei_nulls = q("""
    SELECT year,
           count(*)                                          AS rows,
           round(100 * avg((customers_out IS NULL)::INT), 2)   AS pct_null_customers_out,
           round(100 * avg((total_customers IS NULL)::INT), 2) AS pct_null_total_customers,
           round(100 * avg((county IS NULL)::INT), 2)          AS pct_null_county,
           round(100 * avg((state IS NULL)::INT), 2)           AS pct_null_state
    FROM eaglei GROUP BY year ORDER BY year
    """)
    eaglei_nulls.to_csv(TABLES / "eaglei_nulls_by_year.csv", index=False)
    print(eaglei_nulls)

    # Source: 01_data_audit.ipynb, code cell 8
    # ---- 1e. Zeros: are "no outage" rows stored, or left out? -------------------
    # If zeros are never stored, a missing row could mean 0 outages OR no data collected.
    print(q("""
    SELECT year,
           sum((customers_out = 0)::INT)  AS zero_rows,
           sum((customers_out > 0)::INT)  AS positive_rows,
           sum((customers_out < 0)::INT)  AS negative_rows
    FROM eaglei GROUP BY year ORDER BY year
    """))

    # Source: 01_data_audit.ipynb, code cell 9
    # ---- 1f. Duplicates (same county + same timestamp) ---------------------------
    print(q("""
    SELECT year, count(*) - count(DISTINCT (fips, ts)) AS duplicate_county_timestamps
    FROM eaglei GROUP BY year ORDER BY year
    """))

    # Source: 01_data_audit.ipynb, code cell 10
    # ---- 1g. Value range of customers_out ----------------------------------------
    print(q("""
    SELECT
        quantile_cont(customers_out, [0.5, 0.9, 0.99, 0.999]) AS p50_p90_p99_p999,
        max(customers_out)                                     AS max_out
    FROM eaglei
    """))

    # Source: 01_data_audit.ipynb, code cell 11
    # ---- 1h. Largest single readings vs the county's customer count (MCC) --------
    # Readings above the county's total customers are physically impossible.
    print(q("""
    SELECT e.year, e.fips, e.county, e.state, e.ts, e.customers_out, m.mcc_customers,
           round(e.customers_out / m.mcc_customers, 2) AS ratio_to_mcc
    FROM eaglei e LEFT JOIN mcc m USING (fips)
    ORDER BY e.customers_out DESC NULLS LAST LIMIT 10
    """))

    # Source: 01_data_audit.ipynb, code cell 12
    # ---- 1i. How often does customers_out exceed the county customer count? ------
    over_mcc = q("""
    SELECT e.year,
           count(*)                                                     AS rows_with_mcc,
           sum((e.customers_out > m.mcc_customers)::INT)                AS rows_over_mcc,
           count(DISTINCT CASE WHEN e.customers_out > m.mcc_customers THEN e.fips END) AS counties_over_mcc
    FROM eaglei e JOIN mcc m USING (fips)
    GROUP BY e.year ORDER BY e.year
    """)
    print(over_mcc)

    # Source: 01_data_audit.ipynb, code cell 13
    # ---- 1j. 2024 is the only year with total_customers: check it against MCC ---
    tc_vs_mcc = q("""
    WITH county_tc AS (
        SELECT fips, max(total_customers) AS total_customers_2024
        FROM eaglei WHERE year = 2024 GROUP BY fips
    )
    SELECT r.abbr AS state,
           sum(c.total_customers_2024)                         AS eaglei_total_2024,
           sum(m.mcc_customers)                                AS mcc_total,
           round(sum(c.total_customers_2024) / sum(m.mcc_customers), 2) AS ratio
    FROM county_tc c
    JOIN mcc m USING (fips)
    JOIN regions r ON r.state_fips = left(c.fips, 2)
    GROUP BY r.abbr ORDER BY ratio DESC
    """)
    tc_vs_mcc.to_csv(TABLES / "total_customers_2024_vs_mcc.csv", index=False)
    print(tc_vs_mcc.head(10))

    # Source: 01_data_audit.ipynb, code cell 14
    # ---- 1k. rows where customers_out > total_customers (2024) -------------------
    print(q("""
    SELECT count(*) AS rows_2024, sum((customers_out > total_customers)::INT) AS out_gt_total
    FROM eaglei WHERE year = 2024
    """))

    # Source: 01_data_audit.ipynb, code cell 15
    # ---- 1l. FIPS validity: do EAGLE-I county codes exist in the Census list? ----
    fips_check = q("""
    SELECT e.fips, any_value(e.county) AS county, any_value(e.state) AS state,
           min(e.year) AS first_year, max(e.year) AS last_year, count(*) AS rows,
           c.fips IS NOT NULL AS in_census, m.fips IS NOT NULL AS in_mcc
    FROM eaglei e
    LEFT JOIN census c USING (fips)
    LEFT JOIN mcc m USING (fips)
    GROUP BY e.fips, c.fips, m.fips
    """)
    print(fips_check[["in_census", "in_mcc"]].value_counts())
    print(fips_check[~fips_check["in_census"]].sort_values("rows", ascending=False).head(25))

    # Source: 01_data_audit.ipynb, code cell 16
    # ---- 1m. State name spelling ------------------------------------------------
    # The same territory should not appear under two names.
    print(q("SELECT state, count(DISTINCT fips) AS counties, min(year) AS first_year, max(year) AS last_year "
      "FROM eaglei WHERE state ILIKE '%virgin%' OR state ILIKE '%puerto%' GROUP BY state"))

    # Source: 01_data_audit.ipynb, code cell 17
    # ---- 1n. County coverage per year --------------------------------------------
    coverage = q("""
    SELECT year, count(DISTINCT fips) AS counties_reported, min(ts) AS first_ts, max(ts) AS last_ts
    FROM eaglei GROUP BY year ORDER BY year
    """)
    coverage["pct_of_census_counties"] = (100 * coverage["counties_reported"] / len(census)).round(1)
    coverage.to_csv(TABLES / "eaglei_county_coverage_by_year.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(coverage["year"], coverage["counties_reported"])
    ax.axhline(len(census), color="black", linestyle="--", linewidth=1, label="Census counties")
    ax.set_title("Counties with any EAGLE-I record, by year")
    ax.legend()
    fig_dir = ROOT / "reports" / "figures" / "audit"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "01_county_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(coverage)

    # Source: 01_data_audit.ipynb, code cell 18
    # ---- 1o. Snapshot grid: are any 15-minute snapshots missing? -----------------
    # EAGLE-I takes a snapshot every 15 minutes. We compare the timestamps present
    # against the full expected grid for each year.
    grid = q("""
    WITH bounds AS (
        SELECT year, min(ts) AS start_ts, (make_timestamp(year, 12, 31, 23, 45, 0)) AS end_ts
        FROM eaglei GROUP BY year
    ),
    expected AS (
        SELECT year, unnest(generate_series(start_ts, end_ts, INTERVAL 15 MINUTE)) AS ts FROM bounds
    ),
    present AS (SELECT DISTINCT year, ts FROM eaglei)
    SELECT ex.year,
           count(*)                          AS expected_snapshots,
           count(p.ts)                       AS present_snapshots,
           count(*) - count(p.ts)            AS missing_snapshots,
           round(100 * count(p.ts) / count(*), 2) AS pct_present
    FROM expected ex LEFT JOIN present p USING (year, ts)
    GROUP BY ex.year ORDER BY ex.year
    """)
    off_grid = q("""
    SELECT year, count(*) AS rows_off_15min_grid
    FROM eaglei WHERE minute(ts) % 15 <> 0 OR second(ts) <> 0
    GROUP BY year ORDER BY year
    """)
    grid = grid.merge(off_grid, on="year", how="left").fillna({"rows_off_15min_grid": 0})
    grid.to_csv(TABLES / "eaglei_snapshot_grid.csv", index=False)
    print(grid)

    # Source: 01_data_audit.ipynb, code cell 19
    # ---- 1p. Longest collection gaps (consecutive missing snapshots) -------------
    print(q("""
    WITH snaps AS (
        SELECT DISTINCT ts FROM eaglei WHERE minute(ts) % 15 = 0
    ),
    gaps AS (
        SELECT ts AS gap_after, lead(ts) OVER (ORDER BY ts) AS next_ts FROM snaps
    )
    SELECT gap_after, next_ts, next_ts - gap_after AS gap_length
    FROM gaps WHERE next_ts - gap_after > INTERVAL 2 HOUR
    ORDER BY gap_length DESC LIMIT 15
    """))

    # Source: 01_data_audit.ipynb, code cell 20
    # ---- 1q. Null customers_out: whole snapshots or scattered counties? ----------
    print(q("""
    SELECT year,
           count(*)                     AS null_rows,
           count(DISTINCT ts)           AS snapshots_with_nulls,
           count(DISTINCT fips)         AS counties_with_nulls,
           count(DISTINCT left(fips,2)) AS states_with_nulls
    FROM eaglei WHERE customers_out IS NULL
    GROUP BY year ORDER BY year
    """))

    # Source: 01_data_audit.ipynb, code cell 21
    # ---- 1r. Which states hold the null customers_out rows? ----------------------
    print(q("""
    SELECT year, state, count(*) AS null_rows
    FROM eaglei WHERE customers_out IS NULL
    GROUP BY year, state QUALIFY row_number() OVER (PARTITION BY year ORDER BY count(*) DESC) <= 3
    ORDER BY year, null_rows DESC
    """))

    # Source: 01_data_audit.ipynb, code cell 22
    # ---- 1s. Reporting rate per county -------------------------------------------
    # Share of the year's snapshots in which a county has a row.
    # Before 2022 zeros are not stored, so a low rate there mostly means "no outage".
    # From 2022 zeros are stored, so a low rate means the county was not being collected.
    report_rate = q("""
    WITH snaps AS (SELECT year, count(DISTINCT ts) AS n_snaps FROM eaglei GROUP BY year),
    county_rows AS (SELECT year, fips, count(*) AS n_rows FROM eaglei GROUP BY year, fips)
    SELECT c.year,
           round(median(c.n_rows / s.n_snaps) * 100, 1)        AS median_pct_snapshots_with_row,
           round(100 * avg((c.n_rows / s.n_snaps > 0.9)::INT), 1) AS pct_counties_reporting_90pct
    FROM county_rows c JOIN snaps s USING (year)
    GROUP BY c.year ORDER BY c.year
    """)
    print(report_rate)

    # Source: 01_data_audit.ipynb, code cell 23
    # =============================================================================
    # 2. NOAA STORM EVENTS (details)
    # =============================================================================
    # ---- 2a. Shape, columns, sample ----------------------------------------------
    print(q("SELECT count(*) AS rows, count(DISTINCT EVENT_ID) AS events, count(DISTINCT EPISODE_ID) AS episodes FROM storms_raw"))
    print(q("SELECT * FROM storms_raw USING SAMPLE 3 ROWS"))

    # Source: 01_data_audit.ipynb, code cell 24
    # ---- 2b. Null % for every column ---------------------------------------------
    storm_cols = q("DESCRIBE storms_raw")["column_name"].tolist()
    null_sql = ", ".join(f"round(100 * avg(({c} IS NULL)::INT), 2) AS \"{c}\"" for c in storm_cols)
    storm_nulls = q(f"SELECT {null_sql} FROM storms_raw").T.rename(columns={0: "pct_null_all_rows"})

    # Some columns only apply to certain events (e.g. tornado width only for tornadoes).
    # For those we also measure nulls inside the rows where the column applies.
    APPLIES_TO = {
        "MAGNITUDE":          "EVENT_TYPE IN ('Thunderstorm Wind','Hail','High Wind','Strong Wind','Marine Thunderstorm Wind','Marine High Wind','Marine Hail','Marine Strong Wind')",
        "MAGNITUDE_TYPE":     "EVENT_TYPE IN ('Thunderstorm Wind','High Wind','Strong Wind','Marine Thunderstorm Wind','Marine High Wind','Marine Strong Wind')",
        "TOR_F_SCALE":        "EVENT_TYPE = 'Tornado'",
        "TOR_LENGTH":         "EVENT_TYPE = 'Tornado'",
        "TOR_WIDTH":          "EVENT_TYPE = 'Tornado'",
        "TOR_OTHER_WFO":      "EVENT_TYPE = 'Tornado'",
        "TOR_OTHER_CZ_STATE": "EVENT_TYPE = 'Tornado'",
        "TOR_OTHER_CZ_FIPS":  "EVENT_TYPE = 'Tornado'",
        "TOR_OTHER_CZ_NAME":  "EVENT_TYPE = 'Tornado'",
        "FLOOD_CAUSE":        "EVENT_TYPE IN ('Flood','Flash Flood','Debris Flow')",
        "CATEGORY":           "EVENT_TYPE = 'Hurricane (Typhoon)'",
        "BEGIN_RANGE":        "CZ_TYPE = 'C'",
        "BEGIN_AZIMUTH":      "CZ_TYPE = 'C'",
        "BEGIN_LOCATION":     "CZ_TYPE = 'C'",
        "END_RANGE":          "CZ_TYPE = 'C'",
        "END_AZIMUTH":        "CZ_TYPE = 'C'",
        "END_LOCATION":       "CZ_TYPE = 'C'",
        "BEGIN_LAT":          "CZ_TYPE = 'C'",
        "BEGIN_LON":          "CZ_TYPE = 'C'",
        "END_LAT":            "CZ_TYPE = 'C'",
        "END_LON":            "CZ_TYPE = 'C'",
    }
    storm_nulls["applies_to"] = pd.Series(APPLIES_TO)
    storm_nulls["pct_null_where_applies"] = [
        q(f"SELECT round(100 * avg(({c} IS NULL)::INT), 2) FROM storms_raw WHERE {APPLIES_TO[c]}").iat[0, 0]
        if c in APPLIES_TO else storm_nulls.loc[c, "pct_null_all_rows"]
        for c in storm_nulls.index
    ]
    storm_nulls["drop_candidate"] = storm_nulls["pct_null_where_applies"] > NULL_CUTOFF
    storm_nulls.to_csv(TABLES / "storm_null_rates.csv")
    print(storm_nulls.sort_values("pct_null_where_applies", ascending=False))

    # Source: 01_data_audit.ipynb, code cell 25
    # ---- 2c. Null % chart --------------------------------------------------------
    plot_df = storm_nulls.sort_values("pct_null_where_applies")
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.barh(plot_df.index, plot_df["pct_null_all_rows"], color="lightgray", label="all rows")
    ax.barh(plot_df.index, plot_df["pct_null_where_applies"], color="tab:blue", label="rows where column applies")
    ax.axvline(NULL_CUTOFF, color="red", linestyle="--", linewidth=1, label=f"{NULL_CUTOFF}% cutoff")
    ax.set_xlabel("% null")
    ax.set_title("Storm Events: null % per column")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig_dir = ROOT / "reports" / "figures" / "audit"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "02_storm_column_nulls.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Source: 01_data_audit.ipynb, code cell 26
    # ---- 2d. Type conversion checks ----------------------------------------------
    # Damage is stored as text like "10.00K"; times as "18-FEB-14 10:00:00" in local standard time.
    print(q("""
    SELECT
        sum((TRY_STRPTIME(BEGIN_DATE_TIME, '%d-%b-%y %H:%M:%S') IS NULL)::INT) AS bad_begin_time,
        sum((TRY_STRPTIME(END_DATE_TIME,   '%d-%b-%y %H:%M:%S') IS NULL)::INT) AS bad_end_time,
        sum((MAGNITUDE IS NOT NULL AND TRY_CAST(MAGNITUDE AS DOUBLE) IS NULL)::INT) AS bad_magnitude,
        sum((BEGIN_LAT IS NOT NULL AND TRY_CAST(BEGIN_LAT AS DOUBLE) IS NULL)::INT) AS bad_lat,
        sum((DAMAGE_PROPERTY IS NOT NULL AND NOT regexp_full_match(DAMAGE_PROPERTY, '[0-9.]+[KMB]?'))::INT) AS bad_damage,
        sum((STRPTIME(END_DATE_TIME, '%d-%b-%y %H:%M:%S') < STRPTIME(BEGIN_DATE_TIME, '%d-%b-%y %H:%M:%S'))::INT) AS end_before_begin
    FROM storms_raw
    """))

    # Source: 01_data_audit.ipynb, code cell 27
    # ---- 2e. Damage suffixes (K = thousand, M = million, B = billion) -----------
    print(q("SELECT regexp_extract(DAMAGE_PROPERTY, '[A-Z]$') AS suffix, count(*) AS rows FROM storms_raw GROUP BY 1 ORDER BY 2 DESC"))

    # Source: 01_data_audit.ipynb, code cell 28
    # ---- 2f. Time zones: needed to convert local times to UTC (EAGLE-I is UTC) ---
    print(q("SELECT CZ_TIMEZONE, count(*) AS rows FROM storms_raw GROUP BY 1 ORDER BY 2 DESC"))

    # Source: 01_data_audit.ipynb, code cell 29
    # ---- 2g. Typed storm view ----------------------------------------------------
    # UTC = local time minus the offset in CZ_TIMEZONE (e.g. 'CST-6' -> +6 hours)
    con.execute("""
    CREATE OR REPLACE VIEW storms AS
    SELECT
        CAST(EVENT_ID AS BIGINT)                           AS event_id,
        CAST(EPISODE_ID AS BIGINT)                         AS episode_id,
        CAST(SOURCE_YEAR AS INT)                           AS year,
        EVENT_TYPE                                         AS event_type,
        CZ_TYPE                                            AS cz_type,
        lpad(STATE_FIPS, 2, '0')                           AS state_fips,
        lpad(CZ_FIPS, 3, '0')                              AS cz_fips,
        lpad(STATE_FIPS, 2, '0') || lpad(CZ_FIPS, 3, '0')  AS cz_code,
        STRPTIME(BEGIN_DATE_TIME, '%d-%b-%y %H:%M:%S')
            - to_hours(CAST(regexp_extract(CZ_TIMEZONE, '(-?[0-9]+)$', 1) AS INT)) AS begin_utc,
        STRPTIME(END_DATE_TIME, '%d-%b-%y %H:%M:%S')
            - to_hours(CAST(regexp_extract(CZ_TIMEZONE, '(-?[0-9]+)$', 1) AS INT)) AS end_utc,
        TRY_CAST(MAGNITUDE AS DOUBLE)                      AS magnitude,
        MAGNITUDE_TYPE                                     AS magnitude_type,
        TOR_F_SCALE                                        AS tor_f_scale,
        TRY_CAST(TOR_LENGTH AS DOUBLE)                     AS tor_length,
        TRY_CAST(TOR_WIDTH AS DOUBLE)                      AS tor_width,
        TRY_CAST(CATEGORY AS INT)                          AS hurricane_category,
        TRY_CAST(BEGIN_LAT AS DOUBLE)                      AS begin_lat,
        TRY_CAST(BEGIN_LON AS DOUBLE)                      AS begin_lon,
        TRY_CAST(regexp_extract(DAMAGE_PROPERTY, '([0-9.]+)', 1) AS DOUBLE)
            * CASE regexp_extract(DAMAGE_PROPERTY, '([KMB])$', 1)
                  WHEN 'K' THEN 1e3 WHEN 'M' THEN 1e6 WHEN 'B' THEN 1e9 ELSE 1 END AS damage_property_usd
    FROM storms_raw
    """)
    print(q("SELECT * FROM storms USING SAMPLE 5 ROWS"))

    # Source: 01_data_audit.ipynb, code cell 30
    # ---- 2h. Duration sanity check -----------------------------------------------
    print(q("""
    SELECT event_type,
           count(*) AS events,
           round(median(date_diff('minute', begin_utc, end_utc)) / 60, 1) AS median_hours,
           round(max(date_diff('minute', begin_utc, end_utc)) / 60, 1)    AS max_hours
    FROM storms GROUP BY event_type ORDER BY events DESC LIMIT 15
    """))

    # Source: 01_data_audit.ipynb, code cell 31
    # ---- 2i. County-coded (C) vs forecast-zone-coded (Z) vs marine events -------
    # Zone codes are NOT county FIPS. They must go through the NWS zone->county crosswalk.
    cz_mix = q("""
    SELECT year,
           count(*) AS events,
           round(100 * avg((cz_type = 'C')::INT), 1) AS pct_county,
           round(100 * avg((cz_type = 'Z')::INT), 1) AS pct_zone,
           round(100 * avg((event_type LIKE 'Marine%' OR event_type IN ('Waterspout'))::INT), 1) AS pct_marine
    FROM storms GROUP BY year ORDER BY year
    """)
    print(cz_mix)

    # Source: 01_data_audit.ipynb, code cell 32
    # ---- 2j. Event types and how each is located --------------------------------
    event_types = q("""
    SELECT event_type,
           count(*)                                  AS events,
           round(100 * avg((cz_type = 'C')::INT), 0) AS pct_county_coded,
           round(100 * avg((magnitude IS NOT NULL)::INT), 0) AS pct_with_magnitude,
           round(100 * avg((damage_property_usd > 0)::INT), 1) AS pct_with_damage
    FROM storms GROUP BY event_type ORDER BY events DESC
    """)
    event_types.to_csv(TABLES / "storm_event_types.csv", index=False)
    print(event_types)

    # Source: 01_data_audit.ipynb, code cell 33
    # ---- 2k. County-coded events: valid FIPS? ------------------------------------
    print(q("""
    SELECT
        count(*)                               AS county_coded_events,
        sum((c.fips IS NULL)::INT)             AS fips_not_in_census,
        round(100 * avg((c.fips IS NULL)::INT), 2) AS pct_not_in_census
    FROM storms s LEFT JOIN census c ON c.fips = s.cz_code
    WHERE s.cz_type = 'C'
    """))

    # Source: 01_data_audit.ipynb, code cell 34
    # ---- 2l. Zone-coded events: can the crosswalk map them to counties? ---------
    # Use the crosswalk version that was in effect on the event's date.
    zone_map = q("""
    WITH z AS (
        SELECT s.event_id, s.year, s.event_type, r.abbr || s.cz_fips AS state_zone, CAST(s.begin_utc AS DATE) AS d
        FROM storms s JOIN regions r USING (state_fips)
        WHERE s.cz_type = 'Z'
    ),
    matched AS (
        SELECT z.event_id, count(DISTINCT x.county_fips) AS n_counties
        FROM z JOIN crosswalk x
          ON x.state_zone = z.state_zone
         AND z.d >= CAST(x.effective_date AS DATE)
         AND (x.next_effective_date IS NULL OR z.d < CAST(x.next_effective_date AS DATE))
        GROUP BY z.event_id
    )
    SELECT z.year,
           count(*)                                      AS zone_events,
           round(100 * avg((m.event_id IS NOT NULL)::INT), 1) AS pct_mapped,
           round(avg(m.n_counties), 2)                   AS avg_counties_per_zone_event
    FROM z LEFT JOIN matched m USING (event_id)
    GROUP BY z.year ORDER BY z.year
    """)
    print(zone_map)

    # Source: 01_data_audit.ipynb, code cell 35
    # ---- 2m. Which zone events fail to map? -------------------------------------
    print(q("""
    WITH z AS (
        SELECT s.event_id, s.event_type, s.state_fips, r.abbr || s.cz_fips AS state_zone, CAST(s.begin_utc AS DATE) AS d
        FROM storms s LEFT JOIN regions r USING (state_fips)
        WHERE s.cz_type = 'Z'
    )
    SELECT event_type, count(*) AS unmapped
    FROM z
    WHERE NOT EXISTS (
        SELECT 1 FROM crosswalk x
        WHERE x.state_zone = z.state_zone
          AND z.d >= CAST(x.effective_date AS DATE)
          AND (x.next_effective_date IS NULL OR z.d < CAST(x.next_effective_date AS DATE)))
    GROUP BY event_type ORDER BY unmapped DESC LIMIT 10
    """))

    # Source: 01_data_audit.ipynb, code cell 36
    # =============================================================================
    # 3. SUPPORTING FILES: MCC, Census population, DQI, coverage history
    # =============================================================================
    # ---- 3a. MCC customers vs Census population (is population a usable proxy?) -
    mcc_pop = mcc.merge(census, on="fips", how="outer", indicator=True)
    print(mcc_pop["_merge"].value_counts())
    both = mcc_pop[mcc_pop["_merge"] == "both"]
    print("correlation MCC vs population:", round(both["mcc_customers"].corr(both["pop2023"]), 3))
    print("median customers per person:", round((both["mcc_customers"] / both["pop2023"]).median(), 3))
    print(mcc_pop[mcc_pop["_merge"] != "both"].head(15))

    # Source: 01_data_audit.ipynb, code cell 37
    # ---- 3b. DQI (Data Quality Index) by FEMA region and year -------------------
    dqi = pd.read_csv(DQI_CSV)
    dqi_wide = dqi.pivot(index="fema", columns="year", values="DQI").round(1)
    dqi_wide.to_csv(TABLES / "dqi_by_fema_region.csv")
    print(dqi_wide)

    # Source: 01_data_audit.ipynb, code cell 38
    # ---- 3c. Coverage history: % of customers EAGLE-I tracks, by state ----------
    cov = pd.read_csv(COVERAGE_CSV)
    cov["year"] = pd.to_datetime(cov["year"], format="%m/%d/%y").dt.year
    cov["fema_region"] = cov["state"].map(abbr_to_region)
    cov_region = cov.groupby(["fema_region", "year"])["max_pct_covered"].mean().unstack().round(2)
    print(cov_region)

    # Source: 01_data_audit.ipynb, code cell 39
    # ---- 3d. States with the lowest coverage (latest year available) ------------
    print(cov[cov["year"] == cov["year"].max()].sort_values("max_pct_covered").head(10))

    # Source: 01_data_audit.ipynb, code cell 40
    # =============================================================================
    # 4. REGION COMPARISON: Southeast (FEMA IV) vs the rest
    # =============================================================================
    county_region = census.assign(state_fips=census["fips"].str[:2]).merge(regions, on="state_fips")
    con.register("county_region", county_region)

    region_eaglei = q("""
    WITH snaps AS (SELECT count(DISTINCT ts) AS n FROM eaglei WHERE year = 2023),
    c23 AS (SELECT fips, count(*) AS n_rows FROM eaglei WHERE year = 2023 GROUP BY fips)
    SELECT cr.fema_region,
           count(*)                                         AS census_counties,
           count(c23.fips)                                  AS counties_in_eaglei_2023,
           round(100 * count(c23.fips) / count(*), 1)       AS pct_counties_covered,
           round(100 * median(c23.n_rows / (SELECT n FROM snaps)), 1) AS median_pct_snapshots_reported_2023
    FROM county_region cr LEFT JOIN c23 USING (fips)
    GROUP BY cr.fema_region ORDER BY cr.fema_region
    """)
    region_storms = q("""
    SELECT r.fema_region,
           count(*)                                  AS storm_events,
           round(100 * avg((s.cz_type = 'C')::INT), 1) AS pct_county_coded
    FROM storms s JOIN regions r USING (state_fips)
    GROUP BY r.fema_region ORDER BY r.fema_region
    """)
    region_cmp = (region_eaglei
                  .merge(region_storms, on="fema_region")
                  .merge(dqi_wide[[2022]].rename(columns={2022: "dqi_2022"}), left_on="fema_region", right_index=True)
                  .merge(cov_region[[2022]].rename(columns={2022: "pct_customers_tracked_2022"}),
                         left_on="fema_region", right_index=True))
    region_cmp["is_southeast"] = region_cmp["fema_region"] == 4
    region_cmp.to_csv(TABLES / "region_comparison.csv", index=False)
    print(region_cmp)

    # Source: 01_data_audit.ipynb, code cell 41
    # ---- 4b. Share of all records that fall in the Southeast ---------------------
    print(q("""
    SELECT r.is_southeast, count(*) AS eaglei_rows, round(100 * count(*) / sum(count(*)) OVER (), 1) AS pct_rows
    FROM eaglei e JOIN regions r ON r.state_fips = left(e.fips, 2)
    GROUP BY r.is_southeast
    """))

    # Source: 01_data_audit.ipynb, code cell 42
    # =============================================================================
    # 5. SIGNAL CHECK: do storms line up with outages? (2022-2025, zeros are stored)
    # =============================================================================
    # For each county-coded storm, take the worst hour of outages in the 24 h after it starts.
    # Baseline = the same county and hours exactly one week earlier.
    con.execute("""
    CREATE OR REPLACE TEMP TABLE eaglei_hourly AS
    SELECT fips, date_trunc('hour', ts) AS hr, max(customers_out) AS max_out
    FROM eaglei WHERE year >= 2022 AND customers_out IS NOT NULL
    GROUP BY fips, hr
    """)

    signal = q("""
    WITH ev AS (
        SELECT event_id, event_type, cz_code AS fips, date_trunc('hour', begin_utc) AS start_hr
        FROM storms WHERE cz_type = 'C' AND year >= 2022
    ),
    windows AS (
        SELECT ev.*, unnest(range(0, 24)) AS h FROM ev
    ),
    after AS (
        SELECT w.event_id, any_value(w.event_type) AS event_type, any_value(w.fips) AS fips,
               max(a.max_out) AS peak_after, count(a.hr) AS hours_with_data,
               max(b.max_out) AS peak_week_before
        FROM windows w
        LEFT JOIN eaglei_hourly a ON a.fips = w.fips AND a.hr = w.start_hr + to_hours(w.h)
        LEFT JOIN eaglei_hourly b ON b.fips = w.fips AND b.hr = w.start_hr + to_hours(w.h) - INTERVAL 7 DAY
        GROUP BY w.event_id
    )
    SELECT a.*, m.mcc_customers,
           a.peak_after / m.mcc_customers       AS peak_pct_after,
           a.peak_week_before / m.mcc_customers AS peak_pct_week_before
    FROM after a LEFT JOIN mcc m USING (fips)
    """)
    print("events:", len(signal), "| with no EAGLE-I data in window:", (signal["hours_with_data"] == 0).sum())

    signal_summary = (signal[signal["hours_with_data"] > 0]
                      .groupby("event_type")
                      .agg(events=("event_id", "count"),
                           pct_peak_ge_1pct=("peak_pct_after", lambda s: round(100 * (s >= 0.01).mean(), 1)),
                           pct_baseline_ge_1pct=("peak_pct_week_before", lambda s: round(100 * (s >= 0.01).mean(), 1)),
                           median_peak_pct=("peak_pct_after", lambda s: round(100 * s.median(), 2)))
                      .sort_values("events", ascending=False))
    signal_summary.to_csv(TABLES / "storm_outage_signal.csv")
    print(signal_summary)

    # Source: 01_data_audit.ipynb, code cell 43
    # ---- 5b. Signal chart --------------------------------------------------------
    top = signal_summary.head(10).sort_values("pct_peak_ge_1pct")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top.index, top["pct_baseline_ge_1pct"], color="lightgray", label="same hours, 1 week earlier")
    ax.barh(top.index, top["pct_peak_ge_1pct"], color="tab:blue", alpha=0.7, label="24 h after storm start")
    ax.set_xlabel("% of events with >= 1% of county customers out")
    ax.set_title("Storms vs outages (county-coded events, 2022-2025)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig_dir = ROOT / "reports" / "figures" / "audit"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "03_storm_outage_signal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Source: 01_data_audit.ipynb, code cell 44
    # =============================================================================
    # 6. TRAIN / VALIDATION / TEST READINESS (train 2014-2023, val 2024, test 2025)
    # =============================================================================
    split = (coverage[["year", "counties_reported"]]
             .merge(grid[["year", "pct_present"]], on="year")
             .merge(eaglei_nulls[["year", "pct_null_customers_out", "pct_null_total_customers"]], on="year")
             .merge(cz_mix[["year", "events", "pct_county"]].rename(columns={"events": "storm_events"}), on="year")
             .merge(zone_map[["year", "pct_mapped"]].rename(columns={"pct_mapped": "pct_zone_events_mapped"}), on="year"))
    split["zeros_stored"] = split["year"] >= 2022
    split["split"] = pd.cut(split["year"], [2013, 2023, 2024, 2025], labels=["train", "validation", "test"])
    split.to_csv(TABLES / "split_readiness.csv", index=False)
    print(split)

    # Source: 01_data_audit.ipynb, code cell 45
    # =============================================================================
    # 7. COLUMN RELEVANCE TAGS (for the modeling goal)
    # =============================================================================
    # leakage = only known after the storm has happened, so it cannot be a model input
    RELEVANCE = {
        # EAGLE-I
        "fips_code": "location", "county": "location", "state": "location",
        "customers_out": "outage target", "total_customers": "outage target (denominator)",
        "run_start_time": "time", "SOURCE_YEAR": "time",
        # Storm Events: time / id
        "BEGIN_YEARMONTH": "time", "BEGIN_DAY": "time", "BEGIN_TIME": "time", "END_YEARMONTH": "time",
        "END_DAY": "time", "END_TIME": "time", "BEGIN_DATE_TIME": "time", "END_DATE_TIME": "time",
        "CZ_TIMEZONE": "time", "YEAR": "time", "MONTH_NAME": "time",
        "EPISODE_ID": "id", "EVENT_ID": "id", "DATA_SOURCE": "id",
        # Storm Events: location
        "STATE": "location", "STATE_FIPS": "location", "CZ_TYPE": "location", "CZ_FIPS": "location",
        "CZ_NAME": "location", "WFO": "location", "BEGIN_LAT": "location", "BEGIN_LON": "location",
        "END_LAT": "location", "END_LON": "location", "BEGIN_RANGE": "location", "BEGIN_AZIMUTH": "location",
        "BEGIN_LOCATION": "location", "END_RANGE": "location", "END_AZIMUTH": "location", "END_LOCATION": "location",
        "TOR_OTHER_WFO": "location", "TOR_OTHER_CZ_STATE": "location", "TOR_OTHER_CZ_FIPS": "location",
        "TOR_OTHER_CZ_NAME": "location",
        # Storm Events: storm description. These are OBSERVED values; at prediction time
        # they must come from a forecast/alert instead (observed = stand-in for forecast).
        "EVENT_TYPE": "storm (observed)", "MAGNITUDE": "storm (observed)", "MAGNITUDE_TYPE": "storm (observed)",
        "CATEGORY": "storm (observed)", "FLOOD_CAUSE": "storm (observed)",
        # Storm Events: measured after the fact
        "TOR_F_SCALE": "leakage (rated after survey)", "TOR_LENGTH": "leakage (measured after)",
        "TOR_WIDTH": "leakage (measured after)", "SOURCE": "leakage (report source)",
        "INJURIES_DIRECT": "leakage", "INJURIES_INDIRECT": "leakage", "DEATHS_DIRECT": "leakage",
        "DEATHS_INDIRECT": "leakage", "DAMAGE_PROPERTY": "leakage", "DAMAGE_CROPS": "leakage",
        "EPISODE_NARRATIVE": "leakage (free text)", "EVENT_NARRATIVE": "leakage (free text)",
    }
    tags = pd.DataFrame({"column": list(RELEVANCE), "tag": list(RELEVANCE.values())})
    tags = tags.merge(storm_nulls[["pct_null_all_rows", "pct_null_where_applies", "drop_candidate"]],
                      left_on="column", right_index=True, how="left")
    tags.to_csv(TABLES / "column_tags.csv", index=False)
    print("untagged storm columns:", sorted(set(storm_cols) - set(RELEVANCE)))
    print(tags)

    # Source: 01_data_audit.ipynb, code cell 46
    con.close()


if __name__ == "__main__":
    main()
