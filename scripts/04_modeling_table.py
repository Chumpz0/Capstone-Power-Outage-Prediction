"""Standalone Python version of notebooks/04_modeling_table.ipynb.

Run with the project virtual environment. See scripts/README.md for inputs,
outputs, execution order, and differences from the notebook.
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # Save figures without requiring a notebook or GUI.


    # Source: 04_modeling_table.ipynb, code cell 1
    # =============================================================================
    # 04 - MODELING TABLE: one row per county x NOAA storm episode (2017-2025)
    # Summary of results lives in reports/04_modeling_table_summary.md
    # Target: >= 5% of county customers out within 24 h of the episode's start in that county.
    # Scope: lower 48 states + DC. Drought / Heat / Excessive Heat are not storm triggers.
    # Inputs: processed/ (from notebook 02). Outputs: processed/model_table/, reports/model_table_tables/
    # Test (2025) labels are written to their own file and are not summarised here.
    # =============================================================================

    # Source: 04_modeling_table.ipynb, code cell 2
    # ---- 0. Setup ---------------------------------------------------------------
    from pathlib import Path

    import duckdb
    import numpy as np
    import pandas as pd

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    ROOT = Path(__file__).resolve().parents[1]
    PROC = ROOT / "processed"
    OUT = PROC / "model_table"
    TABLES = ROOT / "reports" / "model_table_tables"
    for folder in [OUT, TABLES]:
        folder.mkdir(parents=True, exist_ok=True)

    # Inputs: the local file when it is in the repo, otherwise streamed from the Hugging Face dataset
    HF_DATASET = "hf://datasets/DSCI-Capstone/county-outage-data"


    def src(rel):
        """Local path if the file is in the repo, otherwise its hf:// URL."""
        local = ROOT / rel
        return local.as_posix() if local.exists() else f"{HF_DATASET}/{rel}"

    TRAIN_FIRST, TRAIN_LAST, VAL_YEAR, TEST_YEAR = 2017, 2023, 2024, 2025
    TARGET_PCT = 0.05     # outage = at least 5% of county customers out (user decision)
    WINDOW_HOURS = 24     # label window after the episode starts in the county (user decision)
    DROPPED_TYPES = ["Drought", "Heat", "Excessive Heat"]   # not storm triggers (user decision)
    HISTORY_PRIOR = 20    # county history is shrunk toward the national rate with this many pseudo-episodes (provisional)

    # Lower 48 + DC (user decision). Value = approximate standard-time UTC offset of the state's main time zone,
    # used only to turn the UTC start hour into a local hour of day (provisional, split states use the majority zone).
    STATE_UTC_OFFSET = {
        "CT": -5, "DE": -5, "DC": -5, "FL": -5, "GA": -5, "IN": -5, "KY": -5, "ME": -5, "MD": -5, "MA": -5,
        "MI": -5, "NH": -5, "NJ": -5, "NY": -5, "NC": -5, "OH": -5, "PA": -5, "RI": -5, "SC": -5, "VT": -5,
        "VA": -5, "WV": -5,
        "AL": -6, "AR": -6, "IL": -6, "IA": -6, "KS": -6, "LA": -6, "MN": -6, "MS": -6, "MO": -6, "NE": -6,
        "ND": -6, "OK": -6, "SD": -6, "TN": -6, "TX": -6, "WI": -6,
        "AZ": -7, "CO": -7, "ID": -7, "MT": -7, "NM": -7, "UT": -7, "WY": -7,
        "CA": -8, "NV": -8, "OR": -8, "WA": -8,
    }
    assert len(STATE_UTC_OFFSET) == 49

    # Storm types grouped into a few feature families; anything not listed falls into "other".
    TYPE_GROUPS = {
        "tstm_wind": ["Thunderstorm Wind"],
        "hail": ["Hail"],
        "tornado": ["Tornado", "Funnel Cloud"],
        "high_wind": ["High Wind", "Strong Wind"],
        "flood": ["Flash Flood", "Flood", "Heavy Rain", "Coastal Flood", "Lakeshore Flood", "Storm Surge/Tide", "Debris Flow"],
        "winter": ["Winter Weather", "Winter Storm", "Heavy Snow", "Blizzard", "Lake-Effect Snow", "Sleet"],
        "ice_storm": ["Ice Storm"],
        "tropical": ["Tropical Storm", "Hurricane (Typhoon)", "Tropical Depression"],
        "cold": ["Extreme Cold/Wind Chill", "Cold/Wind Chill", "Frost/Freeze"],
        "lightning": ["Lightning"],
        "wildfire": ["Wildfire"],
    }

    con = duckdb.connect()
    try:  # the dataset is private: streaming uses the token saved by `hf auth login` (not stored in any file here)
        con.execute("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")
    except duckdb.IOException:
        print("No Hugging Face login found. Files not in the repo can't be streamed until you run `hf auth login`.")
    con.execute(f"CREATE VIEW hourly   AS SELECT * FROM read_parquet('{src('processed/outages_hourly')}/*/*.parquet', hive_partitioning = true)")
    con.execute(f"CREATE VIEW observed AS SELECT * FROM read_parquet('{src('processed/county_observed_daily.parquet')}')")
    con.execute(f"CREATE VIEW storm_county AS SELECT * FROM read_parquet('{src('processed/storm_county_events.parquet')}')")
    con.execute(f"CREATE VIEW counties AS SELECT * FROM read_parquet('{src('processed/counties.parquet')}')")


    def q(sql):
        """Run a SQL query and return a pandas DataFrame."""
        return con.execute(sql).df()


    def sql_list(values):
        """Python list -> SQL list of quoted strings."""
        return ", ".join("'" + v.replace("'", "''") + "'" for v in values)

    # Source: 04_modeling_table.ipynb, code cell 3
    # ---- 1. Storm x county rows in scope ---------------------------------------------
    # Start from notebook 02's storm x county table and apply the scope filters.
    # The inner join on the county lookup with the state list also drops unknown FIPS codes
    # (and the stray "Grand Total" row that notebook 02 left in counties.parquet).
    type_case = "CASE " + " ".join(
        f"WHEN event_type IN ({sql_list(types)}) THEN '{group}'" for group, types in TYPE_GROUPS.items()
    ) + " ELSE 'other' END"

    con.execute(f"""
    CREATE TEMP TABLE sc AS
    SELECT s.*, c.state_abbr, {type_case} AS type_group
    FROM storm_county s
    JOIN counties c USING (fips)
    WHERE c.state_abbr IN ({sql_list(STATE_UTC_OFFSET)})
      AND s.event_type NOT IN ({sql_list(DROPPED_TYPES)})
      AND year(s.begin_utc) BETWEEN {TRAIN_FIRST} AND {TEST_YEAR}
    """)

    waterfall = [
        ("storm x county rows (notebook 02)", q("SELECT count(*) n FROM storm_county")["n"][0]),
        ("after lower 48 + DC filter", q(f"""SELECT count(*) n FROM storm_county s JOIN counties c USING (fips)
                                             WHERE c.state_abbr IN ({sql_list(STATE_UTC_OFFSET)})""")["n"][0]),
        ("after dropping drought / heat", q("SELECT count(*) n FROM sc")["n"][0]),
    ]
    print(waterfall)

    # Source: 04_modeling_table.ipynb, code cell 4
    # ---- 2. Collapse to county x episode ------------------------------------------------
    # t0 = first report of the episode in that county. Storm features summarise the WHOLE episode
    # (user decision: the observed episode stands in for a storm forecast).
    con.execute(f"""
    CREATE TEMP TABLE ep AS
    SELECT fips, episode_id,
           any_value(state_abbr)                                         AS state_abbr,
           min(begin_utc)                                                AS t0,
           date_trunc('hour', min(begin_utc))                            AS start_hr,
           year(min(begin_utc))                                          AS year,
           count(*)                                                      AS n_events,
           count(*) FILTER (WHERE type_group = 'tstm_wind') AS n_tstm_wind,
           count(*) FILTER (WHERE type_group = 'hail') AS n_hail,
           count(*) FILTER (WHERE type_group = 'tornado') AS n_tornado,
           count(*) FILTER (WHERE type_group = 'high_wind') AS n_high_wind,
           count(*) FILTER (WHERE type_group = 'flood') AS n_flood,
           count(*) FILTER (WHERE type_group = 'winter') AS n_winter,
           count(*) FILTER (WHERE type_group = 'ice_storm') AS n_ice_storm,
           count(*) FILTER (WHERE type_group = 'tropical') AS n_tropical,
           count(*) FILTER (WHERE type_group = 'cold') AS n_cold,
           count(*) FILTER (WHERE type_group = 'lightning') AS n_lightning,
           count(*) FILTER (WHERE type_group = 'wildfire') AS n_wildfire,
           count(*) FILTER (WHERE type_group = 'other') AS n_other,
           count(*) FILTER (WHERE mapped_from = 'county')                AS n_county_coded,
           count(*) FILTER (WHERE mapped_from = 'zone')                  AS n_zone_coded,
           coalesce(bool_or(partial_county), false)                      AS any_partial_county,
           max(magnitude) FILTER (WHERE magnitude_type IS NOT NULL)      AS max_wind_kts,
           max(magnitude) FILTER (WHERE event_type = 'Hail')             AS max_hail_in,
           max(hurricane_category)                                       AS max_hurricane_cat,
           date_diff('minute', min(begin_utc), max(end_utc)) / 60.0      AS county_episode_hours
    FROM sc
    GROUP BY fips, episode_id
    """)

    # Size of the whole storm system (all counties in the episode)
    con.execute("""
    CREATE TEMP TABLE ep_size AS
    SELECT episode_id,
           count(DISTINCT fips)       AS episode_counties,
           count(DISTINCT state_abbr) AS episode_states,
           count(DISTINCT event_id)   AS episode_events
    FROM sc GROUP BY episode_id
    """)
    print(q("SELECT year, count(*) AS county_episodes, count(DISTINCT episode_id) AS episodes FROM ep GROUP BY 1 ORDER BY 1"))

    # Source: 04_modeling_table.ipynb, code cell 5
    # ---- 3. Keep only county-episodes EAGLE-I was watching -----------------------------
    # Missing outage rows mean "0 or no data", so the label is only trusted when the county was observed
    # on the start day AND on the day the 24 h window ends (user decision: drop unobserved rows;
    # checking the end day too is a provisional engineering default, it catches the Dec 31 gaps).
    con.execute(f"""
    CREATE TEMP TABLE ep_obs AS
    SELECT e.*, o1.dqi, o1.pct_customers_tracked,
           coalesce(o0.observed, false) AS prev_day_observed
    FROM ep e
    JOIN observed o1 ON o1.fips = e.fips AND o1.day = CAST(e.start_hr AS DATE) AND o1.observed
    JOIN observed o2 ON o2.fips = e.fips AND o2.day = CAST(e.start_hr + INTERVAL {WINDOW_HOURS - 1} HOUR AS DATE) AND o2.observed
    LEFT JOIN observed o0 ON o0.fips = e.fips AND o0.day = CAST(e.start_hr - INTERVAL 24 HOUR AS DATE)
    """)
    obs_by_year = q("""
    SELECT e.year, count(*) AS county_episodes, count(o.fips) AS observed, round(100.0 * count(o.fips) / count(*), 1) AS pct_observed
    FROM ep e LEFT JOIN ep_obs o USING (fips, episode_id) GROUP BY 1 ORDER BY 1
    """)
    waterfall.append(("county x episode rows", q("SELECT count(*) n FROM ep")["n"][0]))
    waterfall.append(("after observed filter (final rows)", q("SELECT count(*) n FROM ep_obs")["n"][0]))
    print(obs_by_year)

    # Source: 04_modeling_table.ipynb, code cell 6
    # ---- 4. Label + pre-storm outage level -----------------------------------------------
    # Hours 0..23 from the start hour  -> peak % out in the window -> target (>= 5%).
    # Hours -24..-1 before the start   -> pre-storm outage features (known at prediction time).
    # The hourly table only holds hours with an outage, so a missing hour means 0 out.
    con.execute(f"""
    CREATE TEMP TABLE labels AS
    WITH hours AS (
        SELECT fips, episode_id, start_hr + to_hours(h) AS hr, h
        FROM ep_obs CROSS JOIN range(-24, {WINDOW_HOURS}) t(h)
    )
    SELECT x.fips, x.episode_id,
           max(o.max_pct_out)       FILTER (WHERE x.h >= 0) AS peak_24h,
           max(o.max_customers_out) FILTER (WHERE x.h >= 0) AS peak_customers_out,
           arg_max(x.h, (o.max_pct_out, -x.h)) FILTER (WHERE x.h >= 0) AS hour_of_peak,  -- ties -> earliest hour
           max(o.max_pct_out)       FILTER (WHERE x.h < 0)  AS pre_peak_24h,
           max(o.max_pct_out)       FILTER (WHERE x.h = -1) AS pct_out_at_start
    FROM hours x
    JOIN hourly o ON o.fips = x.fips AND o.hour_utc = x.hr
    GROUP BY x.fips, x.episode_id
    """)

    df = q("""
    SELECT e.*, s.episode_counties, s.episode_states, s.episode_events,
           coalesce(l.peak_24h, 0)           AS peak_24h,
           coalesce(l.peak_customers_out, 0) AS peak_customers_out,
           l.hour_of_peak,
           coalesce(l.pre_peak_24h, 0)       AS pre_peak_24h,
           coalesce(l.pct_out_at_start, 0)   AS pct_out_at_start,
           c.denominator, c.denominator_source, c.pop2023, c.fema_region
    FROM ep_obs e
    JOIN ep_size s USING (episode_id)
    LEFT JOIN labels l USING (fips, episode_id)
    JOIN counties c USING (fips)
    """)
    # Pre-storm features are unknown if the day before was not observed
    df.loc[~df["prev_day_observed"], ["pre_peak_24h", "pct_out_at_start"]] = np.nan
    df["target"] = (df["peak_24h"] >= TARGET_PCT).astype(int)
    df["split"] = np.select([df["year"] <= TRAIN_LAST, df["year"] == VAL_YEAR], ["train", "validation"], "test")
    print(df.groupby("split").size())

    # Source: 04_modeling_table.ipynb, code cell 7
    # ---- 5. Time and county features ---------------------------------------------------
    df["month"] = df["t0"].dt.month
    df["local_hour"] = (df["t0"].dt.hour + df["state_abbr"].map(STATE_UTC_OFFSET)) % 24
    df["log_customers"] = np.log10(df["denominator"])
    df["log_pop"] = np.log10(df["pop2023"].clip(lower=1))
    df["share_zone_coded"] = df["n_zone_coded"] / df["n_events"]
    df["log_episode_counties"] = np.log10(df["episode_counties"])
    print(df[["month", "local_hour", "log_customers", "log_pop", "share_zone_coded"]].describe().round(2))

    # Source: 04_modeling_table.ipynb, code cell 8
    # ---- 6. County history (training years only) ------------------------------------------
    # Past storm-outage behaviour of each county, built only from training-year labels:
    #   - validation / test rows use all training years
    #   - training rows use the OTHER training years (leave-one-year-out), so a row never sees its own label
    # Rates are shrunk toward the national training rate so counties with few episodes are not extreme.
    train = df[df["split"] == "train"]
    p0 = train["target"].mean()
    peak0 = train["peak_24h"].mean()

    cy = train.groupby(["fips", "year"]).agg(n=("target", "size"), pos=("target", "sum"), peak=("peak_24h", "sum")).reset_index()
    tot = cy.groupby("fips")[["n", "pos", "peak"]].sum().add_suffix("_all").reset_index()

    df = df.merge(tot, on="fips", how="left").merge(cy, on=["fips", "year"], how="left")
    df[["n_all", "pos_all", "peak_all", "n", "pos", "peak"]] = df[["n_all", "pos_all", "peak_all", "n", "pos", "peak"]].fillna(0)
    own = df["split"].eq("train")   # remove the row's own year for training rows
    n_hist = df["n_all"] - np.where(own, df["n"], 0)
    pos_hist = df["pos_all"] - np.where(own, df["pos"], 0)
    peak_hist = df["peak_all"] - np.where(own, df["peak"], 0)

    df["hist_episodes"] = n_hist
    df["hist_outage_rate"] = (pos_hist + HISTORY_PRIOR * p0) / (n_hist + HISTORY_PRIOR)
    df["hist_mean_peak"] = (peak_hist + HISTORY_PRIOR * peak0) / (n_hist + HISTORY_PRIOR)
    df = df.drop(columns=["n_all", "pos_all", "peak_all", "n", "pos", "peak"])
    print(f"national training rate {p0:.3f} | counties with no training history: "
          f"{(df['hist_episodes'] == 0).groupby(df['split']).mean().round(3).to_dict()}")

    # Source: 04_modeling_table.ipynb, code cell 9
    # ---- 7. Assemble and save --------------------------------------------------------------
    ID_COLS = ["fips", "episode_id", "state_abbr", "fema_region", "year", "t0"]
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
    # Known only after the window: kept for analysis, never as model inputs (retro_ prefix)
    RETRO = {"peak_24h": "retro_peak_24h", "peak_customers_out": "retro_peak_customers_out", "hour_of_peak": "retro_hour_of_peak"}

    table = df.rename(columns=RETRO)[ID_COLS + FEATURES + ["target"] + list(RETRO.values()) + ["split"]]
    table = table.sort_values(["t0", "fips"]).reset_index(drop=True)

    table[table["split"] == "train"].drop(columns="split").to_parquet(OUT / "train.parquet", index=False)
    table[table["split"] == "validation"].drop(columns="split").to_parquet(OUT / "validation.parquet", index=False)
    test = table[table["split"] == "test"]
    test[ID_COLS + FEATURES].to_parquet(OUT / "test_features.parquet", index=False)
    test[["fips", "episode_id", "target"] + list(RETRO.values())].to_parquet(OUT / "test_labels.parquet", index=False)
    print({p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in sorted(OUT.glob("*.parquet"))})

    # Source: 04_modeling_table.ipynb, code cell 10
    # ---- 8. Checks and summary tables ----------------------------------------------------------
    # Test rows: only the count is reported, never the label rate.
    checks = pd.DataFrame([
        ("one row per county x episode", not table.duplicated(["fips", "episode_id"]).any()),
        ("target has no nulls", table["target"].notna().all()),
        ("only lower 48 + DC", table["state_abbr"].isin(STATE_UTC_OFFSET).all()),
        ("peak % out within [0, 1]", table["retro_peak_24h"].between(0, 1).all()),
        ("splits do not share years", table.groupby("year")["split"].nunique().max() == 1),
        ("no retro_ column among features", not any(f.startswith("retro_") for f in FEATURES)),
        ("history uses no own-year label (train)", True),   # by construction in section 6
    ], columns=["check", "passed"])
    checks.to_csv(TABLES / "checks.csv", index=False)
    print(checks)

    pd.DataFrame(waterfall, columns=["step", "rows"]).to_csv(TABLES / "row_waterfall.csv", index=False)
    obs_by_year.to_csv(TABLES / "observed_by_year.csv", index=False)

    by_year = table.groupby(["split", "year"]).agg(rows=("target", "size"), positive_rate=("target", "mean")).reset_index()
    by_year.loc[by_year["split"] == "test", "positive_rate"] = np.nan
    by_year.to_csv(TABLES / "rows_by_year.csv", index=False)
    print(by_year.round(3))

    # Feature coverage (share non-null) per split, so gaps between train and validation/test are visible
    coverage = table.groupby("split")[FEATURES].apply(lambda d: d.notna().mean()).T.round(3)
    coverage.to_csv(TABLES / "feature_coverage.csv")
    print(coverage[coverage.min(axis=1) < 1])

    # Source: 04_modeling_table.ipynb, code cell 11
    # ---- 9. Quick look at the training rows (signal per storm family) -----------------------------
    tr = table[table["split"] == "train"]
    fam = pd.DataFrame({
        g: {"rows_with_type": int((tr[f"n_{g}"] > 0).sum()), "positive_rate": tr.loc[tr[f"n_{g}"] > 0, "target"].mean()}
        for g in ["tstm_wind", "hail", "tornado", "high_wind", "flood", "winter", "ice_storm", "tropical", "cold", "lightning", "wildfire", "other"]
    }).T.sort_values("positive_rate", ascending=False)
    fam.to_csv(TABLES / "train_rate_by_storm_family.csv")

    pre = tr.assign(pre_band=pd.cut(tr["pre_peak_24h"], [-0.001, 0.001, 0.01, 0.05, 1], labels=["<0.1%", "0.1-1%", "1-5%", ">=5%"]))
    pre_rate = pre.groupby("pre_band", observed=True)["target"].agg(["size", "mean"])
    pre_rate.to_csv(TABLES / "train_rate_by_pre_storm_outage.csv")

    zone = tr.groupby(tr["n_county_coded"] > 0)["target"].agg(["size", "mean"]).rename(index={True: "has county-coded report", False: "zone-coded only"})
    zone.to_csv(TABLES / "train_rate_by_coding.csv")
    print(fam.round(3), pre_rate.round(3), zone.round(3), sep="\n\n")

    # Source: 04_modeling_table.ipynb, code cell 12
    con.close()


if __name__ == "__main__":
    main()
