"""County-episode features and hourly labels using the existing retrospective definition.

Histories exclude county sums for the row's year but use full-training smoothing
priors and other (possibly later) training years. They are not as-of-time histories.
"""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np

from .config import (
    ProjectPaths,
    TRAIN_FIRST,
    TRAIN_LAST,
    VAL_YEAR,
    TEST_YEAR,
    TARGET_PCT,
    WINDOW_HOURS,
    HISTORY_PRIOR,
    STATE_UTC_OFFSET,
    TYPE_GROUPS,
    DROPPED_TYPES,
    staged_directory,
    require_checks,
    export_results,
)
from .dataset import connection, attach_prepared, query


def sql_list(values):
    """Quote domain string literals for the fixed project taxonomy."""
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def scope_storms(con):
    """Register in-scope lower-48/DC storm-county reports on the caller connection."""
    # Apply model scope to the prepared storm-county table.
    # The inner join on the county lookup with the state list also drops unknown FIPS codes

    type_case = (
        "CASE "
        + " ".join(
            f"WHEN event_type IN ({sql_list(types)}) THEN '{group}'"
            for group, types in TYPE_GROUPS.items()
        )
        + " ELSE 'other' END"
    )

    con.execute(f"""
    CREATE TEMP TABLE sc AS
    SELECT s.*, c.state_abbr, {type_case} AS type_group
    FROM storm_county s
    JOIN counties c USING (fips)
    WHERE c.state_abbr IN ({sql_list(STATE_UTC_OFFSET)})
      AND s.event_type NOT IN ({sql_list(DROPPED_TYPES)})
      AND year(s.begin_utc) BETWEEN {TRAIN_FIRST} AND {TEST_YEAR}
    """)


def aggregate_episodes(con):
    """Register whole-episode storm counts and size; these are retrospective proxies."""
    # t0 = first report of the episode in that county. Storm features summarise the WHOLE episode
    # (user decision: the observed episode stands in for a storm forecast).
    con.execute("""
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


def select_observed_episodes(con):
    """Require coverage on the UTC dates containing label hours 0 and 23."""
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


def label_episodes(con):
    """Return labels in [floor_hour(t0), floor_hour(t0)+24h); preceding hours supply pre-storm levels."""
    q = partial(query, con)
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
    df["split"] = np.select(
        [df["year"] <= TRAIN_LAST, df["year"] == VAL_YEAR],
        ["train", "validation"],
        "test",
    )
    return df


def add_county_time_features(df):
    """Add approximate local standard-time hour and county/episode log transforms."""
    df = df.copy()
    df["month"] = df["t0"].dt.month
    df["local_hour"] = (df["t0"].dt.hour + df["state_abbr"].map(STATE_UTC_OFFSET)) % 24
    df["log_customers"] = np.log10(df["denominator"])
    df["log_pop"] = np.log10(df["pop2023"].clip(lower=1))
    df["share_zone_coded"] = df["n_zone_coded"] / df["n_events"]
    df["log_episode_counties"] = np.log10(df["episode_counties"])
    return df


def add_history(df):
    """Preserve leave-one-year-out county sums with full-training priors; returns a new frame."""
    df = df.copy()
    train = df[df["split"] == "train"]
    p0 = train["target"].mean()
    peak0 = train["peak_24h"].mean()

    cy = (
        train.groupby(["fips", "year"])
        .agg(n=("target", "size"), pos=("target", "sum"), peak=("peak_24h", "sum"))
        .reset_index()
    )
    tot = (
        cy.groupby("fips")[["n", "pos", "peak"]].sum().add_suffix("_all").reset_index()
    )

    df = df.merge(tot, on="fips", how="left").merge(cy, on=["fips", "year"], how="left")
    df[["n_all", "pos_all", "peak_all", "n", "pos", "peak"]] = df[
        ["n_all", "pos_all", "peak_all", "n", "pos", "peak"]
    ].fillna(0)
    own = df["split"].eq("train")  # remove the row's own year for training rows
    n_hist = df["n_all"] - np.where(own, df["n"], 0)
    pos_hist = df["pos_all"] - np.where(own, df["pos"], 0)
    peak_hist = df["peak_all"] - np.where(own, df["peak"], 0)

    df["hist_episodes"] = n_hist
    df["hist_outage_rate"] = (pos_hist + HISTORY_PRIOR * p0) / (n_hist + HISTORY_PRIOR)
    df["hist_mean_peak"] = (peak_hist + HISTORY_PRIOR * peak0) / (
        n_hist + HISTORY_PRIOR
    )
    df = df.drop(columns=["n_all", "pos_all", "peak_all", "n", "pos", "peak"])
    return df


def assemble_table(df):
    """Return a sorted modeling table, feature names and analysis-only outcome names."""
    ID_COLS = ["fips", "episode_id", "state_abbr", "fema_region", "year", "t0"]
    FEATURES = [
        # storm (whole-episode summary)
        "n_events",
        "n_tstm_wind",
        "n_hail",
        "n_tornado",
        "n_high_wind",
        "n_flood",
        "n_winter",
        "n_ice_storm",
        "n_tropical",
        "n_cold",
        "n_lightning",
        "n_wildfire",
        "n_other",
        "n_county_coded",
        "n_zone_coded",
        "share_zone_coded",
        "any_partial_county",
        "max_wind_kts",
        "max_hail_in",
        "max_hurricane_cat",
        "county_episode_hours",
        "episode_counties",
        "log_episode_counties",
        "episode_states",
        "episode_events",
        # time
        "month",
        "local_hour",
        # county
        "log_customers",
        "log_pop",
        "dqi",
        "pct_customers_tracked",
        # conditions before the storm
        "pre_peak_24h",
        "pct_out_at_start",
        # county history (training years only)
        "hist_episodes",
        "hist_outage_rate",
        "hist_mean_peak",
    ]
    # Known only after the window: kept for analysis, never as model inputs (retro_ prefix)
    RETRO = {
        "peak_24h": "retro_peak_24h",
        "peak_customers_out": "retro_peak_customers_out",
        "hour_of_peak": "retro_hour_of_peak",
    }

    table = df.rename(columns=RETRO)[
        ID_COLS + FEATURES + ["target"] + list(RETRO.values()) + ["split"]
    ]
    # Distinct episodes can start in the same county at the same time. Include
    # the unique episode key so SQL aggregation/input order cannot break ties.
    table = table.sort_values(["t0", "fips", "episode_id"]).reset_index(drop=True)
    return table, FEATURES, RETRO


def validate_table(table, features):
    """Enforce model-table keys, labels, time splits and feature/outcome separation."""
    require_checks(
        {
            "nonempty": len(table) > 0,
            "unique county/episode": not table.duplicated(["fips", "episode_id"]).any(),
            "nonmissing keys": table[["fips", "episode_id"]].notna().all().all(),
            "binary target": table.target.notna().all()
            and table.target.isin([0, 1]).all(),
            "target definition": table.target.eq(
                (table.retro_peak_24h >= TARGET_PCT).astype(int)
            ).all(),
            "lower 48 + DC": table.state_abbr.isin(STATE_UTC_OFFSET).all(),
            "peak bounds": table.retro_peak_24h.between(0, 1).all(),
            "split years": table.split.eq(
                np.select(
                    [
                        table.year.between(TRAIN_FIRST, TRAIN_LAST),
                        table.year.eq(VAL_YEAR),
                        table.year.eq(TEST_YEAR),
                    ],
                    ["train", "validation", "test"],
                    default="invalid",
                )
            ).all(),
            "no retrospective feature": not any(
                c.startswith("retro_") or c == "target" for c in features
            ),
            "episode splits": table.groupby("episode_id").split.nunique().le(1).all(),
        },
        name="model table",
    )


def build_table(con):
    """Transform registered prepared views; no file persistence or report side effects."""
    scope_storms(con)
    aggregate_episodes(con)
    select_observed_episodes(con)
    frame = add_history(add_county_time_features(label_episodes(con)))
    table, features, retro = assemble_table(frame)
    validate_table(table, features)
    return table, features, retro


def build(paths=None, *, export_dir=None, write_holdout=False):
    """Publish train/validation/test-feature files; creating sealed labels is explicit.

    This rebuild reads prepared raw outcomes. It is not called by notebook analysis
    or verification of the existing release. Existing holdout files are never read.
    """
    paths = paths or ProjectPaths()
    with connection(paths) as con:
        attach_prepared(con, paths)
        table, features, retro = build_table(con)
    ids = ["fips", "episode_id", "state_abbr", "fema_region", "year", "t0"]
    with staged_directory(paths.processed) as stage:
        for split in ["train", "validation"]:
            table.loc[table.split.eq(split)].drop(columns="split").to_parquet(
                stage / f"{split}.parquet", index=False
            )
        table.loc[table.split.eq("test"), ids + features].to_parquet(
            stage / "test_features.parquet", index=False
        )
        (stage / "build.json").write_text(
            json.dumps({"definition": "retrospective-hourly-v1", "rows": len(table)})
        )
    if write_holdout:
        with staged_directory(paths.root / "data/holdout") as stage:
            table.loc[
                table.split.eq("test"),
                ["fips", "episode_id", "target"] + list(retro.values()),
            ].to_parquet(stage / "test_labels.parquet", index=False)
    summary = table.groupby(["split", "year"]).size().rename("rows").reset_index()
    export_results({"rows_by_year": summary}, directory=export_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build"])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--write-holdout",
        action="store_true",
        help="Explicitly generate separated test labels during a rebuild.",
    )
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args()
    print(
        build(
            ProjectPaths(offline=args.offline),
            export_dir=args.export_dir,
            write_holdout=args.write_holdout,
        )
    )


if __name__ == "__main__":
    main()
