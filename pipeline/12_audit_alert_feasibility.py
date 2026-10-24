"""Assess whether archived NWS alerts can support prediction before storm arrival.

Run: python pipeline/12_audit_alert_feasibility.py   (after stages 08-11)
Outputs:
  reports/forecast_feasibility/   inventory, UGC mapping, lead times, alert follow-up, labels for
    24 h alert windows, and timing of the first major crossing after an alert (evidence for D-21)
  data/03_cleaned/southeast/county_alert_events.parquet   one row per county and alert event;
    stage 13 builds the modeling rows (72 h alert windows) from it

The 24 h alert windows here reproduce the feasibility analysis that informed checkpoint A;
they are not the modeling rows.

Information timing: an alert is available from its product issuance time (utc_prodissue).
Only utc_prodissue, utc_issue (start of validity for that UGC) and utc_init_expire are
used. utc_expire, status and utc_updated describe the event's final state after later
updates, so they are future information at issuance and are never used here.

UGCs: county codes (FLC001) map to county FIPS directly; zone codes (FLZ123) use the
correlation version in effect at issuance, or the nearest version containing the zone.
Storm-based polygon rows carry no UGC; the same warnings also have county rows.

Analyses use training years (t0 or issuance year <= 2023) only. Inventory counts are
structural and cover all downloaded years.
"""
import argparse
import json
import time

from common import (SE_ABBR_BY_FIPS, STORM_ALERT_FAMILIES as STORM_FAMILIES, alert_family_sql as family_sql, connect, copy_query,
                    greedy_window_starts, lit, split_sql, sql_list, write_json)
from labels import MAJOR_CUSTOMERS, major_state, ongoing_at_t0, summarize_windows
from paths import FORECAST_REPORTS, RAW_IEM_VTEC, SOUTHEAST, ZONE_COUNTY

LOOKBACK_HOURS = 72
THRESHOLDS = (1000, 5000, 10000, 25000)


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    started = time.perf_counter()
    FORECAST_REPORTS.mkdir(parents=True, exist_ok=True)
    con = connect()
    files = sorted(RAW_IEM_VTEC.glob('vtec_southeast_*.csv'))
    manifest = json.loads((RAW_IEM_VTEC / 'manifest.json').read_text())
    con.execute(f"""CREATE TABLE raw AS SELECT * FROM read_csv({[str(f) for f in files]}, header=true, all_varchar=true,
        union_by_name=true, filename=true)""")
    con.execute(f"""CREATE TABLE alerts AS SELECT wfo, phenomena, significance, eventid, vtec_year, gtype, ugc, product_id,
            strptime(utc_prodissue, '%Y-%m-%d %H:%M') AT TIME ZONE 'UTC' AS prodissue_utc,
            strptime(utc_issue, '%Y-%m-%d %H:%M') AT TIME ZONE 'UTC' AS issue_utc,
            try_strptime(utc_init_expire, '%Y%m%d%H%M') AT TIME ZONE 'UTC' AS init_expire_utc,
            {family_sql('phenomena')} AS family
        FROM raw""")
    nulls = con.execute('SELECT count(*) FILTER (WHERE prodissue_utc IS NULL), count(*) FILTER (WHERE issue_utc IS NULL) FROM alerts').fetchone()
    copy_query(con, """SELECT vtec_year, family, phenomena || '.' || significance AS alert, gtype, count(*) AS rows,
            count(DISTINCT wfo || phenomena || significance || eventid) AS events, count(ugc) AS ugc_rows
        FROM alerts GROUP BY ALL ORDER BY vtec_year, rows DESC""", FORECAST_REPORTS / 'alert_inventory.csv', fmt='csv')

    print('Mapping alert UGCs to counties', flush=True)
    abbr_to_fips = {abbr: fips for fips, abbr in SE_ABBR_BY_FIPS.items()}
    con.execute(f"CREATE TABLE abbr AS SELECT * FROM (VALUES {', '.join(f'({lit(a)}, {lit(f)})' for a, f in abbr_to_fips.items())}) t(abbr, state_fips)")
    con.execute(f"CREATE VIEW versions AS SELECT * FROM read_parquet({lit(ZONE_COUNTY)})")
    con.execute("""CREATE TABLE ugc_rows AS SELECT a.*, substr(ugc, 1, 2) AS ugc_state, substr(ugc, 3, 1) AS ugc_kind,
            substr(ugc, 1, 2) || substr(ugc, 4, 3) AS state_zone
        FROM alerts a WHERE ugc IS NOT NULL""")
    con.execute("""CREATE TABLE county_alerts AS
        WITH county_codes AS (
            SELECT u.*, b.state_fips || substr(u.ugc, 4, 3) AS county_fips, 'county_ugc' AS ugc_mapping
            FROM ugc_rows u JOIN abbr b ON b.abbr = u.ugc_state WHERE u.ugc_kind = 'C'),
        zone_rows AS (SELECT u.*, (SELECT max(effective_date) FROM (SELECT DISTINCT effective_date FROM versions) d
                                   WHERE d.effective_date <= CAST(u.prodissue_utc AS DATE)) AS eff_date
            FROM ugc_rows u WHERE u.ugc_kind = 'Z' AND u.ugc_state IN (SELECT abbr FROM abbr)),
        zone_versions AS (SELECT DISTINCT state_zone, effective_date FROM versions),
        zone_pick AS (SELECT z.*, coalesce(
                (SELECT max(v.effective_date) FROM zone_versions v WHERE v.state_zone = z.state_zone AND v.effective_date = z.eff_date),
                (SELECT arg_min(v.effective_date, abs(date_diff('day', v.effective_date, CAST(z.prodissue_utc AS DATE))))
                 FROM zone_versions v WHERE v.state_zone = z.state_zone)) AS used_date
            FROM zone_rows z),
        zone_counties AS (SELECT z.* EXCLUDE (eff_date, used_date), v.county_fips,
                CASE WHEN z.used_date = z.eff_date THEN 'zone_effective_version' ELSE 'zone_nearest_version' END AS ugc_mapping
            FROM zone_pick z JOIN versions v ON v.state_zone = z.state_zone AND v.effective_date = z.used_date)
        SELECT * FROM county_codes UNION ALL BY NAME SELECT * FROM zone_counties""")
    copy_query(con, """SELECT ugc_kind, coalesce(m.ugc_mapping, 'unmapped') AS mapping, count(*) AS ugc_rows
        FROM ugc_rows u LEFT JOIN (SELECT DISTINCT product_id, ugc, phenomena, significance, eventid, ugc_mapping FROM county_alerts) m
          USING (product_id, ugc, phenomena, significance, eventid)
        GROUP BY ALL ORDER BY ALL""", FORECAST_REPORTS / 'ugc_mapping.csv', fmt='csv')
    # One row per county and alert event: first issuance reaching the county (later UGC additions included).
    con.execute(f"""CREATE TABLE county_events AS SELECT county_fips, wfo, phenomena, significance, eventid, vtec_year, family,
            min(prodissue_utc) AS first_prodissue_utc, min(issue_utc) AS first_issue_utc,
            arg_min(init_expire_utc, prodissue_utc) AS init_expire_utc, {split_sql('min(prodissue_utc)')} AS split
        FROM county_alerts GROUP BY county_fips, wfo, phenomena, significance, eventid, vtec_year, family""")
    copy_query(con, 'SELECT * FROM county_events ORDER BY county_fips, first_prodissue_utc', SOUTHEAST / 'county_alert_events.parquet')

    print('Lead time of archived alerts before NOAA-recorded onsets (training)', flush=True)
    con.execute(f"CREATE VIEW opps AS SELECT * FROM read_parquet({lit(SOUTHEAST / 'onset_opportunities.parquet')})")
    con.execute(f"CREATE VIEW storm_events AS SELECT * FROM read_parquet({lit(SOUTHEAST / 'storm_events_southeast.parquet')})")
    con.execute(f"""CREATE TABLE prior AS SELECT o.opportunity_id, o.trigger_hazard_groups[1] AS hazard, a.family, a.significance,
            date_diff('minute', a.first_prodissue_utc, o.t0) / 60.0 AS lead_hours
        FROM opps o JOIN county_events a ON a.county_fips = o.county_fips
        WHERE o.split = 'train' AND NOT o.split_boundary_window AND a.family IN {sql_list(STORM_FAMILIES)}
          AND a.significance IN ('W', 'A', 'Y')
          AND a.first_prodissue_utc <= o.t0 AND a.first_prodissue_utc >= o.t0 - INTERVAL {LOOKBACK_HOURS} HOUR
          AND coalesce(a.init_expire_utc, a.first_prodissue_utc + INTERVAL 72 HOUR) >= o.t0 - INTERVAL 1 HOUR""")
    copy_query(con, """WITH base AS (SELECT opportunity_id, trigger_hazard_groups[1] AS hazard FROM opps
                      WHERE split = 'train' AND NOT split_boundary_window),
        best AS (SELECT opportunity_id, max(lead_hours) AS max_lead_any,
                 max(lead_hours) FILTER (WHERE significance = 'W') AS max_lead_warning,
                 max(lead_hours) FILTER (WHERE significance = 'A') AS max_lead_watch,
                 max(lead_hours) FILTER (WHERE significance = 'Y') AS max_lead_advisory
                 FROM prior GROUP BY 1)
        SELECT b.hazard, count(*) AS train_opportunities,
            round(count(p.opportunity_id) / count(*), 4) AS share_with_prior_alert,
            round(count(p.max_lead_warning) / count(*), 4) AS share_with_prior_warning,
            round(count(p.max_lead_watch) / count(*), 4) AS share_with_prior_watch,
            round(median(p.max_lead_any), 2) AS median_earliest_lead_hours,
            round(quantile_cont(p.max_lead_any, 0.25), 2) AS q25_earliest_lead_hours,
            round(quantile_cont(p.max_lead_any, 0.75), 2) AS q75_earliest_lead_hours,
            round(median(p.max_lead_warning), 2) AS median_warning_lead_hours,
            round(count(*) FILTER (WHERE p.max_lead_any >= 6) / count(*), 4) AS share_alert_6h_before,
            round(count(*) FILTER (WHERE p.max_lead_any >= 24) / count(*), 4) AS share_alert_24h_before
        FROM base b LEFT JOIN best p USING (opportunity_id) GROUP BY ROLLUP (b.hazard) ORDER BY train_opportunities DESC""",
               FORECAST_REPORTS / 'onset_prior_alerts.csv', fmt='csv')

    print('Do alerts include storms that were never recorded? (training)', flush=True)
    con.execute(f"""CREATE TABLE followup AS SELECT a.*, EXISTS (
            SELECT 1 FROM storm_events e WHERE e.county_fips = a.county_fips AND e.hazard_group <> 'temperature' AND e.hazard_group <> 'other'
              AND e.begin_utc >= a.first_prodissue_utc - INTERVAL 1 HOUR
              AND e.begin_utc <= least(coalesce(a.init_expire_utc, a.first_prodissue_utc + INTERVAL 72 HOUR), a.first_prodissue_utc + INTERVAL 72 HOUR) + INTERVAL 12 HOUR) AS noaa_event_followed
        FROM county_events a WHERE a.split = 'train' AND a.family IN {sql_list(STORM_FAMILIES)} AND a.significance IN ('W', 'A', 'Y')""")
    copy_query(con, """SELECT family, phenomena || '.' || significance AS alert, count(*) AS county_alerts,
            round(avg(noaa_event_followed::INTEGER), 4) AS share_followed_by_noaa_event,
            round(median(date_diff('minute', first_prodissue_utc, first_issue_utc) / 60.0), 2) AS median_hours_issuance_to_valid
        FROM followup GROUP BY ALL HAVING count(*) >= 50 ORDER BY county_alerts DESC""",
               FORECAST_REPORTS / 'alert_event_followup.csv', fmt='csv')

    print('Alert-triggered county windows and label availability (training)', flush=True)
    # Greedy non-overlapping 24 h windows per county from first issuance of storm-family warnings and watches.
    con.execute(f"""CREATE TABLE triggers AS SELECT county_fips, first_prodissue_utc AS t, family, phenomena, significance
        FROM county_events WHERE family IN {sql_list(STORM_FAMILIES)} AND significance IN ('W', 'A')""")
    starts = greedy_window_starts(con.execute('SELECT county_fips, t FROM triggers ORDER BY county_fips, t').fetchall(), 24)
    con.execute('CREATE TABLE alert_starts (county_fips VARCHAR, t0 TIMESTAMPTZ)')
    con.executemany('INSERT INTO alert_starts VALUES (?, ?)', starts)
    con.execute(f"""CREATE TABLE alert_opps AS WITH tr AS (
            SELECT county_fips, t, list(DISTINCT phenomena || '.' || significance ORDER BY phenomena || '.' || significance) AS trigger_alerts,
                list(DISTINCT family ORDER BY family) AS trigger_families, bool_or(significance = 'W') AS trigger_has_warning
            FROM triggers GROUP BY ALL)
        SELECT s.county_fips || '_A' || strftime(s.t0, '%Y%m%dT%H%M') AS opportunity_id,
            s.county_fips, s.t0, {split_sql('s.t0')} AS split, tr.trigger_alerts, tr.trigger_families, tr.trigger_has_warning,
            -- Realized after t0: audit only, never a predictor.
            EXISTS (SELECT 1 FROM storm_events e WHERE e.county_fips = s.county_fips AND e.hazard_group NOT IN ('temperature', 'other')
                    AND e.begin_utc >= s.t0 - INTERVAL 1 HOUR AND e.begin_utc < s.t0 + INTERVAL 24 HOUR) AS retro_noaa_event_in_window
        FROM alert_starts s JOIN tr ON tr.county_fips = s.county_fips AND tr.t = s.t0""")
    assert con.execute('SELECT count(*) FROM alert_opps').fetchone()[0] == len(starts)
    con.execute("""CREATE TABLE alert_audit AS SELECT opportunity_id, county_fips, t0 FROM alert_opps
        WHERE split = 'train' AND t0 >= TIMESTAMPTZ '2014-11-02 04:00:00+00' AND t0 + INTERVAL 24 HOUR <= TIMESTAMPTZ '2024-01-01 00:00:00+00'""")
    con.execute(f"CREATE VIEW obs AS SELECT county_fips, obs_date, observation_utc, customers_out, on_slot FROM read_parquet({lit(SOUTHEAST / 'eaglei_southeast.parquet')})")
    summarize_windows(con, 'alert_audit', 'obs', 'alert_windows')
    parts = [f"""SELECT '{t}' AS threshold, o.retro_noaa_event_in_window, count(*) AS train_alert_windows,
            count(*) FILTER (WHERE {major_state(t)} = 'major') AS major,
            count(*) FILTER (WHERE {major_state(t)} = 'non_major') AS non_major_strict,
            count(*) FILTER (WHERE {major_state(t, 'cov90')} = 'non_major') AS non_major_cov90,
            count(*) FILTER (WHERE {major_state(t)} = 'unknown') AS unknown_strict
        FROM alert_windows w JOIN alert_opps o USING (opportunity_id) GROUP BY ALL""" for t in THRESHOLDS]
    copy_query(con, f"SELECT * FROM ({' UNION ALL '.join(parts)}) ORDER BY try_cast(threshold AS INTEGER), retro_noaa_event_in_window",
               FORECAST_REPORTS / 'alert_window_labels.csv', fmt='csv')

    print('Timing of the first major crossing after an alert or onset (training)', flush=True)
    # Evidence for the outcome-window decision (D-21): first crossing of 5,000 customers within
    # 96 h of t0, for 24 h alert audit windows and NOAA-onset rows not already major at t0.
    family = ' '.join(f"WHEN list_contains(o.trigger_families, '{f}') THEN '{f}'"
                      for f in ('tropical', 'winter', 'wind', 'convective', 'flood', 'coastal'))
    copy_query(con, f"""WITH rows AS (
            SELECT 'alert' AS design, o.opportunity_id, o.county_fips, o.t0, CASE {family} END AS trigger_family
            FROM alert_opps o JOIN alert_windows w USING (opportunity_id) WHERE NOT {ongoing_at_t0()}
            UNION ALL
            SELECT 'onset', opportunity_id, county_fips, t0, trigger_hazard_groups[1] FROM opps
            WHERE split = 'train' AND NOT split_boundary_window),
        t AS (SELECT * FROM rows WHERE t0 >= TIMESTAMPTZ '2014-11-02 04:00:00+00' AND t0 + INTERVAL 96 HOUR <= TIMESTAMPTZ '2024-01-01 00:00:00+00'),
        days AS (SELECT *, CAST(day AS DATE) AS day FROM (SELECT *, unnest(generate_series(CAST(t0 AS DATE),
                    CAST(t0 + INTERVAL 96 HOUR AS DATE), INTERVAL 1 DAY)) AS day FROM t)),
        firsts AS (SELECT d.design, d.opportunity_id, d.trigger_family, min(date_diff('minute', d.t0, o.observation_utc)) / 60.0 AS hours
            FROM days d JOIN obs o ON o.county_fips = d.county_fips AND o.obs_date = d.day
            WHERE o.observation_utc >= d.t0 AND o.observation_utc < d.t0 + INTERVAL 96 HOUR AND o.customers_out >= {MAJOR_CUSTOMERS}
            GROUP BY ALL)
        SELECT design, trigger_family, count(*) AS windows_crossing_within_96h,
            round(count(*) FILTER (WHERE hours < 24) / count(*), 3) AS share_within_24h,
            round(count(*) FILTER (WHERE hours < 48) / count(*), 3) AS share_within_48h,
            round(count(*) FILTER (WHERE hours < 72) / count(*), 3) AS share_within_72h,
            round(median(hours), 1) AS median_hours_to_first_crossing
        FROM firsts GROUP BY ALL ORDER BY design, windows_crossing_within_96h DESC""",
               FORECAST_REPORTS / 'outcome_window_timing.csv', fmt='csv')

    summary = {
        'files': {f.name: manifest['files'][f.name]['bytes'] for f in files},
        'rows': con.execute('SELECT count(*) FROM alerts').fetchone()[0],
        'unparsed_prodissue_or_issue': list(nulls),
        'county_alert_rows': con.execute('SELECT count(*) FROM county_alerts').fetchone()[0],
        'county_alert_events': con.execute('SELECT count(*) FROM county_events').fetchone()[0],
        'audit_24h_alert_windows_by_split': dict(con.execute('SELECT split, count(*) FROM alert_opps GROUP BY 1 ORDER BY 1').fetchall()),
        'train_alert_windows_summarized': con.execute('SELECT count(*) FROM alert_windows').fetchone()[0],
        'leakage_guard': 'utc_expire, status and utc_updated are not read (final event state, unknown at issuance).',
        'runtime_seconds': round(time.perf_counter() - started, 1),
    }
    write_json(summary, FORECAST_REPORTS / 'summary.json')
    print(json.dumps(summary, indent=2), flush=True)
    con.close()


if __name__ == '__main__':
    main()
