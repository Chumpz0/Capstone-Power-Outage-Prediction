"""Build model-input tables for the approved major-outage task (checkpoints A and B).

Run: python pipeline/13_build_major_outage_dataset.py   (after stages 10 and 12)
Outputs in data/04_model_input/southeast_v1/:
  alert_{train,validation}.parquet, alert_test_features.parquet
      primary design (D-17, D-21): non-overlapping 72 h county windows starting at the
      first storm-family NWS watch or warning issued for the county
  onset_{train,validation}.parquet, onset_test_features.parquet
      comparison design: 24 h county windows starting at a NOAA-recorded onset
  manifest.json   feature lists, label rules, row counts, exclusions, input hashes

Features use only information available at t0 and are computed for every split. Outcome
columns (labels, peaks, coverage) exist for training and validation rows only; the test
files hold identifiers and features, and the frozen final evaluation computes outcomes once.

Target (D-18 to D-21): major = at least 5,000 customers out in [t0, t0 + H), H = 72 h for
alert rows and 24 h for onset rows; non_major under cov90; unknown otherwise. Windows
already major at t0 are 'ongoing' and excluded. `covered_window` (cov90 coverage, not
ongoing) defines the symmetric evaluation population (D-22) and the severity population
(provisional, O-7), whose target is the observed peak.

Feature definitions and availability are documented in docs/feature_dictionary.md.
"""
import argparse
import json

from common import (STORM_ALERT_FAMILIES, STORM_GROUPS, connect, copy_query, file_sha256, greedy_window_starts, lit, split_sql,
                    sql_list, write_json)
from labels import (MAJOR_CUSTOMERS, PRIMARY_RULE, non_major_rule, ongoing_at_t0, primary_label, single_run_crossing,
                    summarize_pre_t0, summarize_windows)
from paths import MODEL_INPUT, SOUTHEAST

OUT = MODEL_INPUT / 'southeast_v1'
VERSION = 2
WINDOW_HOURS = {'alert': 72, 'onset': 24}   # decision D-21
FIRST_T0 = '2014-11-02 04:00:00+00'      # 24 h after the first EAGLE-I run, so the pre-t0 window is covered
SPLIT_END = {'train': '2024-01-01 00:00:00+00', 'validation': '2025-01-01 00:00:00+00', 'test': '2026-01-01 00:00:00+00'}
ACTIVE_CAP = 'INTERVAL 72 HOUR'          # tropical VTEC events carry placeholder expiries weeks ahead
TRIGGER_CODES = ('TO.W', 'SV.W', 'FF.W', 'FA.W', 'FL.W', 'HU.W', 'TR.W', 'SS.W', 'HW.W', 'EW.W', 'WS.W', 'IS.W', 'BZ.W',
                 'CF.W', 'TO.A', 'SV.A', 'FF.A', 'FA.A', 'HU.A', 'TR.A', 'SS.A', 'HW.A', 'WS.A')
FAMILY_PRIORITY = ('tropical', 'winter', 'wind', 'convective', 'flood', 'coastal')
ACTIVE_FAMILIES = STORM_ALERT_FAMILIES + ('temperature',)
SIZE_BINS = ((0, 25_000, '1: <25k'), (25_000, 50_000, '2: 25k-50k'), (50_000, 100_000, '3: 50k-100k'),
             (100_000, 250_000, '4: 100k-250k'), (250_000, 500_000, '5: 250k-500k'), (500_000, 10**9, '6: 500k+'))

ACTIVE_COLS = [f'act_{f}_{s}' for f in ACTIVE_FAMILIES for s in 'WAY']
COMMON_NUMERIC = (['month', 'hour_utc', 'day_of_week', 'log_population', 'log_mcc_valid',
                   'log_at_t0_value', 'at_t0_row_present', 'log_last_hour_max', 'log_pre_max_24h', 'pre_observed', 'pre_null',
                   'hours_since_last_observation', 'hist_days_observed', 'hist_days_major', 'log_hist_max_daily',
                   'hist30_days_ge_1000', 'hist_span_days', 'alerts_prior_72h', 'hours_since_first_active_alert',
                   'max_hours_to_valid_start', 'region_warned_counties', 'state_warned_counties',
                   'region_tropical_alert_counties'] + ACTIVE_COLS)
COMMON_CATEGORICAL = ['state']
ALERT_ONLY = (['trigger_has_warning', 'trigger_hours_to_valid_start', 'trigger_validity_hours']
              + [f'trig_fam_{f}' for f in STORM_ALERT_FAMILIES] + [f"trig_{c.replace('.', '_')}" for c in TRIGGER_CODES])
ONSET_ONLY = [f'noaa_{g}' for g in STORM_GROUPS]
OUTCOMES = ['label', 'y_major', 'peak_customers', 'covered_window', 'peak_complete', 'n_observed', 'max_gap_slots',
            'single_run_crossing', 'bracketed_gap_slots', 'window_peak_exceeds_customers']


def size_bin_sql(col):
    cases = ' '.join(f"WHEN {col} >= {lo} AND {col} < {hi} THEN '{label}'" for lo, hi, label in SIZE_BINS)
    return f"CASE {cases} ELSE 'unknown' END"


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    con = connect()
    inputs = {name: SOUTHEAST / name for name in ('eaglei_southeast.parquet', 'county_reference.parquet', 'county_alert_events.parquet',
                                                   'onset_opportunities.parquet', 'storm_events_southeast.parquet')}
    for name, path in inputs.items():
        con.execute(f'CREATE VIEW {name.split(".")[0]} AS SELECT * FROM read_parquet({lit(path)})')
    con.execute('CREATE VIEW obs AS SELECT county_fips, obs_date, observation_utc, customers_out, on_slot FROM eaglei_southeast')

    print('Alert rows: non-overlapping 72 h county windows from storm-family watches and warnings', flush=True)
    hours = WINDOW_HOURS['alert']
    con.execute(f"""CREATE TABLE triggers AS SELECT county_fips, first_prodissue_utc AS t, family, phenomena, significance
        FROM county_alert_events WHERE family IN {sql_list(STORM_ALERT_FAMILIES)} AND significance IN ('W', 'A')""")
    starts = greedy_window_starts(con.execute('SELECT county_fips, t FROM triggers ORDER BY county_fips, t').fetchall(), hours)
    con.execute('CREATE TABLE alert_starts (county_fips VARCHAR, t0 TIMESTAMPTZ)')
    con.executemany('INSERT INTO alert_starts VALUES (?, ?)', starts)
    con.execute(f"""CREATE TABLE alert_opportunities AS WITH tr AS (
            SELECT county_fips, t, list(DISTINCT phenomena || '.' || significance ORDER BY phenomena || '.' || significance) AS trigger_alerts,
                list(DISTINCT family ORDER BY family) AS trigger_families, bool_or(significance = 'W') AS trigger_has_warning
            FROM triggers GROUP BY ALL)
        SELECT s.county_fips || '_A' || strftime(s.t0, '%Y%m%dT%H%M') AS opportunity_id, s.county_fips, s.t0,
            {split_sql('s.t0')} AS split, tr.trigger_alerts, tr.trigger_families, tr.trigger_has_warning,
            -- Realized after t0: audit only, never a predictor.
            EXISTS (SELECT 1 FROM storm_events_southeast e WHERE e.county_fips = s.county_fips AND e.hazard_group IN {sql_list(STORM_GROUPS)}
                    AND e.begin_utc >= s.t0 - INTERVAL 1 HOUR AND e.begin_utc < s.t0 + INTERVAL {hours} HOUR) AS retro_noaa_event_in_window
        FROM alert_starts s JOIN tr ON tr.county_fips = s.county_fips AND tr.t = s.t0""")
    assert con.execute('SELECT count(*) FROM alert_opportunities').fetchone()[0] == len(starts)

    print('Rows for both designs and eligibility', flush=True)
    trig_cols = ', '.join(f"list_contains(trigger_alerts, '{c}') AS trig_{c.replace('.', '_')}" for c in TRIGGER_CODES)
    fam_cols = ', '.join(f"list_contains(trigger_families, '{f}') AS trig_fam_{f}" for f in STORM_ALERT_FAMILIES)
    primary_family = ' '.join(f"WHEN list_contains(trigger_families, '{f}') THEN '{f}'" for f in FAMILY_PRIORITY)
    noaa_cols = ', '.join(f"list_contains(trigger_hazard_groups, '{g}') AS noaa_{g}" for g in STORM_GROUPS)
    con.execute(f"""CREATE TABLE rows_raw AS
        SELECT 'alert' AS design, opportunity_id, county_fips, t0, split, trigger_has_warning, {trig_cols}, {fam_cols},
            CASE {primary_family} END AS primary_family, {', '.join(f'NULL::BOOLEAN AS noaa_{g}' for g in STORM_GROUPS)},
            NULL::BOOLEAN AS trigger_partial_county, retro_noaa_event_in_window
        FROM alert_opportunities
        UNION ALL BY NAME
        SELECT 'onset' AS design, opportunity_id, county_fips, t0, split, {noaa_cols},
            trigger_hazard_groups[1] AS primary_family, trigger_partial_county
        FROM onset_opportunities""")
    split_end = ' '.join(f"WHEN '{s}' THEN TIMESTAMPTZ '{e}'" for s, e in SPLIT_END.items())
    con.execute(f"""CREATE TABLE rows_all AS SELECT r.*, c.state, c.population_2020, c.mcc_customers, c.mcc_status,
            c.outage_plausibility, c.eaglei_tracked,
            CASE WHEN r.split NOT IN ('train', 'validation', 'test') THEN 'out_of_scope'
                 WHEN NOT c.eaglei_tracked THEN 'county_never_in_eaglei'
                 WHEN r.t0 < TIMESTAMPTZ '{FIRST_T0}' THEN 'before_eaglei_coverage'
                 -- Outcome windows may not reach into the next split.
                 WHEN r.t0 + CASE r.design WHEN 'alert' THEN INTERVAL {WINDOW_HOURS['alert']} HOUR ELSE INTERVAL {WINDOW_HOURS['onset']} HOUR END
                      > CASE r.split {split_end} END THEN 'window_crosses_split_boundary'
                 ELSE 'eligible' END AS eligibility
        FROM rows_raw r JOIN county_reference c USING (county_fips)""")
    con.execute("CREATE TABLE rows AS SELECT * FROM rows_all WHERE eligibility = 'eligible'")
    assert con.execute('SELECT count(*) - count(DISTINCT opportunity_id) FROM rows').fetchone()[0] == 0

    print('EAGLE-I conditions before t0', flush=True)
    summarize_pre_t0(con, 'rows', 'obs', 'pre')

    print('Alerts in effect and regional extent at t0', flush=True)
    # Validity runs from the alert's start to its initial expiry, capped 72 h after the start; an alert
    # issued days before its start is in effect as a forecast product from issuance.
    con.execute(f"""CREATE TABLE alerts AS SELECT county_fips, substr(county_fips, 1, 2) AS state_fips, family, significance,
            phenomena || '.' || significance AS code, first_prodissue_utc AS issued, first_issue_utc AS valid_from,
            greatest(least(coalesce(init_expire_utc, first_issue_utc + INTERVAL 12 HOUR), first_issue_utc + {ACTIVE_CAP}),
                     first_issue_utc) AS valid_to
        FROM county_alert_events WHERE family IN {sql_list(ACTIVE_FAMILIES)}""")
    act = ', '.join(f"count(*) FILTER (WHERE a.family = '{f}' AND a.significance = '{s}') AS act_{f}_{s}" for f in ACTIVE_FAMILIES for s in 'WAY')
    con.execute(f"""CREATE TABLE active AS SELECT r.opportunity_id, {act},
            max(date_diff('minute', a.issued, r.t0)) FILTER (WHERE a.family IN {sql_list(STORM_ALERT_FAMILIES)} AND a.significance IN ('W', 'A')) / 60.0 AS hours_since_first_active_alert,
            max(greatest(date_diff('minute', r.t0, a.valid_from), 0)) FILTER (WHERE a.family IN {sql_list(STORM_ALERT_FAMILIES)} AND a.significance IN ('W', 'A')) / 60.0 AS max_hours_to_valid_start,
            max(greatest(date_diff('minute', a.issued, a.valid_from), 0)) FILTER (WHERE a.issued = r.t0 AND a.family IN {sql_list(STORM_ALERT_FAMILIES)}) / 60.0 AS trigger_hours_to_valid_start,
            max(least(date_diff('minute', a.valid_from, a.valid_to), 72 * 60)) FILTER (WHERE a.issued = r.t0 AND a.family IN {sql_list(STORM_ALERT_FAMILIES)}) / 60.0 AS trigger_validity_hours
        FROM rows r JOIN alerts a ON a.county_fips = r.county_fips AND a.issued <= r.t0 AND a.issued >= r.t0 - {ACTIVE_CAP} AND a.valid_to >= r.t0
        GROUP BY r.opportunity_id""")
    con.execute(f"""CREATE TABLE prior_alerts AS SELECT r.opportunity_id, count(*) AS alerts_prior_72h
        FROM rows r JOIN alerts a ON a.county_fips = r.county_fips AND a.issued < r.t0 AND a.issued >= r.t0 - INTERVAL 72 HOUR
        WHERE a.family IN {sql_list(STORM_ALERT_FAMILIES)} GROUP BY 1""")
    con.execute(f"""CREATE TABLE extent AS SELECT r.opportunity_id,
            count(DISTINCT a.county_fips) FILTER (WHERE a.significance = 'W') AS region_warned_counties,
            count(DISTINCT a.county_fips) FILTER (WHERE a.significance = 'W' AND a.state_fips = substr(r.county_fips, 1, 2)) AS state_warned_counties,
            count(DISTINCT a.county_fips) FILTER (WHERE a.family = 'tropical') AS region_tropical_alert_counties
        FROM rows r JOIN alerts a ON a.issued <= r.t0 AND a.valid_to >= r.t0
        WHERE a.family IN {sql_list(STORM_ALERT_FAMILIES)} AND a.significance IN ('W', 'A') GROUP BY 1""")

    print('County outage history from days before t0', flush=True)
    con.execute(f"""CREATE TABLE daily AS SELECT county_fips, obs_date AS d, max(customers_out) AS daily_max
        FROM eaglei_southeast WHERE customers_out IS NOT NULL GROUP BY 1, 2""")
    con.execute("""CREATE TABLE grid AS SELECT c.county_fips, CAST(g AS DATE) AS d
        FROM (SELECT county_fips FROM county_reference WHERE eaglei_tracked) c,
             (SELECT unnest(generate_series(DATE '2014-11-01', DATE '2026-01-01', INTERVAL 1 DAY)) AS g)""")
    con.execute(f"""CREATE TABLE hist AS SELECT g.county_fips, g.d,
            sum(CASE WHEN x.daily_max IS NOT NULL THEN 1 ELSE 0 END) OVER w365 AS hist_days_observed,
            sum(CASE WHEN x.daily_max >= {MAJOR_CUSTOMERS} THEN 1 ELSE 0 END) OVER w365 AS hist_days_major,
            max(x.daily_max) OVER w365 AS hist_max_daily,
            sum(CASE WHEN x.daily_max >= 1000 THEN 1 ELSE 0 END) OVER w30 AS hist30_days_ge_1000,
            least(date_diff('day', DATE '2014-11-01', g.d), 365) AS hist_span_days
        FROM grid g LEFT JOIN daily x USING (county_fips, d)
        WINDOW w365 AS (PARTITION BY g.county_fips ORDER BY g.d RANGE BETWEEN INTERVAL 365 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING),
               w30 AS (PARTITION BY g.county_fips ORDER BY g.d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING)""")

    print('Outcome windows (training and validation only)', flush=True)
    ceiling = "CASE WHEN r.mcc_status = 'valid' THEN r.mcc_customers ELSE r.population_2020 END"
    parts = []
    for design, hours in WINDOW_HOURS.items():
        con.execute(f"""CREATE OR REPLACE TABLE dev_rows AS SELECT opportunity_id, county_fips, t0 FROM rows
            WHERE split IN ('train', 'validation') AND design = '{design}'""")
        summarize_windows(con, 'dev_rows', 'obs', f'win_{design}', hours=hours)
        parts.append(f"""SELECT w.opportunity_id,
            {primary_label(hours=hours)} AS label,
            CASE {primary_label(hours=hours)} WHEN 'major' THEN 1 WHEN 'non_major' THEN 0 END AS y_major,
            w.max_any AS peak_customers,
            -- Coverage holds whatever the label: the symmetric evaluation and severity population.
            NOT {ongoing_at_t0()} AND {non_major_rule(PRIMARY_RULE, hours)} AS covered_window,
            w.n_observed = {4 * hours} AS peak_complete, w.n_observed, w.max_gap_slots,
            {single_run_crossing(MAJOR_CUSTOMERS)} AS single_run_crossing, w.bracketed_gap_slots,
            coalesce(w.max_any > 1.05 * {ceiling}, false) AS window_peak_exceeds_customers
        FROM win_{design} w JOIN rows r USING (opportunity_id)""")
    con.execute(f"CREATE TABLE outcomes AS {' UNION ALL '.join(parts)}")

    log = lambda col: f'log10(1 + {col})'  # noqa: E731
    con.execute(f"""CREATE TABLE dataset AS SELECT r.design, r.opportunity_id, r.county_fips, r.state, r.t0, r.split,
            CAST(r.t0 AS DATE) AS cluster_date,
            month(r.t0) AS month, hour(r.t0) AS hour_utc, dayofweek(r.t0) AS day_of_week,
            log10(r.population_2020) AS log_population,
            CASE WHEN r.mcc_status = 'valid' THEN log10(r.mcc_customers) END AS log_mcc_valid,
            {log('p.at_t0_value')} AS log_at_t0_value, p.at_t0_row_present::INTEGER AS at_t0_row_present,
            {log('p.last_hour_max')} AS log_last_hour_max, {log('p.pre_max_24h')} AS log_pre_max_24h,
            p.pre_observed, p.pre_null, p.minutes_since_last_observation / 60.0 AS hours_since_last_observation,
            h.hist_days_observed, h.hist_days_major, {log('h.hist_max_daily')} AS log_hist_max_daily,
            h.hist30_days_ge_1000, h.hist_span_days,
            coalesce(q.alerts_prior_72h, 0) AS alerts_prior_72h, a.hours_since_first_active_alert, a.max_hours_to_valid_start,
            coalesce(x.region_warned_counties, 0) AS region_warned_counties, coalesce(x.state_warned_counties, 0) AS state_warned_counties,
            coalesce(x.region_tropical_alert_counties, 0) AS region_tropical_alert_counties,
            {', '.join(f'coalesce(a.{c}, 0) AS {c}' for c in ACTIVE_COLS)},
            r.trigger_has_warning::INTEGER AS trigger_has_warning, a.trigger_hours_to_valid_start, a.trigger_validity_hours,
            {', '.join(f'r.trig_fam_{f}::INTEGER AS trig_fam_{f}' for f in STORM_ALERT_FAMILIES)},
            {', '.join(f"r.trig_{c.replace('.', '_')}::INTEGER AS trig_{c.replace('.', '_')}" for c in TRIGGER_CODES)},
            {', '.join(f'r.noaa_{g}::INTEGER AS noaa_{g}' for g in STORM_GROUPS)},
            -- Analysis fields: subgroup reporting only, never predictors.
            r.primary_family, {size_bin_sql('r.population_2020')} AS size_bin, r.outage_plausibility,
            r.trigger_partial_county, r.retro_noaa_event_in_window,
            {', '.join(f'o.{c}' for c in OUTCOMES)}
        FROM rows r JOIN pre p USING (opportunity_id)
        LEFT JOIN hist h ON h.county_fips = r.county_fips AND h.d = CAST(r.t0 AS DATE)
        LEFT JOIN active a USING (opportunity_id) LEFT JOIN prior_alerts q USING (opportunity_id)
        LEFT JOIN extent x USING (opportunity_id) LEFT JOIN outcomes o USING (opportunity_id)""")
    leaked = con.execute(f"SELECT count(*) FROM dataset WHERE split = 'test' AND ({' OR '.join(f'{c} IS NOT NULL' for c in OUTCOMES)})").fetchone()[0]
    assert leaked == 0, 'test rows must not carry outcome columns'
    assert con.execute('SELECT count(*) FROM dataset').fetchone()[0] == con.execute('SELECT count(*) FROM rows').fetchone()[0]

    features = {'alert': COMMON_NUMERIC + ALERT_ONLY, 'onset': COMMON_NUMERIC + ONSET_ONLY}
    analysis = ['primary_family', 'size_bin', 'outage_plausibility', 'trigger_partial_county', 'retro_noaa_event_in_window']
    ids = ['design', 'opportunity_id', 'county_fips', 't0', 'split', 'cluster_date']
    counts = {}
    for design, feats in features.items():
        cols = ids + COMMON_CATEGORICAL + feats + analysis
        for split in ('train', 'validation'):
            copy_query(con, f"SELECT {', '.join(cols + OUTCOMES)} FROM dataset WHERE design = '{design}' AND split = '{split}' ORDER BY t0, county_fips",
                       OUT / f'{design}_{split}.parquet')
        copy_query(con, f"SELECT {', '.join(cols)} FROM dataset WHERE design = '{design}' AND split = 'test' ORDER BY t0, county_fips",
                   OUT / f'{design}_test_features.parquet')
        counts[design] = {
            'rows': dict(con.execute(f"SELECT split, count(*) FROM dataset WHERE design = '{design}' GROUP BY 1 ORDER BY 1").fetchall()),
            'labels': {f'{s}/{l}': n for s, l, n in con.execute(f"""SELECT split, label, count(*) FROM dataset
                WHERE design = '{design}' AND split <> 'test' GROUP BY ALL ORDER BY ALL""").fetchall()},
            'covered_windows': {f'{s}/{l}': n for s, l, n in con.execute(f"""SELECT split, label, count(*) FROM dataset
                WHERE design = '{design}' AND covered_window GROUP BY ALL ORDER BY ALL""").fetchall()},
            'excluded': {f'{s}/{e}': n for s, e, n in con.execute(f"""SELECT split, eligibility, count(*) FROM rows_all
                WHERE design = '{design}' AND eligibility <> 'eligible' GROUP BY ALL ORDER BY ALL""").fetchall()},
        }
    manifest = {
        'version': VERSION,
        'target': {'major_customers': MAJOR_CUSTOMERS, 'non_major_rule': PRIMARY_RULE,
                   'rule_sql': {d: non_major_rule(PRIMARY_RULE, h) for d, h in WINDOW_HOURS.items()},
                   'ongoing': 'excluded from the primary target (label = ongoing)',
                   'window_hours': WINDOW_HOURS, 'window': '[t0, t0 + window_hours)'},
        'evaluation_population': 'covered_window: cov90 coverage and not ongoing (symmetric, decision D-22)',
        'severity': 'provisional: peak_customers where covered_window',
        'features': features, 'categorical': COMMON_CATEGORICAL, 'analysis_only': analysis, 'outcomes': OUTCOMES,
        'counts': counts,
        'test_policy': 'test files hold features only; outcomes are computed once by the frozen final evaluation',
        'inputs_sha256': {name: file_sha256(path) for name, path in inputs.items()},
        'decisions': ['D-16', 'D-17', 'D-18', 'D-19', 'D-20', 'D-21', 'D-22', 'D-23', 'D-24'],
    }
    write_json(manifest, OUT / 'manifest.json')
    print(json.dumps(counts, indent=2), flush=True)
    con.close()


if __name__ == '__main__':
    main()
