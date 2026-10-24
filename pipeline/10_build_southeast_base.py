"""Build the six-state base tables for the major-outage pipeline.

Run: python pipeline/10_build_southeast_base.py [--force | --check]   (after stages 04, 08, 09)
Outputs in data/03_cleaned/southeast/, each with a .meta.json sidecar (see common.ParquetCache):
  eaglei_southeast.parquet        every EAGLE-I row for AL, FL, GA, NC, SC and TN
  county_reference.parquet        county names, EAGLE-I tracking span, customer denominators
  storm_events_southeast.parquet  one row per NOAA event and county (county-coded or zone-mapped)
  onset_opportunities.parquet     one row per county prediction opportunity at recorded onset
plus reports/label_audit/denominator_validation.csv and southeast_base_manifest.json.

Differences from the national cleaning v3 outputs, which the preliminary baseline keeps using:
- Dec 31 observations are kept. The 2014-2021 source files end at Dec 31 00:00 UTC, so
  those years simply have no rows for the rest of that day; window coverage rules treat
  the missing runs as unknown instead of a blanket date exclusion.
- total_customers is kept as reported (populated in 2024 only) and validated with the
  modeled county customers (MCC) before any percentage is computed.
- No exclusion by realized storm duration: duration is not known at prediction time.
- Zone events are included through the stage 09 crosswalk, and county-coded events must
  pass its name check (miscoded zone records are remapped or dropped there).
- The county list, names and population come from the Census, the independent anchor
  for validating customer denominators (MCC and EAGLE-I total_customers) and for flagging
  counties whose outage peaks exceed their customers.

Onset opportunities: within each county, events in the provisional storm scope
(common.STORM_GROUPS) are processed in time order. The first event starts an opportunity
at t0 = its UTC begin time; every event beginning before t0 + 24 h belongs to that
opportunity's outcome window, and the next opportunity starts at the first later event.
Each county outcome window is therefore counted once, however many NOAA records overlap.
Only trigger events (begin = t0) describe the opportunity; later events are prefixed
retro_ and are audit fields, never predictors.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common import (SE_STATE_FIPS, SOUTHEAST_STATES, STORM_GROUPS, ParquetCache, combined_csv, connect,
                    copy_query, hazard_group_sql, lit, split_sql, sql_list, write_json)
from paths import CENSUS_COUNTY_POP, CLEANED, COMBINED, COVERAGE_HISTORY, LABEL_REPORTS, MCC, SOUTHEAST

VERSION = 1
EAGLEI_START = '2014-11-01 04:00:00+00'   # first EAGLE-I timestamp; earlier onsets have no outcome data
# Denominator validation thresholds.
STATE_TOTAL_TOLERANCE = 0.15   # state sum of county customers vs EIA-861 state customers (coverage_history.csv, 2022)
PER_PERSON = (0.25, 1.2)       # plausible electric customers per 2020 resident; seasonal-home counties run high
TOTAL_STABILITY = 1.2          # max/min of a county's 2024 EAGLE-I total_customers
PEAK_TOLERANCE = 1.05          # observed customers_out may not exceed a valid denominator by more than this


def opportunities_from(events):
    """Greedy, non-overlapping 24 h county windows from time-ordered events."""
    rows = []
    for county, group in events.sort_values(['county_fips', 'begin_utc', 'event_id']).groupby('county_fips', sort=False):
        begins = group.begin_utc.to_numpy()
        start = 0
        while start < len(group):
            t0 = begins[start]
            stop = start + np.searchsorted(begins[start:], t0 + np.timedelta64(24, 'h'), side='left')
            window = group.iloc[start:stop]
            trigger = window[window.begin_utc == t0]
            rows.append({
                'opportunity_id': f"{county}_{pd.Timestamp(t0).strftime('%Y%m%dT%H%M')}",
                'county_fips': county, 't0': t0,
                'trigger_event_ids': sorted(trigger.event_id.tolist()),
                'trigger_event_types': sorted(set(trigger.event_type)),
                'trigger_hazard_groups': sorted(set(trigger.hazard_group)),
                'trigger_from_zone': bool((trigger.mapping == 'zone_crosswalk').any()),
                'trigger_partial_county': bool(trigger.partial_county.eq(True).any()),
                'episode_ids': sorted(set(window.episode_id.dropna().astype('int64').tolist())),
                'retro_events_in_window': len(window),
                'retro_event_types_in_window': sorted(set(window.event_type)),
                'retro_hazard_groups_in_window': sorted(set(window.hazard_group)),
            })
            start = stop
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--force', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    SOUTHEAST.mkdir(parents=True, exist_ok=True)
    LABEL_REPORTS.mkdir(parents=True, exist_ok=True)
    con = connect()
    cache = ParquetCache(con, VERSION, 'southeast_version', force=args.force, check=args.check)
    details_meta = CLEANED / 'storm_details.meta.json'
    zone_map = SOUTHEAST / 'zone_event_counties.parquet'
    county_checks = SOUTHEAST / 'county_event_checks.parquet'
    eagle_path = SOUTHEAST / 'eaglei_southeast.parquet'
    counties_path = SOUTHEAST / 'county_reference.parquet'
    events_path = SOUTHEAST / 'storm_events_southeast.parquet'
    opps_path = SOUTHEAST / 'onset_opportunities.parquet'

    print('Typing Southeast EAGLE-I rows (Dec 31 and total_customers retained)', flush=True)
    eagle_sql = f"""WITH typed AS (SELECT
            try_cast(SOURCE_YEAR AS INTEGER) AS source_year,
            fips_code AS fips_code_raw, run_start_time AS timestamp_raw,
            padded_code(fips_code, 5) AS county_fips,
            trim(county) AS county, trim(state) AS state,
            try_cast(run_start_time AS TIMESTAMP) AT TIME ZONE 'UTC' AS observation_utc,
            try_cast(customers_out AS BIGINT) AS customers_out,
            customers_out IS NOT NULL AND try_cast(customers_out AS BIGINT) IS NULL AS customers_out_unparsed,
            try_cast(total_customers AS BIGINT) AS total_customers,
            total_customers IS NOT NULL AND try_cast(total_customers AS BIGINT) IS NULL AS total_customers_unparsed
        FROM {combined_csv('eaglei_outages_2014_2025.csv')}
        WHERE substr(padded_code(fips_code, 5), 1, 2) IN {sql_list(SE_STATE_FIPS)})
        SELECT *, CAST(observation_utc AS DATE) AS obs_date, on_slot(observation_utc) AS on_slot,
            coalesce(customers_out = 0, false) AS recorded_zero,
            -- Source-file conventions observed in reports/label_audit/coverage_state_year.csv.
            CASE WHEN source_year <= 2016 THEN 'zeros_omitted'
                 WHEN source_year <= 2021 THEN 'zeros_omitted_nulls_present'
                 WHEN source_year <= 2024 THEN 'zeros_intermittent'
                 ELSE 'nulls_present_zeros_rare' END AS reporting_convention
        FROM typed"""
    cache.build(eagle_sql, eagle_path, sources=[COMBINED / 'eaglei_outages_2014_2025.csv'])

    if not args.check:
        con.execute(f'CREATE VIEW eaglei AS SELECT * FROM read_parquet({lit(eagle_path)})')
        dupes, unparsed, bad_keys = con.execute("""SELECT count(*) - count(DISTINCT (county_fips, observation_utc)),
            count(*) FILTER (WHERE customers_out_unparsed OR total_customers_unparsed),
            count(*) FILTER (WHERE county_fips IS NULL OR observation_utc IS NULL) FROM eaglei""").fetchone()
        assert dupes == 0, f'{dupes} duplicate (county_fips, observation_utc) keys'
        assert unparsed == 0, f'{unparsed} non-numeric customer values'
        assert bad_keys == 0, f'{bad_keys} rows without a county or timestamp'

    print('County reference and denominator validation', flush=True)
    # The Census county list is authoritative; utility-derived customer counts are validated against it.
    counties_sql = f"""WITH census AS (SELECT STATE || COUNTY AS county_fips, regexp_replace(CTYNAME, ' County$', '') AS county_name,
                try_cast(ESTIMATESBASE2020 AS BIGINT) AS population_2020, try_cast(POPESTIMATE2023 AS BIGINT) AS population_2023
            FROM read_csv({lit(CENSUS_COUNTY_POP)}, all_varchar=true, encoding='latin-1')
            WHERE SUMLEV = '050' AND STATE IN {sql_list(SE_STATE_FIPS)}),
        tracked AS (SELECT county_fips, any_value(county) AS eaglei_name,
                min(observation_utc) AS first_observation_utc, max(observation_utc) AS last_observation_utc,
                count(*) AS eaglei_rows, max(customers_out) AS max_customers_out_any_year,
                min(total_customers) FILTER (WHERE source_year = 2024) AS total_2024_min,
                median(total_customers) FILTER (WHERE source_year = 2024) AS total_2024_median,
                max(total_customers) FILTER (WHERE source_year = 2024) AS total_2024_max,
                count(*) FILTER (WHERE source_year = 2024 AND customers_out > total_customers) AS rows_2024_out_above_total
            FROM read_parquet({lit(eagle_path)}) GROUP BY 1),
        mcc AS (SELECT padded_code(County_FIPS, 5) AS county_fips, try_cast(Customers AS BIGINT) AS mcc_customers
            FROM read_csv({lit(MCC)}, header=true, all_varchar=true) WHERE padded_code(County_FIPS, 5) IS NOT NULL),
        eia AS (SELECT state AS abbr, try_cast(total_customers AS BIGINT) AS eia_customers_2022,
                try_cast(max_covered AS BIGINT) AS eia_max_covered_2022
            FROM read_csv({lit(COVERAGE_HISTORY)}, header=true, all_varchar=true) WHERE year = '1/1/22'),
        states AS (SELECT * FROM (VALUES {', '.join(f"('{f}', '{a}', '{n}')" for n, (f, a) in SOUTHEAST_STATES.items())}) s(state_fips, abbr, state_name)),
        joined AS (SELECT c.county_fips, s.abbr AS state, s.state_name, c.county_name, c.population_2020, c.population_2023,
                t.* EXCLUDE (county_fips), t.county_fips IS NOT NULL AS eaglei_tracked,
                m.mcc_customers, e.eia_customers_2022, e.eia_max_covered_2022,
                sum(m.mcc_customers) OVER (PARTITION BY s.abbr) AS mcc_state_sum,
                sum(t.total_2024_median) OVER (PARTITION BY s.abbr) AS eaglei_total_state_sum
            FROM census c JOIN states s ON s.state_fips = substr(c.county_fips, 1, 2)
            LEFT JOIN tracked t USING (county_fips) LEFT JOIN mcc m USING (county_fips) LEFT JOIN eia e ON e.abbr = s.abbr),
        rated AS (SELECT *,
            mcc_state_sum / eia_customers_2022 AS mcc_state_ratio,
            eaglei_total_state_sum / eia_customers_2022 AS eaglei_total_state_ratio,
            mcc_customers / population_2020 AS mcc_per_person,
            total_2024_median / population_2020 AS eaglei_total_per_person,
            CASE WHEN mcc_customers IS NULL THEN 'missing'
                 WHEN abs(mcc_state_sum / eia_customers_2022 - 1) > {STATE_TOTAL_TOLERANCE} THEN 'state_total_mismatch'
                 WHEN NOT mcc_customers / population_2020 BETWEEN {PER_PERSON[0]} AND {PER_PERSON[1]} THEN 'implausible_per_person'
                 ELSE 'valid' END AS mcc_status,
            CASE WHEN total_2024_median IS NULL THEN 'missing'
                 WHEN total_2024_max > {TOTAL_STABILITY} * total_2024_min THEN 'unstable_2024'
                 WHEN rows_2024_out_above_total > 0 THEN 'outages_exceed_total'
                 WHEN abs(eaglei_total_state_sum / eia_customers_2022 - 1) > {STATE_TOTAL_TOLERANCE} THEN 'state_total_mismatch'
                 WHEN NOT total_2024_median / population_2020 BETWEEN {PER_PERSON[0]} AND {PER_PERSON[1]} THEN 'implausible_per_person'
                 ELSE 'valid' END AS eaglei_total_status
            FROM joined)
        SELECT *, total_2024_median / mcc_customers AS eaglei_total_to_mcc,
            -- Outcome plausibility: a peak above the county's customers points to outage double counting or misassignment.
            CASE WHEN max_customers_out_any_year IS NULL THEN 'no_outage_rows'
                 WHEN mcc_status = 'valid' AND max_customers_out_any_year > {PEAK_TOLERANCE} * mcc_customers THEN 'peak_exceeds_mcc'
                 WHEN max_customers_out_any_year > population_2020 THEN 'peak_exceeds_population'
                 WHEN mcc_status = 'valid' THEN 'ok' ELSE 'unchecked' END AS outage_plausibility
        FROM rated"""
    cache.build(counties_sql, counties_path, sources=[MCC, COVERAGE_HISTORY, CENSUS_COUNTY_POP], upstream=[eagle_path.with_suffix('.meta.json')])
    if not args.check:
        stray = con.execute(f"""SELECT DISTINCT county_fips FROM read_parquet({lit(eagle_path)})
            WHERE county_fips NOT IN (SELECT county_fips FROM read_parquet({lit(counties_path)}))""").fetchall()
        assert not stray, f'EAGLE-I counties missing from the Census list: {stray}'

    print('Southeast storm events by county (verified county codes plus zone-mapped)', flush=True)
    events_sql = f"""WITH details AS (SELECT * FROM read_parquet({lit(CLEANED / 'storm_details.parquet')})
            WHERE STATE IN {sql_list(SOUTHEAST_STATES)}),
        county_coded AS (SELECT d.event_id_clean AS event_id, d.county_fips, 'county_code' AS mapping,
                NULL::VARCHAR AS zone_match_status, NULL::VARCHAR AS state_zone, NULL::BOOLEAN AS partial_county,
                NULL::BIGINT AS zones_in_county, NULL::BIGINT AS counties_in_zone
            FROM details d JOIN read_parquet({lit(county_checks)}) k ON k.event_id = d.event_id_clean
            WHERE d.CZ_TYPE = 'C' AND k.county_check = 'county_code_verified'),
        zone_mapped AS (SELECT event_id, county_fips,
                CASE WHEN source_cz_type = 'C' THEN 'county_code_as_zone' ELSE 'zone_crosswalk' END,
                match_status, mapped_zone, partial_county, zones_in_county, counties_in_zone
            FROM read_parquet({lit(zone_map)})),
        rows AS (SELECT * FROM county_coded UNION ALL SELECT * FROM zone_mapped)
        SELECT r.*, d.episode_id_clean AS episode_id, d.STATE AS state_name, d.EVENT_TYPE AS event_type,
            {hazard_group_sql('d.EVENT_TYPE')} AS hazard_group, d.CZ_TYPE AS cz_type, d.CZ_NAME AS cz_name, d.WFO AS wfo,
            d.begin_utc, d.CZ_TIMEZONE AS cz_timezone, d.time_status, {split_sql('d.begin_utc')} AS split,
            -- Realized after onset: audit only, never predictors.
            d.end_utc AS retro_end_utc, d.duration_minutes AS retro_duration_minutes, d.magnitude_value AS retro_magnitude
        FROM rows r JOIN details d ON d.event_id_clean = r.event_id"""
    cache.build(events_sql, events_path, sources=[zone_map, county_checks], upstream=[details_meta])
    if not args.check:
        dupes, unknown = con.execute(f"""SELECT count(*) - count(DISTINCT (event_id, county_fips)),
            count(*) FILTER (WHERE county_fips NOT IN (SELECT county_fips FROM read_parquet({lit(counties_path)})))
            FROM read_parquet({lit(events_path)})""").fetchone()
        assert dupes == 0 and unknown == 0, f'{dupes} duplicate event-county rows, {unknown} rows outside the Census county list'

    print('Onset opportunities: one non-overlapping 24 h window per county storm onset', flush=True)
    opps_query = 'SELECT * FROM onset_df ORDER BY county_fips, t0'
    if cache.plan(opps_query, opps_path, sources=[__file__], upstream=[events_path.with_suffix('.meta.json'), counties_path.with_suffix('.meta.json')]) and not args.check:
        events = con.execute(f"""SELECT event_id, county_fips, episode_id, event_type, hazard_group, mapping, partial_county, begin_utc
            FROM read_parquet({lit(events_path)})
            WHERE hazard_group IN {sql_list(STORM_GROUPS)} AND time_status = 'ok' AND begin_utc >= TIMESTAMPTZ '{EAGLEI_START}'""").df()
        events['begin_utc'] = events.begin_utc.dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[us]')
        onset = opportunities_from(events)
        onset['t0'] = pd.to_datetime(onset.t0).dt.tz_localize('UTC')
        con.register('onset_raw', onset)
        # Splits and leakage guards: windows touching a split boundary and episodes spanning splits are quarantined.
        con.execute(f"""CREATE TEMP TABLE onset_df AS
            WITH ep AS (SELECT episode_id, count(DISTINCT split) > 1 AS cross_split
                        FROM read_parquet({lit(events_path)}) WHERE episode_id IS NOT NULL GROUP BY 1),
            opp_ep AS (SELECT opportunity_id, unnest(episode_ids) AS episode_id FROM onset_raw),
            crossing AS (SELECT opportunity_id, bool_or(ep.cross_split) AS cross_split FROM opp_ep JOIN ep USING (episode_id) GROUP BY 1),
            opp AS (SELECT o.*, {split_sql('o.t0')} AS split, o.t0 + INTERVAL 24 HOUR AS window_end FROM onset_raw o)
            SELECT o.*, c.state, c.eaglei_tracked, coalesce(x.cross_split, false) AS cross_split_episode,
                (o.split = 'train' AND o.window_end > TIMESTAMPTZ '2024-01-01 00:00:00+00')
                  OR (o.split = 'validation' AND (o.t0 - INTERVAL 24 HOUR < TIMESTAMPTZ '2024-01-01 00:00:00+00' OR o.window_end > TIMESTAMPTZ '2025-01-01 00:00:00+00'))
                  OR (o.split = 'test' AND (o.t0 - INTERVAL 24 HOUR < TIMESTAMPTZ '2025-01-01 00:00:00+00' OR o.window_end > TIMESTAMPTZ '2026-01-01 00:00:00+00')) AS split_boundary_window
            FROM opp o JOIN read_parquet({lit(counties_path)}) c USING (county_fips) LEFT JOIN crossing x USING (opportunity_id)""")
        assert con.execute('SELECT count(*) - count(DISTINCT opportunity_id) FROM onset_df').fetchone()[0] == 0
        overlap = con.execute("""SELECT count(*) FROM (SELECT county_fips, t0, lead(t0) OVER (PARTITION BY county_fips ORDER BY t0) AS next_t0
            FROM onset_df) WHERE next_t0 < t0 + INTERVAL 24 HOUR""").fetchone()[0]
        assert overlap == 0, f'{overlap} overlapping county windows'
        cache.build(opps_query, opps_path, sources=[__file__], upstream=[events_path.with_suffix('.meta.json'), counties_path.with_suffix('.meta.json')])
    elif not args.check:
        print(f'Reusing {opps_path.name} (southeast_version {VERSION}, hash matches)', flush=True)

    if args.check:
        for name, status in cache.status.items():
            print(f'{name}: {status}', flush=True)
        return

    copy_query(con, f"""SELECT county_fips, state, county_name, eaglei_tracked, population_2020, mcc_customers,
            round(mcc_per_person, 3) AS mcc_per_person, round(mcc_state_ratio, 3) AS mcc_state_ratio, mcc_status,
            total_2024_min, total_2024_median, total_2024_max, rows_2024_out_above_total,
            round(eaglei_total_per_person, 3) AS eaglei_total_per_person, round(eaglei_total_state_ratio, 3) AS eaglei_total_state_ratio,
            eaglei_total_status, round(eaglei_total_to_mcc, 3) AS eaglei_total_to_mcc,
            max_customers_out_any_year, outage_plausibility
        FROM read_parquet({lit(counties_path)}) ORDER BY county_fips""", LABEL_REPORTS / 'denominator_validation.csv', fmt='csv')
    copy_query(con, f"""SELECT state, count(*) AS counties, count(*) FILTER (WHERE eaglei_tracked) AS eaglei_tracked,
            any_value(eia_customers_2022) AS eia_customers_2022, sum(population_2020) AS population_2020,
            any_value(mcc_state_sum) AS mcc_sum, round(any_value(mcc_state_ratio), 3) AS mcc_to_eia,
            any_value(eaglei_total_state_sum) AS eaglei_total_2024_sum, round(any_value(eaglei_total_state_ratio), 3) AS eaglei_total_to_eia,
            count(*) FILTER (WHERE mcc_status = 'valid') AS mcc_valid, count(*) FILTER (WHERE eaglei_total_status = 'valid') AS eaglei_total_valid,
            count(*) FILTER (WHERE outage_plausibility IN ('peak_exceeds_mcc', 'peak_exceeds_population')) AS counties_peak_exceeds_customers
        FROM read_parquet({lit(counties_path)}) GROUP BY 1 ORDER BY 1""", LABEL_REPORTS / 'denominator_validation_by_state.csv', fmt='csv')
    summary = {
        'southeast_version': VERSION, 'outputs': cache.meta,
        'eaglei_rows_by_year': dict(con.execute(f'SELECT source_year, count(*) FROM read_parquet({lit(eagle_path)}) GROUP BY 1 ORDER BY 1').fetchall()),
        'counties': dict(con.execute(f'SELECT state, count(*) FROM read_parquet({lit(counties_path)}) GROUP BY 1 ORDER BY 1').fetchall()),
        'untracked_counties': [r[0] for r in con.execute(f'SELECT county_fips FROM read_parquet({lit(counties_path)}) WHERE NOT eaglei_tracked ORDER BY 1').fetchall()],
        'denominator_status': {
            'mcc': dict(con.execute(f'SELECT mcc_status, count(*) FROM read_parquet({lit(counties_path)}) GROUP BY 1 ORDER BY 1').fetchall()),
            'eaglei_total_2024': dict(con.execute(f'SELECT eaglei_total_status, count(*) FROM read_parquet({lit(counties_path)}) GROUP BY 1 ORDER BY 1').fetchall()),
            'outage_plausibility': dict(con.execute(f'SELECT outage_plausibility, count(*) FROM read_parquet({lit(counties_path)}) GROUP BY 1 ORDER BY 1').fetchall())},
        'event_rows_by_mapping': dict(con.execute(f'SELECT mapping, count(*) FROM read_parquet({lit(events_path)}) GROUP BY 1').fetchall()),
        'events_excluded_from_opportunities': dict(con.execute(f"""SELECT CASE WHEN hazard_group NOT IN {sql_list(STORM_GROUPS)} THEN 'outside provisional storm scope: ' || hazard_group
                WHEN time_status <> 'ok' THEN 'time review: ' || time_status ELSE 'before EAGLE-I start' END, count(*)
            FROM read_parquet({lit(events_path)})
            WHERE hazard_group NOT IN {sql_list(STORM_GROUPS)} OR time_status <> 'ok' OR begin_utc < TIMESTAMPTZ '{EAGLEI_START}'
            GROUP BY 1 ORDER BY 2 DESC""").fetchall()),
        'opportunities_by_split': dict(con.execute(f'SELECT split, count(*) FROM read_parquet({lit(opps_path)}) GROUP BY 1 ORDER BY 1').fetchall()),
        'opportunity_rule': 'greedy non-overlapping 24 h windows per county from storm-scope event onsets',
        'storm_scope_groups': list(STORM_GROUPS),
        'notes': ['Dec 31 retained; 2014-2021 source files end at Dec 31 00:00 UTC, leaving later runs absent.',
                  'No duration-based exclusion; duration is realized after onset.',
                  'retro_ columns are realized after t0 and are audit fields only.',
                  'Opportunities quarantined for modeling when split_boundary_window or cross_split_episode is true.'],
    }
    write_json(summary, SOUTHEAST / 'southeast_base_manifest.json')
    print(summary, flush=True)
    con.close()


if __name__ == '__main__':
    main()
