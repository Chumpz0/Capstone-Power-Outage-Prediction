"""Create typed join-preparation datasets; never modify the combined CSVs.

Run: python pipeline/04_clean_datasets.py [--force | --check]
Requires duckdb, data/reference/ct_region_to_county_weights.csv
(build it with pipeline/03_build_ct_crosswalk.py), and reports/full_profile/summary.json
(build it with pipeline/profile_full.py; row counts are checked against it).
Outputs: Parquet files in data/03_cleaned/, quality reports in reports/quality/.

Versioning: every Parquet output has a sidecar <name>.meta.json holding the
cleaning VERSION and a build hash (common.build_hash) of the query, the shared
SQL macros, the content of its source files, and its upstream outputs' sidecars.
The project root is replaced by a placeholder before hashing, so moving the
whole project folder does not force a rebuild.
An output is reused only when both match; otherwise it is rebuilt. Bump
VERSION whenever a cleaning rule changes meaning. --check reports which outputs
would be rebuilt without building anything.

No labels or final storm/outage join are created here. The national outputs keep
cleaning v3 semantics for the preliminary baseline; the Southeast pipeline
(stages 08-12) reads the combined CSV directly and revisits the Dec 31 rule.
"""
import argparse
import json
from pathlib import Path

from common import ParquetCache, combined_csv as csv_source, connect, lit, write_json
from paths import CLEANED, COMBINED, CT_WEIGHTS, PROFILE_REPORTS, QUALITY_REPORTS

VERSION = 3
# Events longer than this, and all Drought records (monthly summaries), are excluded from join preparation.
MAX_EVENT_HOURS = 72
DEC31 = 'month(observation_utc)=12 AND day(observation_utc)=31'
LONG_RUNNING = f"EVENT_TYPE='Drought' OR duration_minutes>{MAX_EVENT_HOURS * 60}"
TOUCHES_DEC31 = 'crosses_utc_year OR end_utc>=make_timestamptz(year(begin_utc)::BIGINT,12,31,0,0,0.0)'
SOURCES = {'storm_details': 'storm_events_details_all_years.csv',
           'storm_locations': 'storm_events_locations_all_years.csv',
           'storm_fatalities': 'storm_events_fatalities_all_years.csv',
           'eaglei_typed_full': 'eaglei_outages_2014_2025.csv'}


def setup(con):
    con.execute(f"""CREATE TABLE ct_weights AS SELECT region_fips, county_fips, weight::DOUBLE AS weight
        FROM read_csv({lit(CT_WEIGHTS)}, header=true, all_varchar=true)""")


def verify_rules(con):
    assert con.execute("SELECT padded_code('1001',5), padded_code('01001',5), padded_code('100001',5), padded_code('1.0',5)").fetchone() == ('01001','01001',None,None)
    # Fixed source offset applies in summer too; no automatic DST localization.
    assert str(con.execute("SELECT utc_from_offset(TIMESTAMP '2020-07-01 12:00:00','EST-5')").fetchone()[0]) == '2020-07-01 17:00:00+00:00'
    assert str(con.execute("SELECT utc_from_offset(TIMESTAMP '2020-01-01 00:00:00','GST10')").fetchone()[0]) == '2019-12-31 14:00:00+00:00'
    assert con.execute("SELECT utc_from_offset(TIMESTAMP '2020-01-01','unknown')").fetchone()[0] is None
    bad = con.execute('SELECT region_fips, sum(weight) FROM ct_weights GROUP BY 1 HAVING abs(sum(weight)-1)>1e-9').fetchall()
    assert not bad, f'Connecticut region weights must sum to 1: {bad}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=CLEANED)
    parser.add_argument('--reports', type=Path, default=QUALITY_REPORTS, help='Folder for quality report CSVs.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--force', action='store_true', help='Rebuild every Parquet output even when its version and hash match.')
    mode.add_argument('--check', action='store_true', help='Report which outputs are stale, then exit without building.')
    args = parser.parse_args()
    profile_summary = PROFILE_REPORTS / 'summary.json'
    if not args.check and not profile_summary.exists():
        raise SystemExit(f'{profile_summary} is missing. Run pipeline/profile_full.py first: '
                         'this stage checks its row counts against that full-file profile.')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reports = args.reports.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    con = connect()
    setup(con)
    verify_rules(con)
    cache = ParquetCache(con, VERSION, 'cleaning_version', force=args.force, check=args.check)

    def export(query, path, parquet=False):
        fmt = 'FORMAT PARQUET, COMPRESSION ZSTD' if parquet else 'FORMAT CSV, HEADER'
        con.execute(f'COPY ({query}) TO {lit(path)} ({fmt})')

    details_sql = f"""WITH source AS (SELECT * FROM {csv_source(SOURCES['storm_details'])}),
    parsed AS (SELECT *,
        try_cast(EVENT_ID AS BIGINT) AS event_id_clean,
        try_cast(EPISODE_ID AS BIGINT) AS episode_id_clean,
        CASE WHEN STATE='PUERTO RICO' AND STATE_FIPS='99' THEN '72'
             WHEN STATE='VIRGIN ISLANDS' AND STATE_FIPS='96' THEN '78'
             WHEN try_cast(STATE_FIPS AS INTEGER) BETWEEN 1 AND 56 THEN padded_code(STATE_FIPS,2)
        END AS state_fips_clean,
        padded_code(CZ_FIPS,3) AS cz_code_clean,
        try_strptime(BEGIN_DATE_TIME,'%d-%b-%y %H:%M:%S') AS begin_local,
        try_strptime(END_DATE_TIME,'%d-%b-%y %H:%M:%S') AS end_local
        FROM source),
    derived AS (SELECT *,
        CASE WHEN CZ_TYPE='C' AND STATE!='ALASKA' AND state_fips_clean IS NOT NULL AND cz_code_clean!='000'
             THEN state_fips_clean || cz_code_clean END AS county_fips,
        utc_from_offset(begin_local,CZ_TIMEZONE) AS begin_utc,
        utc_from_offset(end_local,CZ_TIMEZONE) AS end_utc
        FROM parsed),
    flagged AS (SELECT *,
        date_diff('minute',begin_utc,end_utc) AS duration_minutes,
        (extract(year FROM begin_utc)!=extract(year FROM end_utc)) AS crosses_utc_year
        FROM derived)
    SELECT *,
        CASE WHEN CZ_TYPE='Z' THEN 'zone_crosswalk_required'
             WHEN STATE='ALASKA' AND CZ_TYPE='C' THEN 'alaska_geography_review'
             WHEN CZ_TYPE!='C' THEN 'unsupported_area_type'
             WHEN county_fips IS NULL THEN 'territory_or_identifier_review'
             ELSE 'county_code_candidate' END AS geography_status,
        CASE WHEN begin_utc IS NULL OR end_utc IS NULL THEN 'invalid_or_unknown_time'
             WHEN end_utc<begin_utc THEN 'negative_duration'
             WHEN CZ_TIMEZONE IN ('PDT-7','EDT-4','CDT-5') THEN 'explicit_daylight_offset_review'
             ELSE 'ok' END AS time_status,
        coalesce({LONG_RUNNING}, false) AS long_running_event,
        -- EAGLE-I observations on Dec 31 (UTC) are dropped, so events touching that day lack full outage coverage.
        coalesce({TOUCHES_DEC31}, false) AS overlaps_dec31_gap,
        CASE WHEN {LONG_RUNNING} THEN 'long_running_event'
             WHEN {TOUCHES_DEC31} THEN 'overlaps_dec31_gap' END AS join_exclusion,
        try_cast(MAGNITUDE AS DOUBLE) AS magnitude_value,
        try_cast(TOR_LENGTH AS DOUBLE) AS tornado_length_value,
        try_cast(TOR_WIDTH AS DOUBLE) AS tornado_width_value
    FROM flagged"""
    support_sql = {}
    for kind in ['locations', 'fatalities']:
        extra = '''try_cast(LOCATION_INDEX AS INTEGER) AS location_index_clean,
            CASE WHEN try_cast(LATITUDE AS DOUBLE) BETWEEN -90 AND 90 THEN try_cast(LATITUDE AS DOUBLE) END AS latitude_clean,
            CASE WHEN try_cast(LONGITUDE AS DOUBLE) BETWEEN -180 AND 180 THEN try_cast(LONGITUDE AS DOUBLE) END AS longitude_clean''' if kind == 'locations' else '''try_cast(FATALITY_ID AS BIGINT) AS fatality_id_clean,
            try_strptime(FATALITY_DATE,'%m/%d/%Y %H:%M:%S') AS fatality_local,
            try_cast(FATALITY_AGE AS INTEGER) AS fatality_age_clean'''
        support_sql[kind] = f'SELECT *, try_cast(EVENT_ID AS BIGINT) AS event_id_clean, {extra} FROM {csv_source(SOURCES["storm_" + kind])}'
    # total_customers is not carried forward: it is populated only in 2024.
    typed_sql = f"""SELECT
        try_cast(SOURCE_YEAR AS INTEGER) AS source_year,
        fips_code AS fips_code_raw, run_start_time AS timestamp_raw,
        padded_code(fips_code,5) AS county_fips,
        trim(county) AS county, trim(state) AS state,
        try_cast(run_start_time AS TIMESTAMP) AT TIME ZONE 'UTC' AS observation_utc,
        try_cast(customers_out AS BIGINT) AS customers_out
        FROM {csv_source(SOURCES['eaglei_typed_full'])}"""
    # Connecticut: EAGLE-I switches from legacy counties to planning regions on 2025-05-29.
    # From the first planning-region observation onward, regions are allocated to legacy
    # counties by 2020 town population and the few overlapping legacy rows are dropped.
    # An allocated value is null unless every contributing region reported a value.
    regions = '(SELECT region_fips FROM ct_weights)'
    cutover = f'(SELECT min(observation_utc) FROM typed WHERE county_fips IN {regions} AND NOT ({DEC31}))'
    eagle_sql = f"""WITH base AS (SELECT * FROM typed WHERE NOT ({DEC31})),
        kept AS (SELECT *, false AS ct_allocated, NULL::BOOLEAN AS ct_allocation_complete FROM base
            WHERE coalesce(county_fips NOT IN {regions}, true)
              AND NOT coalesce(county_fips LIKE '09%' AND observation_utc>={cutover}, false)),
        legacy_names AS (SELECT county_fips, any_value(county) AS county FROM base
            WHERE county_fips IN (SELECT county_fips FROM ct_weights) GROUP BY 1),
        expected AS (SELECT county_fips, count(*) AS regions FROM ct_weights GROUP BY 1),
        allocated AS (SELECT any_value(b.source_year) AS source_year,
                string_agg(b.fips_code_raw, '+' ORDER BY b.fips_code_raw) AS fips_code_raw,
                any_value(b.timestamp_raw) AS timestamp_raw,
                w.county_fips, any_value(b.state) AS state, b.observation_utc,
                CASE WHEN count(b.customers_out)=count(*) THEN round(sum(w.weight*b.customers_out))::BIGINT END AS customers_out,
                count(*) AS regions_present
            FROM base b JOIN ct_weights w ON b.county_fips=w.region_fips
            GROUP BY w.county_fips, b.observation_utc),
        combined AS (SELECT source_year, fips_code_raw, timestamp_raw, county_fips, county, state, observation_utc,
                customers_out, ct_allocated, ct_allocation_complete FROM kept
            UNION ALL
            SELECT a.source_year, a.fips_code_raw, a.timestamp_raw, a.county_fips, n.county, a.state, a.observation_utc,
                a.customers_out, true, a.customers_out IS NOT NULL AND a.regions_present=e.regions
            FROM allocated a JOIN legacy_names n USING (county_fips) JOIN expected e USING (county_fips))
    SELECT *, county_fips IS NOT NULL AND observation_utc IS NOT NULL AS join_key_valid,
        customers_out IS NOT NULL AND customers_out>=0 AS target_observed,
        -- 2014-2021 omit zero-outage rows; 2022+ record them. Flag so the convention can be harmonized.
        coalesce(customers_out=0, false) AS recorded_zero,
        CASE WHEN source_year=2025 THEN 'holdout_2025'
             WHEN source_year=2024 THEN 'validation_candidate_2024' ELSE 'training_candidate' END AS split_role
    FROM combined"""

    details_path = out / 'storm_details.parquet'
    typed_path = out / 'eaglei_typed_full.parquet'
    eagle_path = out / 'eaglei.parquet'
    plan = [(details_sql, details_path, [COMBINED / SOURCES['storm_details']], []),
            *[(support_sql[k], out / f'storm_{k}.parquet', [COMBINED / SOURCES['storm_' + k]], []) for k in support_sql],
            (typed_sql, typed_path, [COMBINED / SOURCES['eaglei_typed_full']], []),
            (eagle_sql, eagle_path, [CT_WEIGHTS], [typed_path.with_suffix('.meta.json')])]
    if args.check:
        for query, path, sources, upstream in plan:
            cache.build(query, path, sources, upstream)
        for name, status in cache.status.items():
            print(f'{name}: {status}', flush=True)
        con.close()
        return

    print('Cleaning NOAA details and supporting tables', flush=True)
    cache.build(*plan[0])
    con.execute(f'CREATE VIEW storms AS SELECT * FROM read_parquet({lit(details_path)})')
    assert con.execute("SELECT count(*) FROM storms WHERE CZ_TYPE!='C' AND county_fips IS NOT NULL").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM storms WHERE STATE='ALASKA' AND county_fips IS NOT NULL").fetchone()[0] == 0
    assert con.execute('SELECT count(*)-count(DISTINCT event_id_clean) FROM storms').fetchone()[0] == 0
    for kind, (query, path, sources, upstream) in zip(support_sql, plan[1:3]):
        cache.build(query, path, sources, upstream)
        con.execute(f'CREATE VIEW {kind} AS SELECT * FROM read_parquet({lit(path)})')
        export(f'SELECT s.* FROM {kind} s ANTI JOIN storms d ON s.event_id_clean=d.event_id_clean', reports / f'{kind}_orphan_events.csv')

    print('Typing full EAGLE-I file (all raw rows) with standardized county codes and UTC timestamps', flush=True)
    cache.build(*plan[3])
    con.execute(f'CREATE VIEW typed AS SELECT * FROM read_parquet({lit(typed_path)})')

    print('Applying Dec 31 exclusion and Connecticut planning-region consolidation', flush=True)
    cache.build(*plan[4])
    con.execute(f'CREATE VIEW outages AS SELECT * FROM read_parquet({lit(eagle_path)})')

    print('Auditing row accounting, identifiers, timestamps, and county coverage', flush=True)
    acct = dict(zip(['raw_rows', 'dec31_rows', 'ct_region_rows', 'ct_legacy_after_cutover_rows'], con.execute(f"""SELECT count(*),
        count(*) FILTER (WHERE {DEC31}),
        count(*) FILTER (WHERE county_fips IN {regions} AND NOT ({DEC31})),
        count(*) FILTER (WHERE county_fips LIKE '09%' AND county_fips NOT IN {regions} AND observation_utc>={cutover} AND NOT ({DEC31}))
        FROM typed""").fetchone()))
    acct['ct_allocated_rows'], acct['final_rows'] = con.execute('SELECT count(*) FILTER (WHERE ct_allocated), count(*) FROM outages').fetchone()
    assert acct['final_rows'] == acct['raw_rows'] - acct['dec31_rows'] - acct['ct_region_rows'] - acct['ct_legacy_after_cutover_rows'] + acct['ct_allocated_rows'], acct
    assert con.execute("SELECT count(*) - count(DISTINCT (county_fips, observation_utc)) FROM outages WHERE county_fips LIKE '09%'").fetchone()[0] == 0, 'Connecticut key collisions after consolidation'
    assert con.execute(f'SELECT count(*) FROM outages WHERE {DEC31}').fetchone()[0] == 0

    # Normalization must be injective on source codes and timestamps. Allocated CT rows carry a composite raw code, so they are excluded here.
    con.execute('CREATE TABLE fips_map AS SELECT DISTINCT fips_code_raw,county_fips FROM outages WHERE NOT ct_allocated')
    con.execute('CREATE TABLE time_map AS SELECT DISTINCT timestamp_raw,observation_utc FROM outages WHERE NOT ct_allocated')
    export('SELECT county_fips, count(*) AS raw_forms FROM fips_map GROUP BY county_fips HAVING count(*)>1', reports / 'fips_normalization_collisions.csv')
    export('SELECT observation_utc,count(*) AS raw_forms FROM time_map GROUP BY observation_utc HAVING count(*)>1', reports / 'timestamp_normalization_collisions.csv')
    assert con.execute('SELECT count(*) FROM (SELECT county_fips FROM fips_map GROUP BY 1 HAVING count(*)>1)').fetchone()[0] == 0, 'Normalized FIPS collisions require rechecking full record keys'
    assert con.execute('SELECT count(*) FROM (SELECT observation_utc FROM time_map GROUP BY 1 HAVING count(*)>1)').fetchone()[0] == 0, 'Normalized timestamp collisions require rechecking full record keys'
    export(f'''WITH dec31 AS (SELECT source_year, count(*) AS dec31_rows_dropped FROM typed WHERE {DEC31} GROUP BY 1)
        SELECT source_year, count(*) AS rows, coalesce(any_value(d.dec31_rows_dropped),0) AS dec31_rows_dropped,
        count(*) FILTER (WHERE NOT join_key_valid) AS invalid_join_keys,
        count(*) FILTER (WHERE NOT target_observed) AS unknown_or_invalid_target,
        count(*) FILTER (WHERE recorded_zero) AS recorded_zero_rows,
        count(*) FILTER (WHERE ct_allocated) AS ct_allocated_rows,
        count(*) FILTER (WHERE extract(year FROM observation_utc)!=source_year) AS source_year_mismatches,
        min(observation_utc) AS first_utc, max(observation_utc) AS last_utc
        FROM outages LEFT JOIN dec31 d USING (source_year) GROUP BY source_year ORDER BY source_year''', reports / 'eaglei_cleaning_by_year.csv')
    con.execute('''CREATE TABLE county_year AS SELECT source_year,county_fips,count(*) AS observed_rows,
        count(customers_out) AS known_outage_rows,min(observation_utc) AS first_utc,max(observation_utc) AS last_utc
        FROM outages GROUP BY source_year,county_fips''')
    export('SELECT * FROM county_year ORDER BY source_year,county_fips', reports / 'eaglei_county_year_coverage.csv')

    # A year/county overlap is only a coverage screen, not evidence of an observation at storm time.
    con.execute('''CREATE TABLE storm_audit AS SELECT s.EVENT_ID,s.SOURCE_YEAR,s.EVENT_TYPE,s.STATE,s.CZ_NAME,s.CZ_TYPE,
        s.county_fips,s.begin_utc,s.end_utc,s.geography_status,s.time_status,s.join_exclusion,
        EXISTS(SELECT 1 FROM county_year e WHERE e.county_fips=s.county_fips) AS county_seen_in_eaglei,
        EXISTS(SELECT 1 FROM county_year e WHERE e.county_fips=s.county_fips
          AND e.first_utc<=s.end_utc AND e.last_utc>=s.begin_utc) AS overlaps_observation_envelope
        FROM storms s''')
    export('SELECT SOURCE_YEAR,join_exclusion,EVENT_TYPE,count(*) AS events FROM storm_audit WHERE join_exclusion IS NOT NULL GROUP BY ALL ORDER BY ALL', reports / 'storm_exclusions.csv')
    # Join preparation keeps only events that are neither long-running nor touching the Dec 31 gap.
    export('SELECT * EXCLUDE (join_exclusion) FROM storm_audit WHERE join_exclusion IS NULL', out / 'storm_join_audit.parquet', True)
    export('''SELECT SOURCE_YEAR,geography_status,time_status,count(*) AS events,
        count(*) FILTER (WHERE county_seen_in_eaglei) AS county_seen_in_eaglei,
        count(*) FILTER (WHERE overlaps_observation_envelope) AS overlaps_observation_envelope
        FROM storm_audit WHERE join_exclusion IS NULL GROUP BY ALL ORDER BY SOURCE_YEAR,geography_status,time_status''', reports / 'storm_join_readiness.csv')
    untracked = "geography_status='county_code_candidate' AND join_exclusion IS NULL AND NOT county_seen_in_eaglei"
    export(f'SELECT * FROM storm_audit WHERE {untracked}', reports / 'unmatched_county_events.csv')
    # Counties NOAA reports but EAGLE-I never observes; candidates to drop from the final modeling table.
    export(f'''SELECT STATE, county_fips, any_value(CZ_NAME) AS cz_name, count(*) AS events,
        min(SOURCE_YEAR) AS first_year, max(SOURCE_YEAR) AS last_year
        FROM storm_audit WHERE {untracked} GROUP BY STATE, county_fips ORDER BY events DESC''', reports / 'untracked_counties.csv')
    export("SELECT STATE,STATE_FIPS,CZ_TYPE,CZ_FIPS,CZ_NAME,count(*) AS events FROM storms WHERE geography_status!='county_code_candidate' GROUP BY ALL ORDER BY events DESC", reports / 'geography_review.csv')
    export("SELECT CZ_TIMEZONE,time_status,count(*) AS events FROM storms GROUP BY ALL ORDER BY events DESC", reports / 'timezone_review.csv')

    counts = {name: con.execute(f'SELECT count(*) FROM {name}').fetchone()[0] for name in ['storms', 'locations', 'fatalities', 'typed']}
    original = json.loads(profile_summary.read_text())
    for name, source in [('storms', 'storm_details'), ('locations', 'storm_locations'), ('fatalities', 'storm_fatalities'), ('typed', 'eaglei')]:
        assert counts[name] == original[source]['rows'], f'Unexpected row loss: {name}'
    rows = {'storms': counts['storms'], 'locations': counts['locations'], 'fatalities': counts['fatalities'],
            'outages_raw': counts['typed'], 'outages': acct['final_rows']}
    exclusions = dict(con.execute('SELECT join_exclusion,count(*) FROM storms WHERE join_exclusion IS NOT NULL GROUP BY 1').fetchall())
    untracked_n = con.execute(f'SELECT count(*), count(DISTINCT county_fips) FROM storm_audit WHERE {untracked}').fetchone()
    manifest = {'cleaning_version': VERSION, 'outputs': cache.meta, 'rows': rows, 'raw_files_unchanged': True,
        'join_grain': 'county_fips + UTC observation/event interval; final target window not yet selected',
        'geography_status_counts': dict(con.execute('SELECT geography_status,count(*) FROM storms GROUP BY 1').fetchall()),
        'time_status_counts': dict(con.execute('SELECT time_status,count(*) FROM storms GROUP BY 1').fetchall()),
        'storm_join_exclusions': exclusions,
        'untracked_county_events': untracked_n[0], 'untracked_counties': untracked_n[1],
        'eaglei_row_accounting': acct,
        'normalization_collisions': 0,
        'decisions': [
            'total_customers dropped: populated only in 2024.',
            'EAGLE-I observations on Dec 31 (UTC) dropped for every year: 2014-2021 files end at Dec 31 00:00 UTC.',
            'Storm events touching Dec 31 UTC are excluded from join preparation (join_exclusion=overlaps_dec31_gap).',
            f'Drought events and events longer than {MAX_EVENT_HOURS} hours are excluded from join preparation (join_exclusion=long_running_event).',
            'Connecticut planning regions (2025-05-29 onward) are allocated to legacy counties by 2020 town population; ct_allocated marks these rows.',
            'customers_out is kept as the outage target; recorded_zero flags zero rows, which exist only from 2022.',
            'County candidates never observed in EAGLE-I are listed in untracked_counties.csv for exclusion from the final model.'],
        'limitations': ['Zone records need a historically appropriate zone-to-county crosswalk.',
            'Alaska C records include forecast-area-style codes; county_fips is withheld pending geographic verification.',
            'County overlap is not proof of continuous coverage or observation at event time.',
            'Territory mappings: PR 99 -> 72 and VI 96 -> 78; other nonstandard regions remain unresolved.',
            'Connecticut allocation assumes outages are spread in proportion to population within each planning region.',
            'Zero-outage rows are absent before 2022 and present afterward; harmonize before defining a target.',
            'Daylight-offset NOAA labels are honored as recorded but flagged for review.',
            'Original NOAA impact/narrative fields are retained for audit, not approved predictive features.',
            'EAGLE-I UTC convention follows the source documentation; 2023-2025 schema continuity is assumed.',
            'No missing target filling, zero filling, clipping, modeling, or final label joins performed.'],
        'sources': ['https://www.weather.gov/media/directives/010_pdfs/pd01016005curr.pdf',
            'https://www.nature.com/articles/s41597-024-03095-5',
            'https://www.census.gov/library/reference/code-lists/ansi.2020.html',
            'https://www2.census.gov/geo/docs/reference/ct_change/ct_cou_to_cousub_crosswalk.xlsx',
            'https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/cities/totals/sub-est2023_9.csv']}
    write_json(manifest, out / 'cleaning_manifest.json')
    print(json.dumps(manifest, indent=2), flush=True)
    con.close()


if __name__ == '__main__':
    main()
