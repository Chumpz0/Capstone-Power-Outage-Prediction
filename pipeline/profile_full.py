"""Profile every row without changing source data.

Run: python pipeline/profile_full.py
Requires duckdb. Outputs go to reports/full_profile/.
DuckDB uses disk-backed staging and spill space for full-file duplicate checks.
"""
from pathlib import Path
import json
import time
import argparse
import duckdb

from paths import COMBINED, PROFILE_REPORTS as OUT

FILES = {
    'storm_details': ('storm_events_details_all_years.csv', ['EVENT_ID']),
    'storm_fatalities': ('storm_events_fatalities_all_years.csv', ['EVENT_ID', 'FATALITY_ID']),
    'storm_locations': ('storm_events_locations_all_years.csv', ['EVENT_ID', 'LOCATION_INDEX']),
    'eaglei': ('eaglei_outages_2014_2025.csv', ['fips_code', 'run_start_time']),
}

def ident(value):
    return '"' + value.replace('"', '""') + '"'

def literal(value):
    return "'" + str(value).replace("'", "''") + "'"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-eaglei', action='store_true', help='Resume an interrupted EAGLE-I duplicate scan from its existing staged database.')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(OUT / 'profile_work.duckdb'))
    con.execute("SET memory_limit='4GB'")
    con.execute('SET threads=2')
    con.execute('SET preserve_insertion_order=false')
    con.execute(f"SET temp_directory={literal(OUT / 'spill')}")
    summaries = json.loads((OUT / 'summary.json').read_text()) if args.resume_eaglei else {}

    def export(query, filename):
        con.execute(f'COPY ({query}) TO {literal(OUT / filename)} (HEADER, FORMAT CSV)')

    for name, (filename, key) in FILES.items():
        if args.resume_eaglei and name != 'eaglei':
            continue
        start = time.time()
        path = COMBINED / filename
        print(f'{name}: loading all rows', flush=True)
        # Preserve identifiers and source text. Interpret conventional NA markers as null.
        if not args.resume_eaglei:
            con.execute('DROP TABLE IF EXISTS data')
            con.execute(f'''CREATE TABLE data AS SELECT * FROM read_csv(
                {literal(path)}, header=true, all_varchar=true,
                nullstr=['', 'NA', 'N/A', 'NULL', 'null', 'NaN', 'nan', 'None', '<NA>', '#N/A'],
                strict_mode=true, ignore_errors=false)''')
        columns = [r[0] for r in con.execute('DESCRIBE data').fetchall()]
        rows = con.execute('SELECT count(*) FROM data').fetchone()[0]
        print(f'{name}: {rows:,} rows; profiling columns', flush=True)
        # All columns are explicitly stored as strings; numeric compatibility is assessed on all rows.
        profiles = []
        for column in columns:
            c = ident(column)
            stats = con.execute(f'''SELECT count({c}), count(DISTINCT {c}),
                count(*) FILTER (WHERE {c} IS NOT NULL AND trim({c})=''),
                count(*) FILTER (WHERE {c} != trim({c})),
                count(try_cast({c} AS DOUBLE)),
                min(try_cast({c} AS DOUBLE)), max(try_cast({c} AS DOUBLE)),
                count(*) FILTER (WHERE try_cast({c} AS DOUBLE)<0)
                FROM data''').fetchone()
            nonnull, distinct, blank, whitespace, numeric, minimum, maximum, negative = stats
            profiles.append({'column': column, 'rows': rows, 'non_null': nonnull,
                'null_count': rows-nonnull, 'null_pct': round(100*(rows-nonnull)/rows, 4) if rows else None,
                'distinct_non_null': distinct, 'whitespace_only': blank,
                'surrounding_whitespace': whitespace, 'numeric_parseable': numeric,
                'numeric_parse_failures': nonnull-numeric,
                'numeric_min': minimum, 'numeric_max': maximum, 'negative_numeric': negative,
                'storage_dtype': 'VARCHAR'})
        import csv
        with (OUT / f'{name}_columns.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(profiles[0]))
            writer.writeheader()
            writer.writerows(profiles)

        missing = ', '.join(f'count(*) FILTER (WHERE {ident(c)} IS NULL) AS {ident(c+"__null")}' for c in columns)
        export(f'SELECT SOURCE_YEAR, count(*) AS rows, {missing} FROM data GROUP BY SOURCE_YEAR ORDER BY SOURCE_YEAR', f'{name}_by_year.csv')
        for group in (['EVENT_TYPE', 'STATE', 'CZ_TYPE'] if name == 'storm_details' else ['state'] if name == 'eaglei' else []):
            export(f'SELECT {ident(group)}, count(*) AS rows, {missing} FROM data GROUP BY {ident(group)} ORDER BY rows DESC', f'{name}_by_{group.lower()}.csv')

        print(f'{name}: full-file duplicate checks', flush=True)
        duplicate_results = {}
        for label, fields in [('exact_rows', columns), ('candidate_keys', key)]:
            grouped = ', '.join(map(ident, fields))
            con.execute(f'CREATE OR REPLACE TABLE duplicate_groups AS SELECT {grouped}, 0::BIGINT AS occurrences FROM data WHERE false')
            # All equal rows/keys share fips_code, so hashing it partitions safely:
            # no duplicates can span buckets, and hash collisions only enlarge a bucket.
            partitions = 16 if name == 'eaglei' else 1
            for bucket in range(partitions):
                where = f'WHERE hash(fips_code) % {partitions} = {bucket}' if partitions > 1 else ''
                con.execute(f'INSERT INTO duplicate_groups SELECT {grouped}, count(*) AS occurrences FROM data {where} GROUP BY {grouped} HAVING count(*)>1')
                if partitions > 1 and (bucket+1) % 4 == 0:
                    print(f'{name}: {label} partitions {bucket+1}/{partitions}', flush=True)
            groups, extra, involved = con.execute('SELECT count(*), coalesce(sum(occurrences-1),0), coalesce(sum(occurrences),0) FROM duplicate_groups').fetchone()
            duplicate_results[label] = {'groups': groups, 'extra_rows': extra, 'all_involved_rows': involved}
            export('SELECT * FROM duplicate_groups ORDER BY occurrences DESC LIMIT 100', f'{name}_{label}_examples.csv')
        missing_keys = con.execute('SELECT count(*) FROM data WHERE ' + ' OR '.join(f'{ident(c)} IS NULL' for c in key)).fetchone()[0]
        checks = {}
        if name == 'eaglei':
            export('''SELECT SOURCE_YEAR, count(*) AS rows,
                min(try_cast(run_start_time AS TIMESTAMP)) AS earliest_timestamp,
                max(try_cast(run_start_time AS TIMESTAMP)) AS latest_timestamp,
                count(*) FILTER (WHERE run_start_time IS NOT NULL AND try_cast(run_start_time AS TIMESTAMP) IS NULL) AS invalid_timestamps,
                count(DISTINCT fips_code) AS distinct_counties,
                count(total_customers) AS total_customers_non_null,
                count(*) FILTER (WHERE try_cast(customers_out AS DOUBLE)>try_cast(total_customers AS DOUBLE)) AS outages_above_total,
                count(*) FILTER (WHERE try_cast(total_customers AS DOUBLE)<=0) AS nonpositive_totals
                FROM data GROUP BY SOURCE_YEAR ORDER BY SOURCE_YEAR''', 'eaglei_coverage.csv')
            checks['invalid_fips_format'] = con.execute("SELECT count(*) FROM data WHERE fips_code IS NOT NULL AND NOT regexp_full_match(fips_code, '[0-9]{4,5}')").fetchone()[0]
        if name == 'storm_details':
            parsed = "SELECT *, try_strptime(BEGIN_DATE_TIME, '%d-%b-%y %H:%M:%S') AS start_dt, try_strptime(END_DATE_TIME, '%d-%b-%y %H:%M:%S') AS end_dt FROM data"
            export(f'''SELECT SOURCE_YEAR, count(*) AS rows,
                min(start_dt) AS earliest_start, max(end_dt) AS latest_end,
                count(*) FILTER (WHERE BEGIN_DATE_TIME IS NOT NULL AND start_dt IS NULL) AS invalid_starts,
                count(*) FILTER (WHERE END_DATE_TIME IS NOT NULL AND end_dt IS NULL) AS invalid_ends,
                count(*) FILTER (WHERE end_dt<start_dt) AS negative_durations
                FROM ({parsed}) GROUP BY SOURCE_YEAR ORDER BY SOURCE_YEAR''', 'storm_details_dates.csv')
        summaries[name] = {'file': filename, 'file_bytes': path.stat().st_size,
            'rows': rows, 'columns': len(columns), 'candidate_key': key,
            'rows_missing_key_fields': missing_keys, 'duplicates': duplicate_results,
            'checks': checks, 'seconds': round(time.time()-start, 1)}
        (OUT / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
        print(f'{name}: complete in {time.time()-start:.1f}s; {duplicate_results}', flush=True)
    con.execute('DROP TABLE IF EXISTS duplicate_groups')
    con.execute('DROP TABLE IF EXISTS data')
    con.close()
    # Remove only the explicitly created, now-closed staging database.
    (OUT / 'profile_work.duckdb').unlink()
    (OUT / 'README.md').write_text('''# Full dataset quality profile

Generated by `pipeline/profile_full.py`. Every source row is read; malformed CSV rows cause an error rather than being skipped. Raw CSVs are unchanged.

- `summary.json`: row counts, exact duplicate counts, candidate-key repeats, and basic checks.
- `*_columns.csv`: full-file missing counts, distinct values, numeric compatibility/ranges, whitespace checks. Numeric statistics apply only to values that parse as numbers; identifiers are not treated as quantitative features.
- `*_by_year.csv`: year coverage and per-column missing counts.
- `storm_details_by_event_type.csv`: missingness conditional on storm type.
- `*_by_state.csv` and `storm_details_by_cz_type.csv`: geographic coverage and missingness.
- `*_examples.csv`: up to 100 duplicate groups, not the full duplicate population. Candidate-key repeats require investigation before removal.
- `eaglei_coverage.csv`: timestamp ranges, counties, and total-customer availability by year.
- `storm_details_dates.csv`: unadjusted source-local datetime parsing and duration checks; these timestamps are not yet timezone aligned.

All source fields are read as VARCHAR to preserve identifiers. Null markers: empty field, NA, N/A, NULL, null, NaN, nan, None, <NA>, #N/A. Exact duplicates use these parsed values without trimming or other cleaning. Distinct counts exclude nulls. Missing keys are reported separately but are included in duplicate grouping. This profile does not establish that absent county/time observations mean zero outages, validate geographic crosswalks, or assess feature availability at prediction time.

Held-out 2025 data is included for structural quality/coverage auditing only; do not use it for feature selection or model tuning. The role of 2024 remains undecided.
''', encoding='utf-8')
    print(f'Finished. Results: {OUT}', flush=True)

if __name__ == '__main__':
    main()
