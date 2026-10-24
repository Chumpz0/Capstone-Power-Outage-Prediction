"""Checks on built outputs. Skipped when the data has not been built (for example in a fresh clone)."""
import pandas as pd
import pytest
from conftest import stage

common = stage('common')
paths = stage('paths')
SE = paths.SOUTHEAST


def parquet(name):
    path = SE / name
    if not path.exists():
        pytest.skip(f'{path} not built')
    return common.lit(path)


def test_eaglei_keys_unique_and_in_census_counties(con):
    eagle, counties = parquet('eaglei_southeast.parquet'), parquet('county_reference.parquet')
    dupes, stray = con.execute(f"""SELECT count(*) - count(DISTINCT (county_fips, observation_utc)),
        count(*) FILTER (WHERE county_fips NOT IN (SELECT county_fips FROM read_parquet({counties})))
        FROM read_parquet({eagle})""").fetchone()
    assert (dupes, stray) == (0, 0)


def test_county_list_matches_census(con):
    counts = dict(con.execute(f"SELECT state, count(*) FROM read_parquet({parquet('county_reference.parquet')}) GROUP BY 1").fetchall())
    assert counts == {'AL': 67, 'FL': 67, 'GA': 159, 'NC': 100, 'SC': 46, 'TN': 95}


def test_event_county_rows_unique(con):
    events = parquet('storm_events_southeast.parquet')
    assert con.execute(f'SELECT count(*) - count(DISTINCT (event_id, county_fips)) FROM read_parquet({events})').fetchone()[0] == 0


def test_zone_mappings_are_accepted_statuses(con):
    statuses = {r[0] for r in con.execute(f"SELECT DISTINCT match_status FROM read_parquet({parquet('zone_event_counties.parquet')})").fetchall()}
    assert statuses <= set(stage('09_build_zone_crosswalk').ACCEPTED)


def test_onset_opportunities_do_not_overlap_and_split_by_year(con):
    opps = parquet('onset_opportunities.parquet')
    overlaps, bad_split, dupes = con.execute(f"""SELECT
        count(*) FILTER (WHERE next_t0 < t0 + INTERVAL 24 HOUR),
        count(*) FILTER (WHERE split <> {common.split_sql('t0')}),
        count(*) - count(DISTINCT opportunity_id)
        FROM (SELECT *, lead(t0) OVER (PARTITION BY county_fips ORDER BY t0) AS next_t0 FROM read_parquet({opps}))""").fetchone()
    assert (overlaps, bad_split, dupes) == (0, 0, 0)


def test_model_input_windows_never_overlap(con):
    folder = paths.MODEL_INPUT / 'southeast_v1'
    manifest = folder / 'manifest.json'
    if not manifest.exists():
        pytest.skip('model inputs not built')
    hours = __import__('json').loads(manifest.read_text())['target']['window_hours']
    for design, h in hours.items():
        files = [common.lit(folder / f'{design}_{s}.parquet') for s in ('train', 'validation', 'test_features')]
        union = ' UNION ALL '.join(f'SELECT county_fips, t0 FROM read_parquet({f})' for f in files)
        overlaps = con.execute(f"""SELECT count(*) FROM (SELECT t0, lead(t0) OVER (PARTITION BY county_fips ORDER BY t0) AS nxt FROM ({union}))
            WHERE nxt < t0 + INTERVAL {h} HOUR""").fetchone()[0]
        assert overlaps == 0, design


def test_label_reports_exclude_the_test_split():
    path = paths.LABEL_REPORTS / 'label_states.csv'
    if not path.exists():
        pytest.skip('label audit not run')
    assert set(pd.read_csv(path).split) <= {'train', 'validation'}
