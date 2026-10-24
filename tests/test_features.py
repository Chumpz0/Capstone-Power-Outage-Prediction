"""Prediction features use only information before t0; the approved target behaves as specified."""
import json
from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest
from conftest import stage, summarize

labels = stage('labels')
paths = stage('paths')
UTC = timezone.utc
T0 = datetime(2023, 6, 1, 12, 0, tzinfo=UTC)


def pre_features(con, rows):
    summarize(con, T0, rows)
    labels.summarize_pre_t0(con, 'opps', 'obs', 'pre')
    return con.execute('SELECT * FROM pre').df().iloc[0]


def test_pre_t0_features_ignore_the_outcome_window(con):
    before = [(T0 - timedelta(minutes=15 * i), 300 + i) for i in range(1, 97)]
    quiet = pre_features(con, before + [(T0 + timedelta(hours=2), 10)])
    storm = pre_features(con, before + [(T0 + timedelta(hours=2), 90000), (T0, 80000)])
    assert quiet.drop('opportunity_id').equals(storm.drop('opportunity_id'))


def test_pre_t0_features_match_the_window_summary(con):
    rows = [(T0 - timedelta(minutes=15 * i), 50 * i) for i in range(1, 40)] + [(T0 + timedelta(hours=1), 7000)]
    w = summarize(con, T0, rows)
    labels.summarize_pre_t0(con, 'opps', 'obs', 'pre')
    p = con.execute('SELECT * FROM pre').df().iloc[0]
    for col in ('at_t0_value', 'last_hour_max', 'pre_max_24h', 'pre_observed'):
        assert p[col] == w[col], col


def test_ongoing_falls_back_to_the_last_hour(con):
    def label(rows):
        summarize(con, T0, rows)
        return con.execute(f'SELECT {labels.primary_label()} FROM w').fetchone()[0]
    window = [(T0 + timedelta(minutes=15 * i), 6000) for i in range(96)]
    assert label([(T0 - timedelta(minutes=15), 6000)] + window) == 'ongoing'
    assert label([(T0 - timedelta(minutes=45), 6000)] + window) == 'ongoing'      # last run absent, earlier run major
    assert label([(T0 - timedelta(minutes=15), 4000), (T0 - timedelta(minutes=45), 6000)] + window) == 'major'  # recovered below at t0
    assert label(window) == 'major'


def test_primary_rule_is_the_approved_one():
    assert (labels.MAJOR_CUSTOMERS, labels.PRIMARY_RULE) == (5000, 'cov90')


def test_model_inputs_keep_outcomes_out_of_features_and_test_files():
    manifest_path = paths.MODEL_INPUT / 'southeast_v1' / 'manifest.json'
    if not manifest_path.exists():
        pytest.skip('model inputs not built')
    manifest = json.loads(manifest_path.read_text())
    forbidden = set(manifest['outcomes']) | set(manifest['analysis_only'])
    for design, feats in manifest['features'].items():
        assert not forbidden & set(feats), design
        assert not [f for f in feats if f.startswith('retro_')], design
        test_cols = pd.read_parquet(paths.MODEL_INPUT / 'southeast_v1' / f'{design}_test_features.parquet').columns
        assert not set(manifest['outcomes']) & set(test_cols), design
