"""Prevent outcome leakage and population drift in teammate-facing loaders."""
import json
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src import major_outage as data


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'MAJOR_INPUT_DIR', tmp_path)
    meta = {'version': 2, 'target': {'major_customers': 5000, 'window_hours': {'alert': 72, 'onset': 24}},
            'categorical': ['state'], 'features': {'alert': ['month'], 'onset': ['month']},
            'outcomes': ['y_major', 'covered_window', 'label'], 'analysis_only': ['retro_later']}
    (tmp_path / 'manifest.json').write_text(json.dumps(meta))
    frame = pd.DataFrame({'state': ['GA'] * 3, 'month': [1, 2, 3], 'covered_window': [True, False, False],
                          'y_major': [1, 1, None], 'label': ['major', 'major', 'ongoing'],
                          'retro_later': [42] * 3, 'opportunity_id': ['a', 'b', 'c'],
                          'county_fips': ['13001'] * 3, 't0': pd.to_datetime(['2023-01-01'] * 3, utc=True),
                          'cluster_date': ['2023-01-01'] * 3})
    frame.to_parquet(tmp_path / 'alert_train.parquet')
    return tmp_path, meta, frame


def test_xy_only_covered_rows_and_allowlisted_features(inputs):
    X, y, groups = data.load_major_outage_xy()
    assert list(X.columns) == ['state', 'month']
    assert y.tolist() == [1] and groups.opportunity_id.tolist() == ['a']
    assert len(data.load_major_outage_windows(covered_only=False)) == 3


def test_development_loader_rejects_test(inputs):
    with pytest.raises(ValueError, match='Development accepts'):
        data.load_major_outage_windows('test')


def test_feature_manifest_cannot_include_outcomes(inputs):
    folder, meta, _ = inputs
    meta['features']['alert'].append('y_major')
    (folder / 'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='Outcome or analysis'):
        data.feature_columns()


def test_test_loader_refuses_outcome_columns(inputs):
    folder, _, frame = inputs
    frame.to_parquet(folder / 'alert_test_features.parquet')
    with pytest.raises(ValueError, match='forbidden outcome'):
        data.load_major_outage_test_features()


def test_test_loader_returns_only_features_and_ids(inputs):
    folder, _, frame = inputs
    frame.drop(columns=['y_major', 'covered_window', 'label']).to_parquet(folder / 'alert_test_features.parquet')
    X, ids = data.load_major_outage_test_features()
    assert list(X) == ['state', 'month'] and 'retro_later' not in ids
    assert len(X) == len(ids) == 3


def test_release_has_no_test_outcomes(inputs, tmp_path):
    from scripts.publish_eda_data import prepare
    folder, _, frame = inputs
    for design in ('alert', 'onset'):
        for split in ('train', 'validation'):
            frame.to_parquet(folder / f'{design}_{split}.parquet')
        frame.to_parquet(folder / f'{design}_test_features.parquet')
    with pytest.raises(ValueError, match='test outcome'):
        prepare(tmp_path / 'release')
