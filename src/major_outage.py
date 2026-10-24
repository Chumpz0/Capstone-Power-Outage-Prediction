"""Approved Southeast inputs. Full windows are for audits; X is allowlisted.

Run scripts/download_eda_data.py once to obtain the immutable, checksum-verified
release. No loader silently mixes legacy/Hugging Face files with local inputs.
"""
import json
import pandas as pd
import pyarrow.parquet as pq
from src import config as C


def manifest():
    path = C.MAJOR_INPUT_DIR / 'manifest.json'
    if not path.exists():
        raise FileNotFoundError('Run python scripts/download_eda_data.py, or build pipeline stage 13.')
    value = json.loads(path.read_text(encoding='utf-8'))
    if value['version'] != 2 or value['target']['major_customers'] != 5000 or value['target']['window_hours'] != {'alert': 72, 'onset': 24}:
        raise ValueError('Inputs do not match the approved version-2 dataset contract')
    return value


def feature_columns(design='alert'):
    if design not in ('alert', 'onset'):
        raise ValueError('design must be alert or onset')
    meta = manifest()
    cols = meta['categorical'] + meta['features'][design]
    forbidden = set(meta['outcomes']) | set(meta['analysis_only'])
    if set(cols) & forbidden or any(c.startswith('retro_') for c in cols):
        raise ValueError('Outcome or analysis-only column in feature manifest')
    return cols


def load_major_outage_windows(split='train', design='alert', *, covered_only=True):
    """Development audit rows with outcomes; never accepts the final test split."""
    if split not in ('train', 'validation'):
        raise ValueError('Development accepts train or validation only; test has a separate feature-only loader')
    feature_columns(design)
    frame = pd.read_parquet(C.MAJOR_INPUT_DIR / f'{design}_{split}.parquet')
    if covered_only:
        frame = frame.loc[frame.covered_window.fillna(False).astype(bool)].copy()
        if frame.y_major.isna().any() or frame.label.eq('ongoing').any():
            raise ValueError('Invalid label in covered population')
    return frame.reset_index(drop=True)


def load_major_outage_xy(split='train', design='alert'):
    """Return X, y and grouping IDs; no preprocessing or model fitting."""
    frame = load_major_outage_windows(split, design)
    return (frame[feature_columns(design)].copy(), frame.y_major.astype('int8'),
            frame[['opportunity_id', 'county_fips', 't0', 'cluster_date']].copy())


def load_major_outage_test_features(design='alert'):
    """For frozen final inference only: return X and IDs, never test outcomes."""
    cols = feature_columns(design)
    path = C.MAJOR_INPUT_DIR / f'{design}_test_features.parquet'
    if set(pq.read_schema(path).names) & set(manifest()['outcomes']):
        raise ValueError('Test file contains forbidden outcome columns')
    ids = ['opportunity_id', 'county_fips', 't0', 'cluster_date']
    frame = pd.read_parquet(path, columns=cols + ids)
    return frame[cols], frame[ids]
