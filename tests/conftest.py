"""Shared fixtures. Pipeline modules are imported from pipeline/ (numbered stages via importlib)."""
import importlib
import sys
from pathlib import Path
import pytest

PIPELINE = Path(__file__).resolve().parents[1] / 'pipeline'
sys.path.insert(0, str(PIPELINE))

import common  # noqa: E402


def stage(name):
    """Import a numbered stage module such as '10_build_southeast_base'."""
    return importlib.import_module(name)


@pytest.fixture
def con():
    connection = common.connect()
    yield connection
    connection.close()


def summarize(con, t0, rows, county='01001', hours=24):
    """Summarize one synthetic opportunity at t0 with observations [(time, customers_out), ...].

    Leaves tables `opps`, `obs` and `w` in the connection and returns the summary row.
    """
    labels = importlib.import_module('labels')
    con.execute('CREATE OR REPLACE TABLE opps (opportunity_id VARCHAR, county_fips VARCHAR, t0 TIMESTAMPTZ)')
    con.execute('INSERT INTO opps VALUES (?, ?, ?)', ['o1', county, t0])
    con.execute('CREATE OR REPLACE TABLE obs (county_fips VARCHAR, obs_date DATE, observation_utc TIMESTAMPTZ, customers_out BIGINT, on_slot BOOLEAN)')
    con.executemany('INSERT INTO obs VALUES (?, ?, ?, ?, on_slot(?::TIMESTAMPTZ))', [[county, t.date(), t, v, t] for t, v in rows])
    labels.summarize_windows(con, 'opps', 'obs', 'w', hours=hours)
    return con.execute('SELECT * FROM w').df().iloc[0]
