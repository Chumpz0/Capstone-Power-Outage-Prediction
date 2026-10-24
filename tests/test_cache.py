"""Cache invalidation responds to source content, rules and versions, but not to project location."""
import json
import os
from conftest import stage

common = stage('common')


def test_build_hash_tracks_source_content(tmp_path):
    source = tmp_path / 'source.csv'
    source.write_text('a,b\n1,2\n')
    first = common.build_hash(1, 'SELECT 1', [source], cache_dir=tmp_path)
    assert common.build_hash(1, 'SELECT 1', [source], cache_dir=tmp_path) == first
    source.write_text('a,b\n1,3\n')
    os.utime(source, ns=(1, 1))                            # force a different modification time
    assert common.build_hash(1, 'SELECT 1', [source], cache_dir=tmp_path) != first


def test_build_hash_tracks_rules_and_version(tmp_path, monkeypatch):
    base = common.build_hash(1, 'SELECT 1', cache_dir=tmp_path)
    assert common.build_hash(2, 'SELECT 1', cache_dir=tmp_path) != base
    assert common.build_hash(1, 'SELECT 2', cache_dir=tmp_path) != base
    monkeypatch.setattr(common, 'MACROS', common.MACROS + ('CREATE MACRO x() AS 1',))
    assert common.build_hash(1, 'SELECT 1', cache_dir=tmp_path) != base


def test_build_hash_ignores_project_location(tmp_path):
    inside = common.build_hash(1, f"SELECT * FROM '{common.ROOT}/x.parquet'", cache_dir=tmp_path)
    assert inside == common.build_hash(1, "SELECT * FROM '<ROOT>/x.parquet'", cache_dir=tmp_path)


def test_check_mode_builds_nothing_and_propagates_staleness(tmp_path, con):
    cache = common.ParquetCache(con, 1, 'v', check=True)
    up, down = tmp_path / 'up.parquet', tmp_path / 'down.parquet'
    cache.build('SELECT 1 AS x', up)
    cache.build('SELECT 2 AS x', down, upstream=[up.with_suffix('.meta.json')])
    assert not up.exists() and not down.exists()
    assert cache.status == {'up.parquet': 'rebuild: no output', 'down.parquet': 'rebuild: upstream output changes'}


def test_reuse_then_rebuild_on_change(tmp_path, con):
    out = tmp_path / 'out.parquet'
    common.ParquetCache(con, 1, 'v').build('SELECT 1 AS x', out)
    meta = json.loads(out.with_suffix('.meta.json').read_text())
    cache = common.ParquetCache(con, 1, 'v')
    cache.build('SELECT 1 AS x', out)
    assert cache.status['out.parquet'] == 'reuse'
    cache = common.ParquetCache(con, 2, 'v')
    cache.build('SELECT 1 AS x', out)
    assert cache.status['out.parquet'].startswith('rebuild') and json.loads(out.with_suffix('.meta.json').read_text()) != meta
