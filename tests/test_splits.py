"""Chronological splits and guards that keep the 2025 test split and future fields out of development."""
import re
from pathlib import Path
from conftest import stage

ROOT = Path(__file__).resolve().parents[1]
common = stage('common')


def test_split_years(con):
    got = con.execute(f"""SELECT {common.split_sql("TIMESTAMPTZ '2023-12-31 23:59:00+00'")}, {common.split_sql("TIMESTAMPTZ '2024-01-01 00:00:00+00'")},
        {common.split_sql("TIMESTAMPTZ '2025-07-04 00:00:00+00'")}, {common.split_sql("TIMESTAMPTZ '2026-01-01 00:00:00+00'")}""").fetchone()
    assert got == ('train', 'validation', 'test', 'out_of_scope')


def test_label_audit_summarizes_only_development_windows():
    source = (ROOT / 'pipeline' / '11_audit_major_labels.py').read_text(encoding='utf-8')
    assert "split IN ('train', 'validation')" in source
    assert "summarize_windows(con, 'audit_opps'" in source


def test_alert_features_never_use_final_event_state():
    source = (ROOT / 'pipeline' / '12_audit_alert_feasibility.py').read_text(encoding='utf-8')
    create = source[source.index('CREATE TABLE alerts AS'):source.index('FROM raw"""')]
    for future_field in ('utc_expire,', 'status', 'utc_updated'):
        assert not re.search(rf'\b{future_field}', create.replace('utc_init_expire', '')), future_field


def test_retro_fields_are_prefixed():
    source = (ROOT / 'pipeline' / '10_build_southeast_base.py').read_text(encoding='utf-8')
    for realized in ('end_utc AS ', 'duration_minutes AS ', 'magnitude_value AS '):
        alias = source.split(realized)[1].split()[0].rstrip(',')
        assert alias.startswith('retro_'), alias
