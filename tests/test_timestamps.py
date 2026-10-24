"""UTC conversion, EAGLE-I quarter-hour slots and outcome-window boundaries."""
from datetime import datetime, timedelta, timezone
from conftest import summarize

UTC = timezone.utc


def test_utc_from_recorded_offset(con):
    got = con.execute("""SELECT utc_from_offset(TIMESTAMP '2020-07-01 12:00:00', 'EST-5'),
        utc_from_offset(TIMESTAMP '2020-07-01 12:00:00', 'CST-6'),
        utc_from_offset(TIMESTAMP '2020-07-01 12:00:00', 'EDT-4'),
        utc_from_offset(TIMESTAMP '2020-01-01 00:00:00', 'unknown')""").fetchone()
    assert [g and g.astimezone(UTC).hour for g in got] == [17, 18, 16, None]


def test_quarter_hour_slots(con):
    floor, ceil, on, sub = con.execute("""SELECT slot_floor(TIMESTAMPTZ '2024-03-10 14:07:30+00'),
        slot_ceil(TIMESTAMPTZ '2024-03-10 14:07:30+00'), on_slot(TIMESTAMPTZ '2024-03-10 14:15:00+00'),
        on_slot(TIMESTAMPTZ '2024-03-10 14:15:00.5+00')""").fetchone()
    assert (floor.minute, ceil.minute, on, sub) == (0, 15, True, False)


def test_window_contains_exactly_96_runs_on_and_off_slot(con):
    for t0 in (datetime(2023, 6, 1, 14, 0, tzinfo=UTC), datetime(2023, 6, 1, 14, 7, tzinfo=UTC)):
        start = datetime(2023, 5, 31, 12, 0, tzinfo=UTC)
        w = summarize(con, t0, [(start + timedelta(minutes=15 * i), 5) for i in range(4 * 52)])
        assert (w.n_observed, w.n_absent, w.max_gap_slots) == (96, 0, 0)


def test_window_end_is_exclusive_and_t0_state_uses_last_complete_run(con):
    t0 = datetime(2023, 6, 1, 14, 0, tzinfo=UTC)
    w = summarize(con, t0, [(t0 - timedelta(minutes=15), 7), (t0, 1), (t0 + timedelta(hours=24), 9999)])
    assert w.max_any == 1            # the run at t0 + 24 h belongs to the next window
    assert w.at_t0_value == 7        # the run starting at t0 is still collecting at t0
    assert w.n_observed == 1


def test_off_slot_t0_uses_previous_complete_run(con):
    t0 = datetime(2023, 6, 1, 14, 7, tzinfo=UTC)
    w = summarize(con, t0, [(datetime(2023, 6, 1, 13, 45, tzinfo=UTC), 30), (datetime(2023, 6, 1, 14, 0, tzinfo=UTC), 40)])
    assert w.at_t0_value == 30       # 14:00 started only 7 minutes before t0
    assert w.n_observed == 0         # and it is before the window starts


def test_year_end_source_gap_is_unknown_not_zero(con):
    # 2014-2021 files end at Dec 31 00:00 UTC: later runs that day are absent, so nothing is implied.
    t0 = datetime(2019, 12, 31, 0, 0, tzinfo=UTC)
    w = summarize(con, t0, [(t0, 120)])
    assert (w.n_observed, w.n_absent) == (1, 95)
