"""Major / non-major / unknown label rules on synthetic EAGLE-I series."""
from datetime import datetime, timedelta, timezone
from conftest import stage, summarize

labels = stage('labels')
UTC = timezone.utc
T0 = datetime(2023, 6, 1, 12, 0, tzinfo=UTC)


def runs(values, start=T0):
    """Consecutive quarter-hour runs from start; None leaves a run absent."""
    return [(start + timedelta(minutes=15 * i), v) for i, v in enumerate(values) if v is not None]


def state(con, threshold, rule='strict'):
    return con.execute(f'SELECT {labels.major_state(threshold, rule)} FROM w').fetchone()[0]


def test_complete_window_below_threshold_is_non_major(con):
    summarize(con, T0, runs([40] * 96))
    assert state(con, 5000) == 'non_major'


def test_observed_crossing_is_major_even_in_sparse_window(con):
    summarize(con, T0, runs([None] * 50 + [6000, 6200]))
    assert state(con, 5000) == 'major'


def test_absent_runs_are_never_zero(con):
    summarize(con, T0, runs([40] * 10))
    assert state(con, 5000) == 'unknown'
    assert con.execute(f'SELECT {labels.implicit_zero_non_major(5000)} FROM w').fetchone()[0]   # diagnostic only


def test_null_runs_do_not_count_as_observed(con):
    summarize(con, T0, runs([40] * 90) + [(T0 + timedelta(minutes=15 * i), None) for i in range(90, 96)])
    w = con.execute('SELECT n_observed, n_null FROM w').fetchone()
    assert w == (90, 6)
    assert state(con, 5000) == 'unknown'


def test_cov90_allows_short_gaps_only(con):
    values = [40] * 96
    for i in (10, 11, 40, 41, 42, 70, 71):
        values[i] = None                                   # 89 observed, longest gap 3 runs
    summarize(con, T0, runs(values))
    assert (state(con, 5000, 'strict'), state(con, 5000, 'cov90')) == ('unknown', 'non_major')
    values = [40] * 90 + [None] * 6                        # 90 observed but a 6-run gap at the end
    summarize(con, T0, runs(values))
    assert state(con, 5000, 'cov90') == 'unknown'


def test_off_grid_row_is_evidence_but_fills_no_run(con):
    summarize(con, T0, [(T0 + timedelta(minutes=7), 8000)])
    w = con.execute('SELECT n_observed, n_offgrid FROM w').fetchone()
    assert w == (0, 1) and state(con, 5000) == 'major'


def test_continuing_versus_new_major(con):
    def timing(rows):
        summarize(con, T0, rows)
        return con.execute(f"""SELECT {labels.major_timing(labels.major_state(5000), labels.t0_state(5000))} FROM w""").fetchone()[0]
    before = T0 - timedelta(minutes=15)
    assert timing([(before, 7000)] + runs([7000])) == 'continuing'
    assert timing([(before, 100)] + runs([7000])) == 'new'
    assert timing(runs([7000])) == 't0_unknown'           # no row at the last available run


def test_single_run_spike_is_flagged(con):
    summarize(con, T0, runs([100, 9000, 100]))
    assert con.execute(f'SELECT {labels.single_run_crossing(5000)} FROM w').fetchone()[0]
    summarize(con, T0, runs([100, 9000, 8000]))
    assert not con.execute(f'SELECT {labels.single_run_crossing(5000)} FROM w').fetchone()[0]


def test_cov90_scales_with_the_window():
    assert labels.non_major_rule('cov90', 24) == 'n_observed >= 87 AND max_gap_slots <= 4'
    assert labels.non_major_rule('cov90', 72) == 'n_observed >= 260 AND max_gap_slots <= 4'
    assert labels.non_major_rule('strict', 72) == 'n_observed = 288'


def test_72_hour_window_counts_288_runs(con):
    w = summarize(con, T0, runs([40] * 288 + [9000]), hours=72)      # the run at t0 + 72 h is outside
    assert (w.n_observed, w.n_absent, w.max_any) == (288, 0, 40)
    values = [40] * 288
    for i in range(0, 280, 10):
        values[i] = None                                              # 28 single-run gaps: 260 observed
    summarize(con, T0, runs(values), hours=72)
    assert con.execute(f"SELECT {labels.major_state(5000, 'cov90', 72)} FROM w").fetchone()[0] == 'non_major'
    values[5] = None                                                  # 259 observed
    summarize(con, T0, runs(values), hours=72)
    assert con.execute(f"SELECT {labels.major_state(5000, 'cov90', 72)} FROM w").fetchone()[0] == 'unknown'


def test_bracketed_gaps_mark_probable_collection_failures(con):
    summarize(con, T0, runs([2000, None, None, None, 2100, 50, None, 40]))
    w = con.execute('SELECT bracketed_gap_slots, max_gap_slots FROM w').fetchone()
    assert w[0] == 3                                       # only the gap between two values >= 1000
    assert w[1] == 88                                      # the window end is unobserved
