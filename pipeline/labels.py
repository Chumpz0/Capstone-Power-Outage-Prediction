"""Outcome-window summaries and major-outage label rules shared by stages 11-13 and tests.

The outcome window of an opportunity with prediction time t0 is [t0, t0 + H), H hours
long (24 by default; 72 for alert rows under decision D-21). It contains exactly 4H
scheduled EAGLE-I runs (:00, :15, :30, :45 UTC). Each scheduled run in a county is one of
  observed  a row with a known customers_out value (zeros are explicit only from 2022)
  null      a row whose customers_out is missing
  absent    no row. EAGLE-I does not distinguish zero outages from a collection gap.
Rows at other minutes ("off-grid") are kept as evidence of a crossing but never fill a run.

Information available at t0 is limited to runs that started at or before t0 - 15 min;
the run starting at t0 is still collecting. The t0 state uses the last such run, and the
pre-t0 summaries look back 24 hours whatever the window length.

Label states for a customer-count threshold T:
  major      some observed value in the window is >= T (an observed crossing is enough,
             even in an incomplete window)
  non_major  the coverage rule holds and every observed value is < T
  unknown    anything else. Absent runs are never read as zero.
Coverage rules for non_major:
  strict     every scheduled run observed
  cov90      at least 90% of scheduled runs observed (rounded up: 87 of 96, 260 of 288)
             and no gap longer than 4 runs (1 hour)
The asymmetry (positives from partial windows, negatives only from covered ones) is a
known selection risk; the "symmetric" population (cov90 coverage whatever the label)
removes it and is the primary evaluation population (decision D-22).
"""
import math

PRE_WINDOW = 'INTERVAL 24 HOUR'
AVAILABLE_LAG = 'INTERVAL 15 MINUTE'
# Approved at checkpoints A and B and after the first baselines (decisions D-18 to D-22).
MAJOR_CUSTOMERS = 5000
PRIMARY_RULE = 'cov90'
# Absent runs between two observations of at least this many customers are almost surely
# collection gaps: EAGLE-I counts do not fall to zero and recover within 15 minutes.
BRACKET_CUSTOMERS = 1000


def slots(hours):
    return 4 * hours


def non_major_rule(rule, hours=24):
    """SQL coverage condition for a non-major label in an H-hour window."""
    n = slots(hours)
    return {'strict': f'n_observed = {n}',
            'cov90': f'n_observed >= {math.ceil(0.9 * n)} AND max_gap_slots <= 4'}[rule]


def summarize_windows(con, opportunities, observations, out, hours=24):
    """Create table `out`: one row per opportunity with window coverage and t0 state.

    `opportunities` needs opportunity_id, county_fips and t0 (TIMESTAMPTZ).
    `observations` needs county_fips, obs_date (UTC DATE), observation_utc, customers_out
    and on_slot. Each (county_fips, observation_utc) must be unique.
    """
    n = slots(hours)
    window = f'INTERVAL {hours} HOUR'
    con.execute(f"""CREATE OR REPLACE TEMP TABLE {out}_matches AS
        WITH days AS (
            SELECT opportunity_id, county_fips, t0, CAST(day AS DATE) AS day FROM (
                SELECT opportunity_id, county_fips, t0,
                    unnest(generate_series(CAST(t0 - {PRE_WINDOW} AS DATE), CAST(t0 + {window} AS DATE), INTERVAL 1 DAY)) AS day
                FROM {opportunities}))
        SELECT d.opportunity_id, d.t0, o.observation_utc, o.customers_out, o.on_slot,
            o.observation_utc >= d.t0 AS in_window
        FROM days d JOIN {observations} o ON o.county_fips = d.county_fips AND o.obs_date = d.day
        WHERE o.observation_utc >= d.t0 - {PRE_WINDOW} AND o.observation_utc < d.t0 + {window}""")
    con.execute(f"""CREATE OR REPLACE TABLE {out} AS
        WITH known AS (
            SELECT opportunity_id, observation_utc, customers_out,
                lag(observation_utc) OVER w AS prev_time, lag(customers_out) OVER w AS prev_value
            FROM {out}_matches WHERE in_window AND on_slot AND customers_out IS NOT NULL
            WINDOW w AS (PARTITION BY opportunity_id ORDER BY observation_utc)),
        gaps AS (
            SELECT opportunity_id,
                max(date_diff('minute', prev_time, observation_utc) // 15 - 1) AS max_interior_gap,
                sum(CASE WHEN prev_value >= {BRACKET_CUSTOMERS} AND customers_out >= {BRACKET_CUSTOMERS}
                         THEN date_diff('minute', prev_time, observation_utc) // 15 - 1 ELSE 0 END) AS bracketed_gap_slots,
                min(observation_utc) AS first_obs, max(observation_utc) AS last_obs
            FROM known GROUP BY opportunity_id),
        agg AS (
            SELECT opportunity_id,
                count(DISTINCT observation_utc) FILTER (WHERE in_window AND on_slot AND customers_out IS NOT NULL) AS n_observed,
                count(*) FILTER (WHERE in_window AND on_slot AND customers_out IS NULL) AS n_null,
                count(*) FILTER (WHERE in_window AND NOT on_slot) AS n_offgrid,
                count(*) FILTER (WHERE in_window AND on_slot AND customers_out = 0) AS n_zero,
                max(customers_out) FILTER (WHERE in_window AND on_slot) AS max_on_slot,
                max(customers_out) FILTER (WHERE in_window) AS max_any,
                -- Second-highest run: a crossing seen in one 15-minute run only may be a reporting spike.
                list_sort(list(customers_out) FILTER (WHERE in_window AND customers_out IS NOT NULL), 'DESC')[2] AS second_max_any,
                arg_max(observation_utc, customers_out) FILTER (WHERE in_window AND customers_out IS NOT NULL) AS peak_utc,
                count(DISTINCT observation_utc) FILTER (WHERE NOT in_window AND on_slot AND customers_out IS NOT NULL
                    AND observation_utc <= t0 - {AVAILABLE_LAG}) AS pre_observed,
                max(customers_out) FILTER (WHERE NOT in_window AND observation_utc <= t0 - {AVAILABLE_LAG}) AS pre_max_24h,
                max(customers_out) FILTER (WHERE observation_utc = slot_floor(t0 - {AVAILABLE_LAG})) AS at_t0_value,
                count(*) FILTER (WHERE observation_utc = slot_floor(t0 - {AVAILABLE_LAG})) AS at_t0_rows,
                max(customers_out) FILTER (WHERE observation_utc >= t0 - INTERVAL 1 HOUR AND observation_utc <= t0 - {AVAILABLE_LAG}) AS last_hour_max
            FROM {out}_matches GROUP BY opportunity_id)
        SELECT o.opportunity_id,
            coalesce(a.n_observed, 0) AS n_observed, coalesce(a.n_null, 0) AS n_null,
            {n} - coalesce(a.n_observed, 0) - coalesce(a.n_null, 0) AS n_absent,
            coalesce(a.n_offgrid, 0) AS n_offgrid, coalesce(a.n_zero, 0) AS n_zero,
            a.max_on_slot, a.max_any, a.second_max_any, a.peak_utc,
            coalesce(a.pre_observed, 0) AS pre_observed, a.pre_max_24h,
            a.at_t0_value, coalesce(a.at_t0_rows, 0) > 0 AS at_t0_row_present, a.last_hour_max,
            CASE WHEN g.first_obs IS NULL THEN {n} ELSE greatest(
                coalesce(g.max_interior_gap, 0),
                date_diff('minute', slot_ceil(o.t0), g.first_obs) // 15,
                date_diff('minute', g.last_obs, slot_ceil(o.t0) + INTERVAL {(n - 1) * 15} MINUTE) // 15) END AS max_gap_slots,
            coalesce(g.bracketed_gap_slots, 0) AS bracketed_gap_slots
        FROM {opportunities} o LEFT JOIN agg a USING (opportunity_id) LEFT JOIN gaps g USING (opportunity_id)""")
    con.execute(f'DROP TABLE {out}_matches')


def summarize_pre_t0(con, opportunities, observations, out):
    """Create table `out` from runs that started at or before t0 - 15 min only (prediction features).

    Never reads the outcome window, so it is safe for every split. Columns match the
    pre-t0 columns of summarize_windows.
    """
    con.execute(f"""CREATE OR REPLACE TABLE {out} AS
        WITH days AS (
            SELECT opportunity_id, county_fips, t0, CAST(day AS DATE) AS day FROM (
                SELECT opportunity_id, county_fips, t0,
                    unnest(generate_series(CAST(t0 - {PRE_WINDOW} AS DATE), CAST(t0 AS DATE), INTERVAL 1 DAY)) AS day
                FROM {opportunities})),
        m AS (SELECT d.opportunity_id, d.t0, o.observation_utc, o.customers_out, o.on_slot
            FROM days d JOIN {observations} o ON o.county_fips = d.county_fips AND o.obs_date = d.day
            WHERE o.observation_utc >= d.t0 - {PRE_WINDOW} AND o.observation_utc <= d.t0 - {AVAILABLE_LAG}),
        agg AS (SELECT opportunity_id,
                count(DISTINCT observation_utc) FILTER (WHERE on_slot AND customers_out IS NOT NULL) AS pre_observed,
                count(*) FILTER (WHERE on_slot AND customers_out IS NULL) AS pre_null,
                max(customers_out) AS pre_max_24h,
                max(customers_out) FILTER (WHERE observation_utc = slot_floor(t0 - {AVAILABLE_LAG})) AS at_t0_value,
                count(*) FILTER (WHERE observation_utc = slot_floor(t0 - {AVAILABLE_LAG})) AS at_t0_rows,
                max(customers_out) FILTER (WHERE observation_utc >= t0 - INTERVAL 1 HOUR) AS last_hour_max,
                date_diff('minute', max(observation_utc) FILTER (WHERE customers_out IS NOT NULL), any_value(t0)) AS minutes_since_last_observation
            FROM m GROUP BY opportunity_id)
        SELECT o.opportunity_id, coalesce(a.pre_observed, 0) AS pre_observed, coalesce(a.pre_null, 0) AS pre_null,
            a.pre_max_24h, a.at_t0_value, coalesce(a.at_t0_rows, 0) > 0 AS at_t0_row_present, a.last_hour_max,
            a.minutes_since_last_observation
        FROM {opportunities} o LEFT JOIN agg a USING (opportunity_id)""")


def ongoing_at_t0(threshold=MAJOR_CUSTOMERS):
    """Already major at t0: the last available run, or the latest observation in the prior hour if that run is absent."""
    return (f"coalesce(CASE WHEN at_t0_value IS NOT NULL THEN at_t0_value >= {threshold} "
            f"ELSE last_hour_max >= {threshold} END, false)")


def primary_label(threshold=MAJOR_CUSTOMERS, rule=PRIMARY_RULE, hours=24):
    """Approved target: 'ongoing' (excluded), else major / non_major / unknown."""
    return f"CASE WHEN {ongoing_at_t0(threshold)} THEN 'ongoing' ELSE {major_state(threshold, rule, hours)} END"


def major_state(threshold, rule='strict', hours=24):
    """SQL label state for a threshold expression; a NULL threshold yields 'unknown'."""
    return (f"CASE WHEN max_any >= {threshold} THEN 'major' "
            f"WHEN coalesce(max_any, 0) < {threshold} AND {non_major_rule(rule, hours)} THEN 'non_major' "
            "ELSE 'unknown' END")


def t0_state(threshold):
    """SQL state at prediction time from the last available run."""
    return (f"CASE WHEN at_t0_value >= {threshold} THEN 'major_at_t0' "
            f"WHEN at_t0_value < {threshold} THEN 'below_at_t0' ELSE 'unknown_at_t0' END")


def major_timing(label, at_t0):
    """Separate continuing major outages from new threshold crossings."""
    return (f"CASE WHEN {label}<>'major' THEN NULL WHEN {at_t0}='major_at_t0' THEN 'continuing' "
            f"WHEN {at_t0}='below_at_t0' THEN 'new' ELSE 't0_unknown' END")


def single_run_crossing(threshold):
    """Major only because of one observation: candidate spike, reported separately."""
    return f'max_any >= {threshold} AND coalesce(second_max_any, 0) < {threshold}'


def implicit_zero_non_major(threshold):
    """DIAGNOSTIC ONLY, not an approved rule: windows that would be non_major if absent runs were zeros."""
    return f'coalesce(max_any, 0) < {threshold} AND n_null = 0'
