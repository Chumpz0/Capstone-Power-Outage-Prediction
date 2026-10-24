"""Audit whether major-outage labels are trustworthy before any threshold is chosen.

Run: python pipeline/11_audit_major_labels.py   (after stage 10)
Outputs in reports/label_audit/ (summary.json lists every file):
  coverage_*.csv            structural reporting coverage by county, state and year (all years)
  label_states.csv          major / non_major / unknown counts by split, threshold and coverage rule
  selection_effects.csv     training only: label-state shares by county size, state, era, hazard
  threshold_candidates.csv  training only: how often each candidate threshold is crossed, and where
  t0_state.csv              continuing versus new major outages at prediction time
  severity_completeness.csv windows whose peak is fully observed versus only bounded below
  collection_gaps.csv       absent runs bracketed by large outages (evidence that absent != zero)

Evaluation rules enforced here:
- Window labels are computed for training and validation opportunities only. The 2025
  test split contributes opportunity counts and structural county-year coverage, never
  outcome-window summaries or label states.
- Threshold candidates and selection effects use training data only. Validation counts
  are reported as usable-sample sizes, not for choosing a threshold.
- Absent runs are never read as zero. The implicit-zero column is a diagnostic of how many
  negatives would depend on that assumption, not a label.
"""
import argparse
import json

from common import STORM_GROUPS, connect, copy_query, lit, write_json
from labels import implicit_zero_non_major, major_state, major_timing, single_run_crossing, summarize_windows, t0_state
from paths import LABEL_REPORTS, SOUTHEAST

COUNT_THRESHOLDS = (1000, 2500, 5000, 10000, 25000, 50000)
PCT_THRESHOLDS = (0.01, 0.05, 0.10, 0.25)
RULES = ('strict', 'cov90')
# County size by 2020 Census population: independent of utility reporting.
SIZE_BINS = ((0, 25_000, '1: <25k'), (25_000, 50_000, '2: 25k-50k'), (50_000, 100_000, '3: 50k-100k'),
             (100_000, 250_000, '4: 100k-250k'), (250_000, 500_000, '5: 250k-500k'), (500_000, 10**9, '6: 500k+'))
PLAUSIBLE = "outage_plausibility = 'ok'"


def size_bin_sql(col):
    cases = ' '.join(f"WHEN {col} >= {lo} AND {col} < {hi} THEN '{label}'" for lo, hi, label in SIZE_BINS)
    return f"CASE {cases} ELSE 'unknown' END"


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    LABEL_REPORTS.mkdir(parents=True, exist_ok=True)
    con = connect()
    for name, file in [('eaglei', 'eaglei_southeast.parquet'), ('counties', 'county_reference.parquet'),
                       ('opps', 'onset_opportunities.parquet')]:
        con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({lit(SOUTHEAST / file)})')
    outputs = {}

    def report(name, query, order=None):
        # Sorting in an outer query lets ORDER BY use output columns in expressions.
        copy_query(con, f'SELECT * FROM ({query}) ORDER BY {order}' if order else query, LABEL_REPORTS / name, fmt='csv')
        outputs[name] = con.execute(f"SELECT count(*) FROM read_csv({lit(LABEL_REPORTS / name)})").fetchone()[0]

    print('Structural coverage by county, state and year (all years)', flush=True)
    con.execute("""CREATE TABLE state_runs AS SELECT substr(county_fips, 1, 2) AS state_fips, source_year,
            count(DISTINCT observation_utc) FILTER (WHERE on_slot) AS state_runs,
            min(observation_utc) AS first_utc, max(observation_utc) AS last_utc
        FROM eaglei GROUP BY ALL""")
    report('coverage_county_year.csv', """SELECT e.county_fips, c.state, c.county_name, e.source_year,
            any_value(e.reporting_convention) AS reporting_convention, count(*) AS rows,
            count(*) FILTER (WHERE e.on_slot AND e.customers_out IS NOT NULL) AS observed_runs,
            count(*) FILTER (WHERE e.on_slot AND e.customers_out IS NULL) AS null_runs,
            count(*) FILTER (WHERE e.recorded_zero) AS zero_rows,
            count(*) FILTER (WHERE NOT e.on_slot) AS offgrid_rows,
            count(DISTINCT e.obs_date) AS days_with_rows,
            any_value(s.state_runs) AS state_runs,
            round(count(*) FILTER (WHERE e.on_slot AND e.customers_out IS NOT NULL) / any_value(s.state_runs), 4) AS observed_share_of_state_runs,
            min(e.observation_utc) AS first_utc, max(e.observation_utc) AS last_utc
        FROM eaglei e JOIN counties c USING (county_fips)
        JOIN state_runs s ON s.state_fips = substr(e.county_fips, 1, 2) AND s.source_year = e.source_year
        GROUP BY e.county_fips, c.state, c.county_name, e.source_year ORDER BY e.county_fips, e.source_year""")
    report('coverage_state_year.csv', f"""WITH cy AS (SELECT * FROM read_csv({lit(LABEL_REPORTS / 'coverage_county_year.csv')}, types={{'county_fips': 'VARCHAR'}}))
        SELECT cy.state, cy.source_year, any_value(cy.reporting_convention) AS reporting_convention,
            count(*) AS counties_with_rows, sum(cy.rows) AS rows,
            round(sum(cy.zero_rows) / sum(cy.rows), 4) AS zero_row_share,
            round(sum(cy.null_runs) / sum(cy.rows), 4) AS null_row_share,
            sum(cy.offgrid_rows) AS offgrid_rows,
            count(*) FILTER (WHERE cy.zero_rows > 0) AS counties_with_explicit_zeros,
            any_value(cy.state_runs) AS state_runs,
            -- Scheduled runs between the state's first and last observation with no row anywhere in the state.
            any_value(date_diff('minute', s.first_utc, s.last_utc) // 15 + 1 - s.state_runs) AS statewide_dark_runs,
            round(median(cy.observed_share_of_state_runs), 4) AS median_county_observed_share,
            round(quantile_cont(cy.observed_share_of_state_runs, 0.9), 4) AS p90_county_observed_share
        FROM cy JOIN state_runs s ON s.state_fips = substr(cy.county_fips, 1, 2) AND s.source_year = cy.source_year
        GROUP BY cy.state, cy.source_year ORDER BY cy.state, cy.source_year""")
    report('coverage_dec31.csv', """SELECT source_year, count(*) FILTER (WHERE month(observation_utc) = 12 AND day(observation_utc) = 31) AS dec31_rows,
            count(*) FILTER (WHERE month(observation_utc) = 12 AND day(observation_utc) = 31 AND hour(observation_utc) + minute(observation_utc) > 0) AS dec31_rows_after_0000,
            max(observation_utc) AS last_utc
        FROM eaglei GROUP BY 1 ORDER BY 1""")

    print('Opportunity eligibility (test split: counts only)', flush=True)
    con.execute(f"""CREATE TABLE opp_status AS SELECT o.*, c.mcc_customers, c.mcc_status, c.total_2024_median, c.eaglei_total_status,
            c.population_2020, c.outage_plausibility,
            -- Largest plausible count for the county: a valid MCC, else the population as a loose ceiling.
            CASE WHEN c.mcc_status = 'valid' THEN c.mcc_customers ELSE c.population_2020 END AS customer_ceiling,
            CASE WHEN o.split NOT IN ('train', 'validation', 'test') THEN 'out_of_scope'
                 WHEN o.split_boundary_window THEN 'split_boundary_window'
                 WHEN o.cross_split_episode THEN 'cross_split_episode'
                 WHEN NOT o.eaglei_tracked THEN 'county_never_in_eaglei'
                 ELSE 'eligible' END AS eligibility,
            year(o.t0) AS year,
            CASE WHEN year(o.t0) <= 2017 THEN '2014-2017' WHEN year(o.t0) <= 2021 THEN '2018-2021' ELSE '2022+' END AS era,
            month(o.t0) AS month, o.trigger_hazard_groups[1] AS primary_hazard
        FROM opps o JOIN counties c USING (county_fips)""")
    report('opportunity_eligibility.csv', """SELECT split, eligibility, count(*) AS opportunities
        FROM opp_status GROUP BY ALL ORDER BY split, eligibility""")
    con.execute("CREATE TABLE audit_opps AS SELECT opportunity_id, county_fips, t0 FROM opp_status WHERE eligibility = 'eligible' AND split IN ('train', 'validation')")
    assert con.execute("SELECT count(*) FROM audit_opps a JOIN opp_status o USING (opportunity_id) WHERE o.split = 'test'").fetchone()[0] == 0
    con.execute('CREATE VIEW obs AS SELECT county_fips, obs_date, observation_utc, customers_out, on_slot FROM eaglei')

    print('Summarizing outcome windows (training and validation only)', flush=True)
    summarize_windows(con, 'audit_opps', 'obs', 'windows')
    con.execute(f"""CREATE TABLE w AS SELECT o.opportunity_id, o.county_fips, o.state, o.split, o.year, o.era, o.month,
            o.primary_hazard, o.trigger_from_zone, o.trigger_partial_county, o.population_2020, o.customer_ceiling,
            o.mcc_customers, o.mcc_status, o.outage_plausibility, {size_bin_sql('o.population_2020')} AS size_bin,
            w.*  EXCLUDE (opportunity_id),
            w.max_any > {1.05} * o.customer_ceiling AS window_peak_exceeds_customers
        FROM windows w JOIN opp_status o USING (opportunity_id)""")

    thresholds = []
    for t in COUNT_THRESHOLDS:
        for rule in RULES:
            thresholds.append(f"""SELECT opportunity_id, 'count' AS kind, '{t}' AS threshold, '{rule}' AS rule,
                {major_state(t, rule)} AS label, {t0_state(t)} AS at_t0,
                {implicit_zero_non_major(t)} AS implicit_zero_non_major,
                coalesce(customer_ceiling < {t}, false) AS threshold_exceeds_ceiling,
                {single_run_crossing(t)} AS single_run FROM w""")
    for p in PCT_THRESHOLDS:
        # Percentages only where the modeled denominator validated and the county's outage peaks are plausible.
        denom = f"CASE WHEN mcc_status = 'valid' AND {PLAUSIBLE} THEN ceil({p} * mcc_customers) END"
        for rule in RULES:
            thresholds.append(f"""SELECT opportunity_id, 'pct_mcc' AS kind, '{p:.0%}' AS threshold, '{rule}' AS rule,
                CASE WHEN ({denom}) IS NULL THEN 'no_valid_denominator' ELSE {major_state(denom, rule)} END AS label,
                {t0_state(denom)} AS at_t0, {implicit_zero_non_major(denom)} AS implicit_zero_non_major,
                false AS threshold_exceeds_ceiling, coalesce({single_run_crossing(denom)}, false) AS single_run FROM w""")
    con.execute(f"""CREATE TABLE labels AS SELECT l.*, {major_timing('l.label', 'l.at_t0')} AS major_timing
        FROM ({' UNION ALL '.join(thresholds)}) l""")

    print('Label states, selection effects and threshold candidates', flush=True)
    report('label_states.csv', """SELECT w.split, l.kind, l.threshold, l.rule, count(*) AS opportunities,
            count(*) FILTER (WHERE l.label = 'major') AS major,
            count(*) FILTER (WHERE l.major_timing = 'new') AS major_new,
            count(*) FILTER (WHERE l.major_timing = 'continuing') AS major_continuing,
            count(*) FILTER (WHERE l.major_timing = 't0_unknown') AS major_t0_unknown,
            count(*) FILTER (WHERE l.label = 'non_major') AS non_major,
            count(*) FILTER (WHERE l.label = 'unknown') AS unknown,
            count(*) FILTER (WHERE l.label = 'no_valid_denominator') AS no_valid_denominator,
            count(*) FILTER (WHERE l.label = 'non_major' AND l.threshold_exceeds_ceiling) AS non_major_threshold_above_customers,
            count(*) FILTER (WHERE l.label = 'unknown' AND l.implicit_zero_non_major) AS unknown_non_major_if_absent_were_zero,
            round(count(*) FILTER (WHERE l.label IN ('major', 'non_major')) / count(*), 4) AS known_share
        FROM labels l JOIN w USING (opportunity_id) GROUP BY ALL""", order='split DESC, kind, try_cast(threshold AS INTEGER), threshold, rule')
    report('label_robustness.csv', """SELECT w.split, l.threshold, count(*) FILTER (WHERE l.label = 'major') AS major,
            count(*) FILTER (WHERE l.label = 'major' AND l.single_run) AS major_single_run_only,
            count(*) FILTER (WHERE l.label = 'major' AND w.window_peak_exceeds_customers) AS major_peak_above_customer_ceiling,
            count(*) FILTER (WHERE l.label = 'major' AND w.outage_plausibility IN ('peak_exceeds_mcc', 'peak_exceeds_population')) AS major_in_implausible_peak_county,
            count(*) FILTER (WHERE l.label = 'major' AND w.n_offgrid > 0 AND w.max_on_slot < try_cast(l.threshold AS INTEGER)) AS major_from_offgrid_only,
            count(*) FILTER (WHERE l.label = 'major' AND w.trigger_partial_county) AS major_partial_county_trigger
        FROM labels l JOIN w USING (opportunity_id) WHERE l.kind = 'count' AND l.rule = 'strict'
        GROUP BY ALL""", order='split DESC, try_cast(threshold AS INTEGER)')
    dims = {'size_bin': 'w.size_bin', 'state': 'w.state', 'era': 'w.era',
            'primary_hazard': 'w.primary_hazard', 'month': 'w.month::VARCHAR',
            'trigger_from_zone': 'w.trigger_from_zone::VARCHAR', 'trigger_partial_county': 'w.trigger_partial_county::VARCHAR'}
    report('selection_effects.csv', ' UNION ALL '.join(f"""SELECT '{d}' AS dimension, {expr} AS value, l.threshold, l.rule,
            count(*) AS opportunities,
            round(count(*) FILTER (WHERE l.label = 'major') / count(*), 4) AS major_share,
            round(count(*) FILTER (WHERE l.label = 'non_major') / count(*), 4) AS non_major_share,
            round(count(*) FILTER (WHERE l.label = 'unknown') / count(*), 4) AS unknown_share,
            round(avg(w.n_observed), 1) AS mean_observed_runs
        FROM labels l JOIN w USING (opportunity_id)
        WHERE w.split = 'train' AND l.kind = 'count' GROUP BY ALL""" for d, expr in dims.items()) + ' ORDER BY dimension, value, threshold, rule')
    report('threshold_candidates.csv', f"""WITH t AS (SELECT l.threshold, l.kind, w.*, l.label, l.major_timing, l.threshold_exceeds_ceiling
            FROM labels l JOIN w USING (opportunity_id) WHERE w.split = 'train' AND l.rule = 'strict')
        SELECT kind, threshold, count(*) AS train_opportunities,
            count(*) FILTER (WHERE label = 'major') AS major_windows,
            round(count(*) FILTER (WHERE label = 'major') / count(*), 4) AS major_share,
            count(*) FILTER (WHERE major_timing = 'new') AS major_new,
            count(*) FILTER (WHERE major_timing = 'continuing') AS major_continuing,
            count(*) FILTER (WHERE label = 'non_major') AS non_major_strict,
            count(DISTINCT county_fips) FILTER (WHERE label = 'major') AS counties_with_major,
            count(DISTINCT county_fips) AS counties,
            count(DISTINCT county_fips) FILTER (WHERE threshold_exceeds_ceiling) AS counties_with_fewer_customers_than_threshold,
            count(DISTINCT year) FILTER (WHERE label = 'major') AS years_with_major,
            {', '.join(f"count(*) FILTER (WHERE label = 'major' AND state = '{s}') AS major_{s}" for s in ('AL', 'FL', 'GA', 'NC', 'SC', 'TN'))},
            {', '.join(f"count(*) FILTER (WHERE label = 'major' AND primary_hazard = '{g}') AS major_{g}" for g in STORM_GROUPS)}
        FROM t GROUP BY kind, threshold ORDER BY kind, try_cast(threshold AS INTEGER), threshold""")
    report('threshold_by_size.csv', """SELECT l.threshold, w.size_bin,
            count(*) AS train_opportunities, count(DISTINCT w.county_fips) AS counties,
            count(*) FILTER (WHERE l.label = 'major') AS major_windows,
            round(count(*) FILTER (WHERE l.label = 'major') / count(*), 4) AS major_share,
            round(count(*) FILTER (WHERE l.label = 'non_major') / count(*), 4) AS non_major_share,
            round(count(*) FILTER (WHERE l.label = 'unknown') / count(*), 4) AS unknown_share,
            count(*) FILTER (WHERE l.threshold_exceeds_ceiling) AS opportunities_threshold_above_customers,
            round(median(w.max_any / nullif(w.mcc_customers, 0)) FILTER (WHERE l.label = 'major' AND w.mcc_status = 'valid'), 4) AS median_peak_share_of_mcc_in_majors
        FROM labels l JOIN w USING (opportunity_id)
        WHERE w.split = 'train' AND l.kind = 'count' AND l.rule = 'strict' GROUP BY ALL""", order='try_cast(threshold AS INTEGER), size_bin')
    report('t0_state.csv', """SELECT w.split, l.kind, l.threshold, l.at_t0, count(*) AS opportunities,
            count(*) FILTER (WHERE l.label = 'major') AS major_windows,
            count(*) FILTER (WHERE w.pre_max_24h >= try_cast(l.threshold AS INTEGER)) AS crossed_in_prior_24h
        FROM labels l JOIN w USING (opportunity_id) WHERE l.rule = 'strict' AND l.kind = 'count'
        GROUP BY ALL""", order='split DESC, try_cast(threshold AS INTEGER), at_t0')
    report('severity_completeness.csv', """SELECT w.split, l.threshold,
            count(*) FILTER (WHERE l.label = 'major') AS major_windows,
            count(*) FILTER (WHERE l.label = 'major' AND w.n_observed = 96) AS major_peak_fully_observed,
            count(*) FILTER (WHERE l.label = 'major' AND w.n_observed < 96) AS major_peak_lower_bound_only,
            count(*) FILTER (WHERE l.label = 'major' AND w.bracketed_gap_slots > 0) AS major_with_bracketed_gaps,
            count(*) FILTER (WHERE w.max_any > 0) AS windows_with_positive_observation,
            count(*) FILTER (WHERE w.max_any > 0 AND w.n_observed = 96) AS positive_windows_complete,
            count(*) FILTER (WHERE w.n_observed = 96 AND coalesce(w.max_any, 0) = 0) AS complete_all_zero_windows
        FROM labels l JOIN w USING (opportunity_id) WHERE l.rule = 'strict' AND l.kind = 'count'
        GROUP BY ALL""", order='split DESC, try_cast(threshold AS INTEGER)')
    report('collection_gaps.csv', """SELECT w.split, w.primary_hazard, count(*) AS opportunities,
            count(*) FILTER (WHERE w.max_any >= 1000) AS windows_reaching_1000,
            count(*) FILTER (WHERE w.max_any >= 1000 AND w.bracketed_gap_slots > 0) AS reaching_1000_with_bracketed_gap,
            round(avg(w.bracketed_gap_slots) FILTER (WHERE w.max_any >= 1000), 2) AS mean_bracketed_gap_runs_when_reaching_1000,
            count(*) FILTER (WHERE w.n_null > 0) AS windows_with_null_values,
            count(*) FILTER (WHERE w.n_offgrid > 0) AS windows_with_offgrid_rows
        FROM w WHERE w.split = 'train' GROUP BY ALL ORDER BY opportunities DESC""")
    report('observed_runs_distribution.csv', """SELECT split, era,
            CASE WHEN n_observed = 96 THEN '96 (complete)' WHEN n_observed >= 87 THEN '87-95' WHEN n_observed >= 48 THEN '48-86'
                 WHEN n_observed >= 1 THEN '1-47' ELSE '0' END AS observed_runs, count(*) AS opportunities,
            count(*) FILTER (WHERE max_any > 0) AS with_positive_observation
        FROM w GROUP BY ALL ORDER BY split DESC, era, observed_runs""")

    key = con.execute("""SELECT * FROM (SELECT w.split, l.threshold, l.rule, count(*) FILTER (WHERE l.label = 'major') AS major,
            count(*) FILTER (WHERE l.label = 'non_major') AS non_major, count(*) FILTER (WHERE l.label = 'unknown') AS unknown
        FROM labels l JOIN w USING (opportunity_id) WHERE l.kind = 'count' GROUP BY ALL)
        ORDER BY split DESC, try_cast(threshold AS INTEGER), rule""").fetchall()
    summary = {
        'outputs': outputs,
        'eligibility': {f'{s}/{e}': n for s, e, n in con.execute('SELECT split, eligibility, count(*) FROM opp_status GROUP BY ALL ORDER BY ALL').fetchall()},
        'windows_summarized': dict(con.execute('SELECT split, count(*) FROM w GROUP BY 1').fetchall()),
        'label_counts': [{'split': s, 'threshold': t, 'rule': r, 'major': a, 'non_major': b, 'unknown': c} for s, t, r, a, b, c in key],
        'thresholds': {'count': list(COUNT_THRESHOLDS), 'pct_of_valid_mcc': list(PCT_THRESHOLDS)},
        'rules': {'strict': 'all 96 runs observed', 'cov90': '>=87 runs observed and no gap > 4 runs'},
        'test_split': 'opportunity counts only; no outcome windows read',
    }
    write_json(summary, LABEL_REPORTS / 'summary.json')
    print(json.dumps({k: summary[k] for k in ('eligibility', 'windows_summarized')}, indent=2), flush=True)
    con.close()


if __name__ == '__main__':
    main()
