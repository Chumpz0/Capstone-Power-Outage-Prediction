"""Shared, provisional storm/outage EDA transformations.

Missing hours remain unknown. `major` describes the observed peak, not a certified
negative outcome for partially observed windows. Windows include both endpoints.
"""


def build_storm_outage_tables(con, outages, storms, census, *, major=0.05,
                              pre_h=1, post_h=12, max_h=48, customer_factor=1.2):
    """Create cust, hourly, ev and imp tables on a DuckDB connection.

    Inputs are DuckDB-readable paths. Same-year ACS (2024 for 2025) and the
    housing multiplier are retrospective EDA conventions, not forecast features.
    """
    if not 0 < major <= 1 or customer_factor <= 0:
        raise ValueError("major must be in (0, 1]; customer_factor must be positive")
    if min(pre_h, post_h, max_h) < 0:
        raise ValueError("Window hours must be nonnegative")
    OUT, STORMS, CENSUS = (str(p).replace("'", "''") for p in (outages, storms, census))
    MAJOR, PRE_H, POST_H, MAX_H = major, pre_h, post_h, max_h

    # Provisional customer estimate; retain the multiplier as an explicit parameter.
    con.sql(f'''
    create or replace table cust as
    select fips_code, acs_year, housing_units, population, {customer_factor} * housing_units as cust_est,
           split_part(county_name, ', ', 1) as county, split_part(county_name, ', ', 2) as state_name
    from '{CENSUS}'
    where housing_units > 0
    ''')

    # 2. County-hours. ACS ends at 2024, so 2025 uses the 2024 estimate.
    con.sql(f'''
    create or replace table hourly as
    select o.fips_code, o.hr, c.cust_est, o.out as raw_out,
           o.readings, o.valid_readings, o.out > c.cust_est as capped,
           least(o.out, c.cust_est) as out
    from (select fips_code, date_trunc('hour', run_start_time) as hr, median(customers_out) as out,
                 count(*) as readings, count(customers_out) as valid_readings
          from '{OUT}'
          group by all
          having count(customers_out) > 0) o
    join cust c on c.fips_code = o.fips_code and c.acs_year = least(year(o.hr), 2024)
    ''')

    # 3. Storm event x county rows, with their outage window [w0, w1]
    SKIP = ["Drought", "Heat", "Excessive Heat", "Rip Current", "Dense Fog", "Freezing Fog",
            "Astronomical Low Tide", "High Surf", "Dust Devil", "Funnel Cloud", "Frost/Freeze"]
    skip_sql = ", ".join(f"'{t}'" for t in SKIP)
    first_hr, last_hr = con.sql("select min(hr), max(hr) from hourly").fetchall()[0]
    if first_hr is None:
        raise ValueError("No usable outage hours matched Census estimates")

    con.sql(f'''
    create or replace table ev as
    select event_id, episode_id, s.fips_code, event_type, year, state_abbr, begin_utc, end_utc, begin_local,
           magnitude, magnitude_type, tor_f_scale, damage_property_usd, mapped_from, c.cust_est,
           date_trunc('hour', begin_utc) - interval {PRE_H} hour as w0,
           date_trunc('hour', least(end_utc, begin_utc + interval {MAX_H} hour)) + interval {POST_H} hour as w1
    from '{STORMS}' s
    -- the inner join also drops the 59 GA rows with bad FIPS codes (notebook 02)
    join cust c on c.fips_code = s.fips_code and c.acs_year = least(s.year, 2024)
    where event_type not in ({skip_sql})
      and date_trunc('hour', begin_utc) between '{first_hr}' and '{last_hr}'
    ''')

    # 4. Outage impact per event-county
    con.sql(f'''
    create or replace table imp as
    with win as (   -- every hour in the event window
      select event_id, fips_code, unnest(generate_series(w0, w1, interval 1 hour)) as hr from ev
    ),
    pre as (        -- the 24 hours before the window
      select event_id, fips_code,
             unnest(generate_series(w0 - interval 24 hour, w0 - interval 1 hour, interval 1 hour)) as hr
      from ev
    ),
    a as (
      select w.event_id, w.fips_code,
             max(h.out) as peak,
             count(*) as expected_hours, count(h.out) as observed_hours,
             sum(h.out) as cust_hours,
             -- hour of the peak; on ties (common once hours hit the cap) take the earliest, so reruns agree
             first(w.hr order by h.out desc, w.hr) filter (where h.out is not null) as peak_hr
      from win w left join hourly h using (fips_code, hr)
      group by all
    ),
    b as (
      select p.event_id, p.fips_code, avg(h.out) as pre_level, count(h.out) as pre_observed_hours
      from pre p left join hourly h using (fips_code, hr)
      group by all
    )
    select ev.*, a.peak, a.cust_hours, a.peak_hr, b.pre_level,
           a.expected_hours, a.observed_hours, b.pre_observed_hours,
           a.observed_hours::double / a.expected_hours as observed_fraction,
           ev.w0 < timestamp '{first_hr}' or ev.w1 > timestamp '{last_hr}' as boundary_window,
           a.peak / ev.cust_est as peak_share,
           a.peak / ev.cust_est >= {MAJOR} as major,
           date_diff('hour', date_trunc('hour', ev.begin_utc), a.peak_hr) as hours_to_peak
    from ev
    join a using (event_id, fips_code)
    join b using (event_id, fips_code)
    ''')


    duplicate_keys = con.sql("select count(*) - count(distinct(event_id, fips_code)) from ev").fetchall()[0][0]
    if duplicate_keys:
        raise ValueError("Event-county keys must be unique")


def grouped_burden(con, group_columns, events_table="ev"):
    """Sum observed outage levels once per group/county/hour, across event windows.

    Group totals may overlap each other (e.g. wind and flood during one storm).
    All-unobserved groups retain NULL burden. Identifiers are validated because
    they are interpolated into SQL; data values are never interpolated here.
    """
    import re
    identifiers = [events_table, *group_columns]
    if not group_columns or any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", s) for s in identifiers):
        raise ValueError("Use simple SQL identifiers for tables and grouping columns")
    groups = ", ".join(group_columns)
    keys = ", ".join(dict.fromkeys([*group_columns, "fips_code"]))
    return con.sql(f"""
        with windows as (
          select distinct {keys}, unnest(generate_series(w0,w1,interval 1 hour)) as hr
          from {events_table}
        )
        select {groups}, sum(h.out) as cust_hours,
               count(*) as expected_hours, count(h.out) as observed_hours
        from windows w left join hourly h using(fips_code, hr)
        group by {groups}
    """).df()
