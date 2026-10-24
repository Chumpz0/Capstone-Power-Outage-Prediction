"""County opportunities: overlapping storm records must not duplicate county outcomes."""
import pandas as pd
from conftest import stage

base = stage('10_build_southeast_base')


def events(rows):
    frame = pd.DataFrame(rows, columns=['event_id', 'county_fips', 'begin_utc', 'event_type', 'hazard_group', 'mapping', 'partial_county'])
    frame['episode_id'] = 1
    frame['begin_utc'] = pd.to_datetime(frame.begin_utc).astype('datetime64[us]')
    return frame


def test_overlapping_events_share_one_county_window():
    opps = base.opportunities_from(events([
        (1, '12086', '2023-06-01 12:00', 'Thunderstorm Wind', 'convective_wind', 'county_code', None),
        (2, '12086', '2023-06-01 12:00', 'Tropical Storm', 'tropical', 'zone_crosswalk', True),
        (3, '12086', '2023-06-02 06:00', 'Flood', 'rain_flood', 'county_code', None),      # inside the first window
        (4, '12086', '2023-06-02 12:00', 'Tornado', 'convective_wind', 'county_code', None),  # exactly 24 h later: new window
        (5, '12011', '2023-06-01 13:00', 'Tropical Storm', 'tropical', 'zone_crosswalk', False),
    ]))
    assert len(opps) == 3
    first = opps[(opps.county_fips == '12086')].sort_values('t0').iloc[0]
    assert first.trigger_event_ids == [1, 2]                 # only events beginning at t0 describe it
    assert first.retro_events_in_window == 3                 # the later flood is outcome-period audit data
    assert first.trigger_from_zone and first.trigger_partial_county
    assert opps.opportunity_id.is_unique


def test_windows_never_overlap_within_a_county():
    times = pd.date_range('2023-01-01', periods=200, freq='5h')
    opps = base.opportunities_from(events([(i, '01001', t, 'Hail', 'hail', 'county_code', None) for i, t in enumerate(times)]))
    gaps = opps.sort_values('t0').t0.diff().dropna()
    assert (gaps >= pd.Timedelta(hours=24)).all()


def test_greedy_alert_windows_are_72_hours_apart():
    common = stage('common')
    t = pd.Timestamp('2024-09-24 12:00', tz='UTC')
    hours = [0, 30, 71.9, 72, 150, 10]
    rows = sorted([('13121', t + pd.Timedelta(hours=h)) for h in hours[:5]] + [('13089', t + pd.Timedelta(hours=hours[5]))])
    starts = common.greedy_window_starts(rows, 72)
    assert starts == [('13089', t + pd.Timedelta(hours=10)), ('13121', t), ('13121', t + pd.Timedelta(hours=72)),
                      ('13121', t + pd.Timedelta(hours=150))]


def test_zone_event_maps_to_each_county_once(con):
    con.execute("""CREATE TABLE z AS SELECT * FROM (VALUES (10, '13121'), (10, '13089'), (11, '13121')) t(event_id, county_fips)""")
    assert con.execute('SELECT count(*) - count(DISTINCT (event_id, county_fips)) FROM z').fetchone()[0] == 0
