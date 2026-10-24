"""County FIPS, zone codes and zone/county name verification."""
from conftest import stage

crosswalk = stage('09_build_zone_crosswalk')
fetch = stage('08_fetch_reference_sources')


def test_padded_code(con):
    rows = con.execute("""SELECT padded_code('1001', 5), padded_code('01001', 5), padded_code(' 1001 ', 5),
        padded_code('100001', 5), padded_code('1.0', 5), padded_code('A1001', 5), padded_code(NULL, 5)""").fetchone()
    assert rows == ('01001', '01001', '01001', None, None, None, None)


def test_state_fips_and_abbreviations_cover_six_states():
    assert set(stage('common').SE_ABBR_BY_FIPS.values()) == {'AL', 'FL', 'GA', 'NC', 'SC', 'TN'}
    assert all(len(f) == 2 for f in stage('common').SE_STATE_FIPS)


def test_correlation_file_effective_dates():
    assert fetch.effective_date('bp07oc14') == '2014-10-07'
    assert fetch.effective_date('bp16ap26') == '2026-04-16'
    assert fetch.effective_date('bp01my18') == '2018-05-01'


def test_county_names_tolerate_spacing_and_sub_county_labels():
    assert crosswalk.names_match('DE KALB', ['DeKalb'])
    assert crosswalk.names_match('ST. JOHNS', ['St. Johns'])
    assert crosswalk.names_match('INALND ST. JOHNS', ['St. Johns'])   # NOAA typo, county still named
    assert not crosswalk.names_match('OCONEE', ['Calhoun'])


def test_zone_name_check():
    assert crosswalk.name_check('LOWER MOBILE', 'Lower Mobile', ['Mobile'])[1] == 'verified'
    # Zone renamed but the county name is in the NOAA record: verified through the county.
    assert crosswalk.name_check('EASTERN DARE', 'Outer Banks Dare', ['Dare', 'Mainland Dare'])[1] == 'verified'
    assert crosswalk.name_check('TROUT RIVER', 'Trout River', ['Duval'])[1] == 'verified'
    assert crosswalk.name_check('BALDWIN COASTAL', 'Dale', ['Dale'])[1] == 'conflict'
