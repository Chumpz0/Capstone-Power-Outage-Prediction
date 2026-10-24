"""Download public reference sources for the Southeast pipeline.

Run: python pipeline/08_fetch_reference_sources.py [--zones] [--alerts] [--census] [--years 2014-2025]
With no flags everything is fetched. Files go to new raw folders and are never modified:
  data/01_raw/nws_zone_county/<tag>.dbx + manifest.json      NWS zone-county correlation versions
  data/01_raw/iem_vtec/vtec_southeast_<year>.csv + manifest.json   archived NWS alerts
  data/01_raw/census_counties/co-est2023-alldata.csv + manifest.json   Census county population
An existing file is kept after its checksum is verified against the manifest; delete a
file to fetch it again.

Census county population estimates (Vintage 2023) give an independent county list,
canonical county names, and a size measure that does not depend on utility reporting.

Zone-county correlation files (pipe-delimited, no header) map NWS public forecast zones
to counties. weather.gov serves versions from April 2019 onward; earlier versions come
from pinned Internet Archive captures. Versions that were never captured are listed in
MISSING_VERSIONS so the crosswalk can flag events falling in those periods.

VTEC alerts come from the Iowa Environmental Mesonet watch/warning/advisory archive: one
request per UTC year (events starting in that year) for the six states. Rows are one per
event and county/zone UGC, or one per storm-based warning polygon (no UGC).
"""
import argparse
import gzip
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from common import SOUTHEAST_STATES, write_json
from paths import CENSUS_COUNTY_POP, RAW_CENSUS_COUNTIES, RAW_IEM_VTEC, RAW_ZONE_COUNTY

USER_AGENT = 'storm-outage-research/1.0 (reproducible academic pipeline)'
NWS = 'https://www.weather.gov/source/gis/Shapefiles/County/{tag}.dbx'
WAYBACK = 'https://web.archive.org/web/{stamp}id_/{original}'
OLD_NWS = 'http://www.nws.noaa.gov:80/geodata/catalog/wsom/data/{tag}.dbx'
# tag -> Internet Archive capture timestamp and original URL (None: served by weather.gov).
ZONE_VERSIONS = {
    'bp07oc14': ('20141107115206', OLD_NWS), 'bp03de14': ('20141107121349', OLD_NWS),
    'bp04fe15': ('20150223210701', OLD_NWS), 'bp08se15': ('20151024020856', OLD_NWS),
    'bp01oc15': ('20151024020427', OLD_NWS), 'bp10nv15': ('20160113223922', OLD_NWS),
    'bp16jn16': ('20160911145801', OLD_NWS),
    'bp01nv16': ('20161215195739', 'http://www.nws.noaa.gov/geodata/catalog/wsom/data/{tag}.dbx'),
    'bp04ap17': ('20170508101340', 'http://www.nws.noaa.gov/geodata/catalog/wsom/data/{tag}.dbx'),
    'bp02ap19': None, 'bp10oc19': None, 'bp03mr20': ('20201019062733', NWS), 'bp10nv20': None,
    'bp30mr21': None, 'bp08se21': None, 'bp22mr22': None, 'bp07se22': ('20220805173608', NWS),
    'bp13se22': None, 'bp08mr23': None, 'bp19se23': ('20250524150746', NWS), 'bp05mr24': None,
    'bp18mr25': None, 'bp16ap26': None,
}
# Releases that changed Southeast public zones but were never archived (NWS zone change log).
# bp03mr26 is omitted on purpose: its changes were postponed to bp16ap26.
MISSING_VERSIONS = {
    'bp28my15': {'effective_date': '2015-05-28', 'wfos': ['MOB'], 'change': 'SCN 15-14 reconfigured Mobile public zones'},
    'bp05de17': {'effective_date': '2017-12-05', 'wfos': ['ILM'], 'change': 'SCN 17-102 reconfigured Wilmington NC zones'},
    'bp01my18': {'effective_date': '2018-05-01', 'wfos': ['JAX'], 'change': 'SCN 17-122 reconfigured Jacksonville zones (moved to the 1 May 2018 release)'},
}
MONTHS = {'ja': 1, 'fe': 2, 'mr': 3, 'ap': 4, 'my': 5, 'jn': 6, 'jl': 7, 'au': 8, 'se': 9, 'oc': 10, 'nv': 11, 'de': 12}
IEM = ('https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py?accept=csv'
       '&sts={y}-01-01T00:00Z&ets={y1}-01-01T00:00Z&location_group=states&{states}')
CENSUS = 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv'


def effective_date(tag):
    """bp07oc14 -> 2014-10-07: correlation files are named by their valid (effective) date."""
    return f'20{tag[6:8]}-{MONTHS[tag[4:6]]:02d}-{int(tag[2:4]):02d}'


def download(url, attempts=3):
    """Return (bytes, note). Archive captures may replay a gzip-encoded original response."""
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = response.read()
            if payload[:2] == b'\x1f\x8b':
                return gzip.decompress(payload), 'gzip-decoded'
            return payload, None
        except OSError as error:
            if attempt == attempts:
                raise
            print(f'  retry {attempt} after {error}', flush=True)
            time.sleep(10 * attempt)


def fetch(folder, name, url, validate, extra):
    """Fetch one file unless it already exists with the recorded checksum."""
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files': {}}
    path = folder / name
    if path.exists():
        recorded = manifest['files'].get(name, {}).get('sha256')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if recorded != actual:
            raise SystemExit(f'{path} exists but does not match manifest.json; raw files are never overwritten. '
                             'Delete it to fetch again.')
        print(f'Keeping {name} (checksum verified)', flush=True)
        return
    print(f'Fetching {name}', flush=True)
    payload, note = download(url)
    validate(payload)
    pending = path.with_name(path.name + '.pending')
    pending.write_bytes(payload)
    pending.replace(path)
    manifest['files'][name] = {'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                               'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), **extra}
    if note:
        manifest['files'][name]['transfer'] = note
    write_json(manifest, manifest_path)
    time.sleep(2)


def check_correlation(payload):
    lines = payload.decode('latin-1').splitlines()
    assert len(lines) > 3000, f'unexpectedly short correlation file ({len(lines)} lines)'
    assert all(line.count('|') == 10 for line in lines[:50]), 'expected 11 pipe-delimited fields'


def check_vtec(payload):
    assert payload.startswith(b'wfo,utc_issue,utc_expire,utc_prodissue'), payload[:200]


def check_census(payload):
    assert payload.startswith(b'SUMLEV,REGION,DIVISION,STATE,COUNTY,STNAME,CTYNAME,ESTIMATESBASE2020'), payload[:200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zones', action='store_true')
    parser.add_argument('--alerts', action='store_true')
    parser.add_argument('--census', action='store_true')
    parser.add_argument('--years', default='2014-2025', help='Alert years, e.g. 2014-2025.')
    args = parser.parse_args()
    both = not (args.zones or args.alerts or args.census)
    if args.census or both:
        fetch(RAW_CENSUS_COUNTIES, CENSUS_COUNTY_POP.name, CENSUS, check_census,
              {'vintage': 'Census Bureau county population estimates, Vintage 2023 (2020 base)'})
    if args.zones or both:
        for tag, capture in ZONE_VERSIONS.items():
            url = NWS.format(tag=tag) if capture is None else WAYBACK.format(stamp=capture[0], original=capture[1].format(tag=tag))
            fetch(RAW_ZONE_COUNTY, f'{tag}.dbx', url, check_correlation,
                  {'effective_date': effective_date(tag), 'source': 'weather.gov' if capture is None else 'Internet Archive capture'})
        manifest = json.loads((RAW_ZONE_COUNTY / 'manifest.json').read_text())
        manifest['missing_versions'] = MISSING_VERSIONS
        manifest['format'] = 'STATE|ZONE|CWA|NAME|STATE_ZONE|COUNTY|FIPS|TIME_ZONE|FE_AREA|LAT|LON (https://www.weather.gov/gis/ZoneCounty)'
        write_json(manifest, RAW_ZONE_COUNTY / 'manifest.json')
    if args.alerts or both:
        first, last = (int(y) for y in args.years.split('-'))
        states = '&'.join(f'states={abbr}' for _, abbr in SOUTHEAST_STATES.values())
        for year in range(first, last + 1):
            fetch(RAW_IEM_VTEC, f'vtec_southeast_{year}.csv', IEM.format(y=year, y1=year + 1, states=states), check_vtec,
                  {'events_starting': f'[{year}-01-01, {year + 1}-01-01) UTC', 'states': [a for _, a in SOUTHEAST_STATES.values()]})


if __name__ == '__main__':
    main()
