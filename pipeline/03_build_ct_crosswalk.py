"""Build population weights that allocate Connecticut planning regions to legacy counties.

Connecticut replaced its 8 counties with 9 planning regions (FIPS 09110-09190).
EAGLE-I switched to the regions on 2025-05-29, while NOAA keeps the legacy counties.
Regions cross county lines, so each region is split by the 2020 census population
of its towns (county subdivisions).

Run: python pipeline/03_build_ct_crosswalk.py   (downloads Census files if they are missing)
Output: data/reference/ct_region_to_county_weights.csv
"""
import csv
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from paths import CENSUS_CT as SRC, CT_WEIGHTS as OUT
SOURCES = {
    'ct_cou_to_cousub_crosswalk.xlsx': 'https://www2.census.gov/geo/docs/reference/ct_change/ct_cou_to_cousub_crosswalk.xlsx',
    'sub-est2023_9.csv': 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/cities/totals/sub-est2023_9.csv',
}
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def download():
    SRC.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (SRC / name).exists():
            urllib.request.urlretrieve(url, SRC / name)


def read_xlsx_rows(path):
    """Read the first worksheet with the standard library (no openpyxl dependency)."""
    z = zipfile.ZipFile(path)
    shared = [''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS)]
    rows = []
    for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).iter(f"{{{NS['m']}}}row"):
        values = []
        for cell in row.findall('m:c', NS):
            v = cell.find('m:v', NS)
            if v is None:
                inline = cell.find('m:is', NS)
                values.append(''.join(inline.itertext()) if inline is not None else '')
            else:
                values.append(shared[int(v.text)] if cell.get('t') == 's' else v.text)
        rows.append(values)
    header = [h.split('\n')[0] for h in rows[0]]
    # Footnote rows below the table have fewer cells; keep only Connecticut town rows.
    return [dict(zip(header, r)) for r in rows[1:] if len(r) == len(header) and r[0] == '09']


def main():
    download()
    # Unpopulated water areas appear as 'County subdivisions not defined' (code 00000).
    towns = [t for t in read_xlsx_rows(SRC / 'ct_cou_to_cousub_crosswalk.xlsx') if t['COUSUBFP'] != '00000']
    with open(SRC / 'sub-est2023_9.csv', encoding='latin-1', newline='') as f:
        pop = {r['COUSUB']: int(r['ESTIMATESBASE2020']) for r in csv.DictReader(f) if r['SUMLEV'] == '061'}
    region_county = defaultdict(int)
    region_total = defaultdict(int)
    missing = []
    for t in towns:
        if t['COUSUBFP'] not in pop:
            missing.append(t['COUSUB_NAMELSAD'])
            continue
        region = '09' + t['NEW_COUNTYFP']
        county = '09' + t['OLD_COUNTYFP']
        region_county[region, county] += pop[t['COUSUBFP']]
        region_total[region] += pop[t['COUSUBFP']]
    assert len(towns) == 169, f'Expected 169 Connecticut towns, found {len(towns)}'
    assert not missing, f'Towns without population: {missing}'
    rows = [{'region_fips': r, 'county_fips': c, 'population_2020': p, 'weight': p / region_total[r]}
            for (r, c), p in sorted(region_county.items())]
    assert sorted({r['region_fips'] for r in rows}) == [f'09{n}' for n in range(110, 200, 10)]
    assert sorted({r['county_fips'] for r in rows}) == [f'09{n:03d}' for n in range(1, 16, 2)]
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
