"""Checkpoint 2. Run with python scripts/check_customer_targets.py."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C
from src.outage_targets import THRESHOLDS, county_targets, summarize_targets

OUTPUT = C.PROCESSED_DIR / 'eda/customer_targets'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((ROOT / 'docs/checkpoint_1_observation_reliability.json').read_text())
    if not prior.get('code_files'):
        raise ValueError('Rebuild observation summaries: code fingerprints are missing')
    for item in prior['files'] + prior['code_files']:
        if hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Checkpoint 1 source changed: ' + item['path'])
    years = list(range(2014, 2024))
    census = pd.read_parquet(C.PROCESSED_DIR / 'reference/county_census.parquet')
    census = census[census.acs_year.isin(years)]
    rows, seasonal, peak_chunks = [], [], {6: [], 12: []}
    columns = ['fips_code', 'run_start_time', 'customers_out']
    for year in years:
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        hours = int((end - start).total_seconds() / 3600)
        raw = pd.read_parquet(C.PROCESSED_DIR / f'eaglei/outages_{year}.parquet', columns=columns)
        if year < 2023:
            buffer = pd.read_parquet(C.PROCESSED_DIR / f'eaglei/outages_{year+1}.parquet', columns=columns,
                                    filters=[('run_start_time', '<=', end + pd.Timedelta(hours=12))])
            raw = pd.concat([raw, buffer], ignore_index=True)
        grouped = raw.groupby('fips_code', sort=False)
        info = census[census.acs_year.eq(year)].copy()
        assert len(info) == 534 and info.fips_code.is_unique
        info['housing_quartile'] = pd.qcut(info.housing_units, 4, labels=['Q1 smallest', 'Q2', 'Q3', 'Q4 largest'])
        next_housing = census[census.acs_year.eq(year + 1)].set_index('fips_code').housing_units
        months = pd.date_range(start, periods=hours, freq='h').month.to_numpy()
        seasons = {'DJF': np.isin(months, [12, 1, 2]), 'MAM': np.isin(months, [3, 4, 5]),
                   'JJA': np.isin(months, [6, 7, 8]), 'SON': np.isin(months, [9, 10, 11])}
        for county in info.itertuples():
            readings = grouped.get_group(county.fips_code) if county.fips_code in grouped.indices else raw.iloc[:0]
            windows = county_targets(readings, start=start, hours=hours,
                envelope_start=prior['envelope_start'], envelope_end=prior['envelope_end_exclusive'],
                housing_units=county.housing_units, next_housing_units=next_housing.get(county.fips_code))
            for horizon, window in windows.items():
                strict = window['in_bounds'] & (window['usable'] == horizon * 4) & ~window['flagged']
                peak_chunks[horizon].append(window['recorded_peak'][strict])
                keys = dict(year=year, horizon=horizon, fips_code=county.fips_code, state=C.STATES[county.fips_code[:2]],
                            county_name=county.county_name, housing_quartile=str(county.housing_quartile))
                for threshold in THRESHOLDS:
                    rows.append({**keys, 'threshold': threshold, **summarize_targets(window, horizon, threshold)})
                    for season, mask in seasons.items():
                        seasonal.append(dict(year=year, state=keys['state'], horizon=horizon, threshold=threshold,
                                             season=season, **summarize_targets(window, horizon, threshold, mask)))
        print(f'Completed target analysis: {year}', flush=True)
    county_frame = pd.DataFrame(rows)
    measures = list(summarize_targets(window, horizon, 5000))
    def aggregate(frame, keys):
        return frame.groupby(keys, observed=True)[measures].sum().reset_index()
    season_frame = aggregate(pd.DataFrame(seasonal), ['year', 'state', 'horizon', 'threshold', 'season'])
    county_frame.to_parquet(OUTPUT / 'county_year.parquet', index=False)
    season_frame.to_parquet(OUTPUT / 'state_season.parquet', index=False)
    national = aggregate(county_frame, ['horizon', 'threshold'])
    for item in prior['totals']:
        check = national[national.horizon.eq(item['horizon'])]
        assert check.in_bounds.eq(item['in_bounds']).all()
        assert check.strict.eq(item['complete_unflagged']).all()
        assert check.complete.eq(item['complete']).all()
    assert np.array_equal(aggregate(season_frame, ['horizon', 'threshold'])[measures], national[measures])
    quantiles = []
    for h, chunks in peak_chunks.items():
        peaks = np.concatenate(chunks)
        quantiles.append({'horizon': h, 'n': len(peaks), 'mean': float(peaks.mean()),
                         **{str(q): float(np.quantile(peaks, q)) for q in [0, .25, .5, .75, .9, .95, .99, 1]}})
    metadata = dict(checkpoint=2, training_years=years, files=prior['files'],
        window='origin < timestamp <= origin + horizon; exact UTC quarter-hour grid',
        primary_threshold='strictly greater than 5000 customers', thresholds=list(THRESHOLDS),
        peak='Uncapped maximum of unique finite nonnegative integer exact-grid readings',
        hourly_median='Maximum of medians within (hour, hour+1h]; partial-hour medians use available readings',
        proxy_exclusion='Remove counts above 1.2 times same-year housing; never fill resulting gaps',
        cohort='Complete and unflagged windows; future completeness is an outcome audit, not a model feature',
        interpretation='Overlapping county-hour windows, not independent outage events or population prevalence',
        totals=national.to_dict('records'), strict_peak_quantiles=quantiles,
        by_year=aggregate(county_frame, ['year', 'horizon', 'threshold']).to_dict('records'),
        by_state=aggregate(county_frame, ['state', 'horizon', 'threshold']).to_dict('records'),
        by_housing_quartile=aggregate(county_frame, ['housing_quartile', 'horizon', 'threshold']).to_dict('records'),
        by_season=aggregate(season_frame, ['season', 'horizon', 'threshold']).to_dict('records'))
    code_paths = ['src/outage_targets.py', 'src/observation_quality.py', 'scripts/check_customer_targets.py',
                  'scripts/check_observation_reliability.py']
    metadata['code_files'] = [{'path': p, 'sha256': hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in code_paths]
    (ROOT / 'docs/checkpoint_2_customer_targets.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(national.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
