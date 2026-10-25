"""Uncapped future customer-count targets, with explicit observation quality."""
import numpy as np
import pandas as pd

from src.observation_quality import prepare_county_grid, window_quality

THRESHOLDS = (1000, 2500, 5000, 10000)


def county_targets(readings, *, start, hours, envelope_start, envelope_end,
                   housing_units, next_housing_units=None, horizons=(6, 12)):
    """Return quality, recorded peaks and sensitivity peaks for each horizon.

    Primary peaks use unique valid exact-grid readings. Off-grid observations
    flag a window; they do not silently become quarter-hour measurements.
    Hourly medians use (hour, hour + 1h], matching the target boundaries.
    Proxy exclusions remove, never cap, counts above 1.2 times housing units.
    Readings in the year after start use next_housing_units when supplied. All out-of-envelope peaks remain NaN.
    """
    arrays, grid, excluded = prepare_county_grid(
        readings, start=start, hours=hours, housing_units=housing_units,
        next_housing_units=next_housing_units, horizons=horizons)
    quality = window_quality(arrays, start=start, hours=hours,
                             envelope_start=envelope_start, envelope_end=envelope_end,
                             horizons=horizons)
    # Pandas median keeps all-missing hours missing without warning or filling.
    hourly = pd.DataFrame(grid[1:].reshape(-1, 4)).median(axis=1).to_numpy()
    for horizon, window in quality.items():
        ends = np.arange(hours) * 4 + horizon * 4
        for name, array, width, indices in [
            ('recorded_peak', grid, horizon * 4, ends),
            ('excluded_peak', excluded, horizon * 4, ends),
            ('median_peak', hourly, horizon, np.arange(hours) + horizon - 1),
        ]:
            result = pd.Series(array).rolling(width, min_periods=1).max().to_numpy()[indices]
            result[~window['in_bounds']] = np.nan
            window[name] = result
        cumulative = np.r_[0, np.cumsum(np.isfinite(excluded))]
        window['excluded_usable'] = cumulative[ends + 1] - cumulative[np.arange(hours) * 4 + 1]
    return quality


def labels(window, horizon, threshold, *, exclude_proxy=False):
    """1 = observed exceedance; 0 = complete negative; NaN = uncertain.

    Flags remain separate: these labels alone do not define the model cohort.
    """
    peak = window['excluded_peak' if exclude_proxy else 'recorded_peak']
    usable = window['excluded_usable' if exclude_proxy else 'usable']
    result = np.full(len(peak), np.nan)
    result[window['in_bounds'] & (usable == 4 * horizon)] = 0
    result[window['in_bounds'] & (peak > threshold)] = 1
    return result


def summarize_targets(window, horizon, threshold, mask=None):
    eligible = window['in_bounds'].copy()
    if mask is not None:
        eligible &= mask
    complete = window['usable'] == horizon * 4
    strict = eligible & complete & ~window['flagged']
    partial = eligible & (window['usable'] > 0) & ~complete
    positive = window['recorded_peak'] > threshold
    count = lambda a: int(np.count_nonzero(a))
    result = {
        'in_bounds': count(eligible),
        'unknown': count(eligible & (window['usable'] == 0)),
        'complete': count(eligible & complete),
        'complete_positive': count(eligible & complete & positive),
        'strict': count(strict), 'strict_positive': count(strict & positive),
        'strict_negative': count(strict & ~positive),
        'partial_positive_unflagged': count(partial & positive & ~window['flagged']),
        'partial_positive_flagged': count(partial & positive & window['flagged']),
        'partial_uncertain': count(partial & ~positive),
        'median_positive_strict': count(strict & (window['median_peak'] > threshold)),
        'median_lost_positive_strict': count(strict & positive & (window['median_peak'] <= threshold)),
        'proxy_excluded_positive_complete': count(eligible & complete & (window['excluded_peak'] > threshold)),
        'proxy_lost_positive_complete': count(eligible & complete & positive & ~(window['excluded_peak'] > threshold)),
        'proxy_uncertain_complete': count(eligible & complete & np.isnan(labels(window, horizon, threshold, exclude_proxy=True))),
    }
    assert result['unknown'] + result['complete'] + result['partial_positive_unflagged'] + result['partial_positive_flagged'] + result['partial_uncertain'] == result['in_bounds']
    return result
