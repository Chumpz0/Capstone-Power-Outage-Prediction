"""Observation completeness only: no outage targets or model features.

Windows are (origin, origin + horizon], sampled on the exact UTC quarter-hour
grid. Duplicate slots and nonfinite/negative/fractional counts are not usable.
An exceedance of the housing-based customer proxy is a flag, not a correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUARTER_NS = pd.Timedelta(minutes=15).value


def prepare_county_grid(readings, *, start, hours, housing_units,
                        next_housing_units=None, horizons=(6, 12)):
    """Prepare shared quality counts and peaks on the UTC quarter-hour grid.

    Readings in start's calendar year use housing_units. The following year
    uses next_housing_units, or housing_units when none is supplied.
    Return slot counts, unique valid readings and proxy-excluded readings.
    """
    if (not isinstance(hours, (int, np.integer)) or hours <= 0 or not horizons
            or any(not isinstance(h, (int, np.integer)) or h <= 0 for h in horizons)):
        raise ValueError("Hours and horizons must be positive whole hours")
    start = pd.Timestamp(start)
    if start.tz is not None or start != start.floor("h"):
        raise ValueError("Origins must start on an exact hour in naive UTC")
    year_end = pd.Timestamp(start.year + 1, 1, 1)
    if start + pd.Timedelta(hours=hours + max(horizons)) >= pd.Timestamp(start.year + 2, 1, 1):
        raise ValueError("Only the starting and following housing years are supported")
    length = hours * 4 + max(horizons) * 4 + 1
    time = pd.to_datetime(readings["run_start_time"])
    has_time = time.notna().to_numpy()
    offsets = (time[has_time].astype("datetime64[ns]").astype("int64").to_numpy()
               - start.value)
    values = readings.loc[has_time, "customers_out"].to_numpy(dtype=float)
    slot = offsets // QUARTER_NS
    on_grid = offsets % QUARTER_NS == 0
    in_range = (slot >= 0) & (slot < length)
    aligned = on_grid & in_range
    finite = np.isfinite(values)
    valid = finite & (values >= 0) & (values == np.floor(np.where(finite, values, 0)))
    next_year_slot = (year_end.value - start.value) // QUARTER_NS
    proxy = 1.2 * np.where(
        np.arange(length) < next_year_slot, housing_units,
        housing_units if next_housing_units is None else next_housing_units)
    above_proxy = np.zeros(len(values), dtype=bool)
    above_proxy[aligned] = values[aligned] > proxy[slot[aligned]]

    def counts(mask):
        return np.bincount(slot[mask], minlength=length)[:length]

    present = counts(aligned)
    usable = counts(aligned & valid)
    usable[present != 1] = 0
    zeros = counts(aligned & valid & (values == 0))
    zeros[present != 1] = 0
    arrays = {
        "usable": usable,
        "present": (present > 0).astype(int),
        "null": (counts(aligned & np.isnan(values)) > 0).astype(int),
        "invalid": (counts(aligned & ~np.isnan(values) & ~valid) > 0).astype(int),
        "duplicate": (present > 1).astype(int),
        "zero": zeros,
        "proxy_flag": (counts(aligned & valid & above_proxy) > 0).astype(int),
    }
    # Ceil off-grid timestamps so an hourly boundary still includes exactly the
    # readings in (origin, end], without treating those readings as aligned slots.
    off_slot = (offsets[~on_grid] + QUARTER_NS - 1) // QUARTER_NS
    off_slot = off_slot[(off_slot >= 0) & (off_slot < length)]
    arrays["off_grid"] = np.bincount(off_slot, minlength=length)[:length]
    selected = np.flatnonzero(aligned & valid)
    selected = selected[present[slot[selected]] == 1]
    grid = np.full(length, np.nan)
    grid[slot[selected]] = values[selected]
    excluded = grid.copy()
    excluded[grid > proxy] = np.nan
    return arrays, grid, excluded


def window_quality(arrays, *, start, hours, envelope_start, envelope_end, horizons):
    """Sum prepared slot counts over (origin, origin + horizon]."""
    start = pd.Timestamp(start)
    origins = np.arange(hours) * 4
    origin_times = start.value + np.arange(hours) * pd.Timedelta(hours=1).value
    results = {}
    for horizon in horizons:
        end = origins + horizon * 4
        window = {}
        for key, array in arrays.items():
            cumulative = np.r_[0, np.cumsum(array)]
            window[key] = cumulative[end + 1] - cumulative[origins + 1]
        window["in_bounds"] = (
            (origin_times >= pd.Timestamp(envelope_start).value)
            & (origin_times + pd.Timedelta(hours=horizon).value < pd.Timestamp(envelope_end).value))
        window["flagged"] = ((window["proxy_flag"] + window["invalid"]
                              + window["duplicate"] + window["off_grid"]) > 0)
        results[horizon] = window
    return results


def county_windows(readings, *, start, hours, envelope_start, envelope_end,
                   housing_units, next_housing_units=None, horizons=(6, 12)):
    """Return observation quality arrays for each horizon and hourly origin.

    Timestamps are naive UTC; envelope_end is exclusive. Readings in the year
    after start use next_housing_units when supplied, independent of the
    length of the origin period.
    """
    arrays, _, _ = prepare_county_grid(
        readings, start=start, hours=hours, housing_units=housing_units,
        next_housing_units=next_housing_units, horizons=horizons)
    return window_quality(arrays, start=start, hours=hours,
                          envelope_start=envelope_start, envelope_end=envelope_end,
                          horizons=horizons)


def summarize_windows(window, horizon, mask=None):
    """Counts for the in-bounds population; boundary counts use the full frame."""
    selected = np.ones(len(window["usable"]), dtype=bool) if mask is None else mask
    eligible = selected & window["in_bounds"]
    n = window["usable"]
    complete = n == horizon * 4
    result = {
        "origins": int(selected.sum()),
        "boundary_excluded": int((selected & ~window["in_bounds"]).sum()),
        "in_bounds": int(eligible.sum()),
        "unknown": int((eligible & (n == 0)).sum()),
        "partial": int((eligible & (n > 0) & ~complete).sum()),
        "complete": int((eligible & complete).sum()),
        "complete_unflagged": int((eligible & complete & ~window["flagged"]).sum()),
        "at_least_half": int((eligible & (n >= horizon * 2)).sum()),
        "at_least_90pct": int((eligible & (n >= np.ceil(horizon * 4 * .9))).sum()),
        "any_observed": int((eligible & (n > 0)).sum()),
        "flagged_windows": int((eligible & window["flagged"]).sum()),
        "expected_slots": int(eligible.sum()) * horizon * 4,
    }
    for key in ["usable", "present", "null", "invalid", "duplicate", "zero", "proxy_flag", "off_grid"]:
        result[f"{key}_window_slots"] = int(window[key][eligible].sum())
        result[f"{key}_windows"] = int((eligible & (window[key] > 0)).sum())
    return result
