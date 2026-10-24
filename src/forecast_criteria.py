"""Chronological assignment for future outcome windows; no model fitting."""
import pandas as pd


def split_origins(origins, horizon):
    """Exclude windows ending at/after each exclusive split boundary.

    This conservatively assigns the boundary observation to the next split.
    Use naive UTC timestamps, consistent with the processed outage files.
    """
    if horizon not in (6, 12):
        raise ValueError("Expected a 6-hour or 12-hour horizon")
    times = pd.DatetimeIndex(origins)
    if times.tz is not None:
        raise ValueError("Convert timestamps to naive UTC before assignment")
    result = pd.Series("excluded", index=times, dtype="str")
    for name, start, end in [
        ("train", "2014-11-01 04:00", "2024-01-01"),
        ("validation", "2024-01-01", "2025-01-01"),
        ("test", "2025-01-01", "2026-01-01"),
    ]:
        keep = (times >= pd.Timestamp(start)) & (times + pd.Timedelta(hours=horizon) < pd.Timestamp(end))
        result.iloc[keep] = name
    return result
