"""Checkpoint 1: training-period observation reliability, without target analysis.

Run: python scripts/check_observation_reliability.py
Writes summary JSON in docs/ and detailed Parquet summaries under data/processed/eda/.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C
from src.observation_quality import county_windows, summarize_windows

OUTPUT = C.PROCESSED_DIR / "eda/observation_reliability"
HORIZONS = (6, 12)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    years = sorted(C.TRAIN_YEARS)
    if years != list(range(2014, 2024)):
        raise ValueError("This checkpoint is frozen to training years 2014-2023")
    census_path = C.PROCESSED_DIR / "reference/county_census.parquet"
    census = pd.read_parquet(census_path)
    census = census[census.acs_year.isin(years)].copy()
    assert not census.duplicated(["fips_code", "acs_year"]).any()
    paths = [C.PROCESSED_DIR / f"eaglei/outages_{year}.parquet" for year in years]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    envelope_start = pd.read_parquet(paths[0], columns=["run_start_time"]).run_start_time.min()
    envelope_end = pd.Timestamp(years[-1] + 1, 1, 1)
    county_rows, month_rows, hour_rows, slot_rows, raw_rows = [], [], [], [], []
    columns = ["fips_code", "run_start_time", "customers_out"]
    for year, path in zip(years, paths):
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        hours = int((end - start).total_seconds() / 3600)
        raw = pd.read_parquet(path, columns=columns)
        if raw.run_start_time.isna().any():
            raise ValueError("Null timestamps require a separate source-level exclusion audit")
        if not raw.run_start_time.between(start, end, inclusive="left").all():
            raise ValueError("Annual outage file contains out-of-year timestamps")
        county_info = census[census.acs_year.eq(year)].copy()
        if len(county_info) != 534:
            raise ValueError("Expected the full 534-county study universe")
        county_info["housing_quartile"] = pd.qcut(county_info.housing_units, 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"])
        if not raw.fips_code.isin(county_info.fips_code).all():
            raise ValueError("Outage county absent from Census study universe")
        # Read only a short next-year buffer, and never open validation/test data.
        if year < years[-1]:
            buffer = pd.read_parquet(C.PROCESSED_DIR / f"eaglei/outages_{year+1}.parquet", columns=columns,
                                     filters=[("run_start_time", "<=", end + pd.Timedelta(hours=max(HORIZONS)))])
            readings = pd.concat([raw, buffer], ignore_index=True)
        else:
            readings = raw
        grouped = readings.groupby("fips_code", sort=False)
        raw_grouped = raw.groupby("fips_code", sort=False)
        origins = pd.date_range(start, periods=hours, freq="h")
        next_housing = census[census.acs_year.eq(year + 1)].set_index("fips_code").housing_units
        for county in county_info.itertuples():
            fips = county.fips_code
            county_readings = grouped.get_group(fips) if fips in grouped.indices else readings.iloc[:0]
            current = raw_grouped.get_group(fips) if fips in raw_grouped.indices else raw.iloc[:0]
            values = current.customers_out.to_numpy(dtype=float)
            stamps = current.run_start_time.astype("datetime64[ns]").astype("int64").to_numpy()
            finite = np.isfinite(values)
            numeric_valid = finite & (values >= 0) & (values == np.floor(np.where(finite, values, 0)))
            raw_rows.append({"year": year, "fips_code": fips, "state": C.STATES[fips[:2]],
                "rows": len(current), "null": int(np.isnan(values).sum()), "zero": int((values == 0).sum()),
                "invalid": int((~np.isnan(values) & ~numeric_valid).sum()),
                "off_grid": int((stamps % pd.Timedelta(minutes=15).value != 0).sum()),
                "duplicate_rows": int(current.duplicated("run_start_time").sum()),
                "proxy_flags": int((numeric_valid & (values > 1.2 * county.housing_units)).sum()),
                "first": str(current.run_start_time.min()), "last": str(current.run_start_time.max())})
            windows = county_windows(county_readings, start=start, hours=hours,
                envelope_start=envelope_start, envelope_end=envelope_end,
                housing_units=county.housing_units, next_housing_units=next_housing.get(fips), horizons=HORIZONS)
            for horizon, window in windows.items():
                keys = {"year": year, "horizon": horizon, "fips_code": fips, "state": C.STATES[fips[:2]]}
                county_rows.append({**keys, "county_name": county.county_name,
                    "housing_units": int(county.housing_units), "housing_quartile": str(county.housing_quartile),
                    **summarize_windows(window, horizon)})
                for month in range(1, 13):
                    month_rows.append({**keys, "month": month,
                        **summarize_windows(window, horizon, origins.month == month)})
                for hour in range(24):
                    hour_rows.append({**keys, "hour": hour,
                        **summarize_windows(window, horizon, origins.hour == hour)})
                histogram = np.bincount(window["usable"][window["in_bounds"]], minlength=horizon * 4 + 1)
                for slots, count in enumerate(histogram):
                    if count:
                        slot_rows.append({"year": year, "state": keys["state"], "horizon": horizon,
                                          "usable_slots": slots, "windows": int(count)})
        print(f"Completed {year}: {len(raw):,} rows, all 534 counties", flush=True)
        del raw, readings, grouped, raw_grouped

    counties = pd.DataFrame(county_rows)
    measures = [c for c in counties.columns if c not in {
        "year", "horizon", "fips_code", "state", "county_name", "housing_units", "housing_quartile"}]
    def aggregate(frame, keys):
        return frame.groupby(keys, observed=True)[measures].sum().reset_index()
    months = aggregate(pd.DataFrame(month_rows), ["year", "state", "horizon", "month"])
    hours_df = aggregate(pd.DataFrame(hour_rows), ["year", "state", "horizon", "hour"])
    slots = pd.DataFrame(slot_rows).groupby(["year", "state", "horizon", "usable_slots"]).windows.sum().reset_index()
    raw_summary = pd.DataFrame(raw_rows)
    frames = {"county_year": counties, "state_month": months, "state_hour": hours_df,
              "slot_distribution": slots, "raw_county_year": raw_summary}
    for name, frame in frames.items():
        frame.to_parquet(OUTPUT / f"{name}.parquet", index=False)
    # These identities validate every county-year/horizon before reporting totals.
    assert (counties.unknown + counties.partial + counties.complete == counties.in_bounds).all()
    assert (counties.in_bounds + counties.boundary_excluded == counties.origins).all()
    assert (counties.complete_unflagged <= counties.complete).all()
    assert (counties.usable_window_slots <= counties.expected_slots).all()
    assert np.array_equal(aggregate(months, ["year", "horizon"])[measures].to_numpy(),
                          aggregate(counties, ["year", "horizon"])[measures].to_numpy())
    assert np.array_equal(aggregate(hours_df, ["year", "horizon"])[measures].to_numpy(),
                          aggregate(counties, ["year", "horizon"])[measures].to_numpy())
    source_paths = [*paths, census_path]
    metadata = {
        "checkpoint": 1, "training_years": years, "horizons_hours": HORIZONS,
        "time_basis": "UTC", "window": "origin < timestamp <= origin + horizon",
        "origin_frame": "Every hour in calendar years 2014-2023 for all 534 Census counties",
        "envelope_start": str(envelope_start), "envelope_end_exclusive": str(envelope_end),
        "boundary_rule": "Exclude origins before envelope start and windows ending at/after exclusive cutoff; other county/file gaps remain missing",
        "complete_rule": "24 or 48 distinct exact quarter-hour slots, each with one finite nonnegative integer count; explicit zero is usable",
        "flag_rule": "Any off-grid reading, duplicate slot, invalid count or count above 1.2 x same-year housing units; proxy flag is not proven error",
        "aggregation_note": "Window-slot counts repeat observations across overlapping windows; raw_county_year counts original rows once",
        "housing_quartiles": "Within each training year over all 534 counties; retrospective composition diagnostic only",
        "holdout_access": "No 2024/2025 outage files read; no target thresholds or model fitting",
        "code_files": [
            {"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
            for name in ["src/observation_quality.py", "scripts/check_observation_reliability.py"]
        ],
        "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths],
        "output_directory": OUTPUT.relative_to(ROOT).as_posix(),
        "totals": aggregate(counties, ["horizon"]).to_dict("records"),
        "by_year": aggregate(counties, ["year", "horizon"]).to_dict("records"),
        "by_state": aggregate(counties, ["state", "horizon"]).to_dict("records"),
        "by_housing_quartile": aggregate(counties, ["housing_quartile", "horizon"]).to_dict("records"),
        "raw_totals": raw_summary[["rows", "null", "zero", "invalid", "off_grid", "duplicate_rows", "proxy_flags"]].sum().to_dict(),
        "counties_never_stored": raw_summary.groupby("fips_code").rows.sum().loc[lambda s:s.eq(0)].index.tolist(),
    }
    target = ROOT / "docs/checkpoint_1_observation_reliability.json"
    target.write_text(json.dumps(metadata, indent=2, default=int) + "\n", encoding="utf-8")
    print(json.dumps(metadata["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
