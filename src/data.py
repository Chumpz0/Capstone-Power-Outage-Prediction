"""
Load project data from one place, whoever you are and wherever you run.

    from src.data import load_outages, load_storm_counties, load_census

    out = load_outages([2024], columns=["fips_code", "customers_out", "run_start_time"])
    storms = load_storm_counties()

Each loader reads data/processed/ on your machine if the file is there, and
otherwise streams it from the team's private Hugging Face dataset
(config.HF_REPO_ID). For Hugging Face you need to be logged in once:
    hf auth login        (paste a READ token from huggingface.co/settings/tokens)
In Colab you can instead set the HF_TOKEN secret.
"""
from __future__ import annotations

import pandas as pd

from src import config as C


def path(rel: str) -> str:
    """Local path if the file exists, else the hf:// path on the shared dataset."""
    local = C.PROCESSED_DIR / rel
    if local.exists():
        return str(local)
    return f"hf://datasets/{C.HF_REPO_ID}/{rel}"


def duckdb_path(rel: str) -> str:
    """Like path(), but for DuckDB SQL: forward slashes, and `rel` may be a glob
    such as "eaglei/outages_*.parquet"."""
    if any(C.PROCESSED_DIR.glob(rel)):
        return (C.PROCESSED_DIR / rel).as_posix()
    return f"hf://datasets/{C.HF_REPO_ID}/{rel}"


def _read(rel: str, **kw) -> pd.DataFrame:
    p = path(rel)
    return pd.read_csv(p, **kw) if rel.endswith(".csv") else pd.read_parquet(p, **kw)


def load_outages(years: list[int] | None = None, columns: list[str] | None = None,
                 states: list[str] | None = None) -> pd.DataFrame:
    """EAGLE-I 15-min county outages. `states` = 2-letter codes, e.g. ["GA"].
    One year is ~10-20M rows, so start with one or two years and only the columns you need."""
    years = years or C.YEARS
    if columns and states and "fips_code" not in columns:
        columns = ["fips_code", *columns]
    frames = []
    for y in years:
        df = _read(f"eaglei/outages_{y}.parquet", columns=columns)
        if states:
            fips = [k for k, v in C.STATES.items() if v in states]
            df = df[df["fips_code"].str[:2].isin(fips)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_storm_counties(columns: list[str] | None = None) -> pd.DataFrame:
    """One row per (NOAA storm event, affected county), all years."""
    return _read("storms/storm_event_counties.parquet", columns=columns)


def load_storm_events(year: int) -> pd.DataFrame:
    """Raw NOAA events for one year (study-area states, before county mapping)."""
    return _read(f"storms/storm_events_{year}.parquet")


def load_census() -> pd.DataFrame:
    return _read("reference/county_census.parquet")


def load_zone_county() -> pd.DataFrame:
    return _read("reference/zone_county.parquet")


def load_customer_counts() -> pd.DataFrame:
    """EAGLE-I modeled customer count per county (to get % of customers out).
    Adds a 5-digit `fips_code` and makes `Customers` numeric."""
    df = _read("eaglei/MCC.csv", dtype=str)
    df["fips_code"] = df["County_FIPS"].str.zfill(5)
    df["Customers"] = pd.to_numeric(df["Customers"])
    return df


def load_coverage_history() -> pd.DataFrame:
    return _read("eaglei/coverage_history.csv")


def load_dqi() -> pd.DataFrame:
    """EAGLE-I data quality index per FEMA region and year."""
    return _read("eaglei/DQI.csv")
