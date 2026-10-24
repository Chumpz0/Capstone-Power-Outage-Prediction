"""Quick sanity check of the processed data. Run after download_data.py:
    python scripts/check_data.py
"""
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

P = C.PROCESSED_DIR


def eaglei_summary() -> None:
    print("\n== EAGLE-I outages (15-min, study-area counties) ==")
    rows = []
    for f in sorted((P / "eaglei").glob("outages_*.parquet")):
        if int(f.stem.split('_')[1]) >= 2025:
            print(f'  {f.name}: held-out file; outcome summaries skipped')
            continue
        df = pd.read_parquet(f, columns=["fips_code", "customers_out", "run_start_time"])
        rows.append({
            "year": f.stem.split("_")[1],
            "rows": len(df),
            "counties": df["fips_code"].nunique(),
            "first": df["run_start_time"].min(),
            "last": df["run_start_time"].max(),
            "max_customers_out": int(df["customers_out"].max()),
            "null_customers_out": int(df["customers_out"].isna().sum()),
        })
    print(pd.DataFrame(rows).to_string(index=False) if rows else "  none yet")


def storm_summary() -> None:
    print("\n== NOAA Storm Events ==")
    f = P / "storms" / "storm_event_counties.parquet"
    if not f.exists():
        print("  none yet")
        return
    ev = pd.read_parquet(f, columns=["event_id", "year", "event_type", "mapped_from"])
    print(ev.groupby("year")["event_id"].nunique().rename("events").to_string())
    print("\nTop event types (event-county rows):")
    print(ev["event_type"].value_counts().head(12).to_string())
    print("\nMapped from:", ev["mapped_from"].value_counts().to_dict())


def census_summary() -> None:
    print("\n== Census ==")
    f = P / "reference" / "county_census.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        print(df.groupby("acs_year")["fips_code"].nunique().rename("counties").to_string())
    else:
        print("  none yet")


def sizes() -> None:
    print("\n== Disk use ==")
    for f in sorted(P.rglob("*.parquet")):
        print(f"  {f.relative_to(C.ROOT)}  {f.stat().st_size / 1e6:,.1f} MB  "
              f"({pq.ParquetFile(f).metadata.num_rows:,} rows)")


if __name__ == "__main__":
    eaglei_summary()
    storm_summary()
    census_summary()
    sizes()
