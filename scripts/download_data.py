"""
Download and filter all project data to the Southeast study area.

Run from the repo root:
    python scripts/download_data.py                 # everything, all years
    python scripts/download_data.py --years 2024 2025
    python scripts/download_data.py --only storms census
    python scripts/download_data.py --keep-raw      # keep the big raw CSVs

Sources
  eaglei  : EAGLE-I county outages every 15 min (ORNL, figshare)  ~1.2 GB/yr raw
  storms  : NOAA Storm Events details files                         ~10 MB/yr
  zones   : NWS forecast-zone -> county lookup (to map zone storms to counties)
  census  : ACS 5-yr county population + housing units, county land area
  eia     : EIA-861 utility data zips (optional, raw only; use --only eia)

Outputs land in data/processed/ as Parquet. Re-running skips finished files.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

HEADERS = {"User-Agent": "GSU-DSCI4350-capstone-power-outage/1.0"}
STATE_FIPS = set(C.STATES)                      # {"01", "12", ...}
ABBR_TO_FIPS = {v: k for k, v in C.STATES.items()}
TZ_FALLBACK = {"EST": -5, "CST": -6, "MST": -7, "PST": -8, "AST": -4,
               "AKST": -9, "HST": -10, "EDT": -4, "CDT": -5}


# --------------------------------------------------------------------------- helpers
def download(url: str, dest: Path, expected_size: int | None = None,
             retries: int = 10) -> Path:
    """Stream a file to disk with a progress bar. Skips if already complete.
    A dropped connection resumes from the .part file (HTTP Range) instead of failing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"  already have {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if expected_size and have == expected_size:
            break
        headers = {**HEADERS, **({"Range": f"bytes={have}-"} if have else {})}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                r.raise_for_status()
                if have and r.status_code != 206:   # server ignored Range: start over
                    have = 0
                total = (have + int(r.headers.get("content-length", 0))) or expected_size
                with open(tmp, "ab" if have else "wb") as f, \
                        tqdm(total=total, initial=have, unit="B", unit_scale=True,
                             desc=dest.name, leave=False) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            if not expected_size or tmp.stat().st_size == expected_size:
                break
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError,
                requests.Timeout) as e:
            if attempt == retries:
                raise
            print(f"  {dest.name}: connection dropped ({e.__class__.__name__}), "
                  f"resuming (attempt {attempt + 1}/{retries})")
            time.sleep(min(60, 5 * attempt))
    tmp.replace(dest)
    return dest


def get_json(url: str, **params):
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- EAGLE-I
def figshare_files() -> dict[str, dict]:
    files = get_json(f"https://api.figshare.com/v2/articles/{C.FIGSHARE_ARTICLE_ID}/files",
                     page_size=100)
    return {f["name"]: f for f in files}


def normalize_eaglei_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across releases and keep study-area counties."""
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"sum": "customers_out", "fips": "fips_code"})
    df["fips_code"] = (pd.to_numeric(df["fips_code"], errors="coerce")
                       .astype("Int64").astype(str).str.zfill(5))
    df = df[df["fips_code"].str[:2].isin(STATE_FIPS)].copy()
    df["customers_out"] = pd.to_numeric(df["customers_out"], errors="coerce").astype("float32")
    df["run_start_time"] = pd.to_datetime(df["run_start_time"], errors="coerce")
    keep = ["fips_code", "county", "state", "customers_out", "run_start_time"]
    return df[[c for c in keep if c in df.columns]]


def filter_eaglei_csv(src: Path, dest: Path, chunksize: int = 2_000_000) -> int:
    """Read a big EAGLE-I CSV in chunks and write only study-area rows to Parquet."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.part")
    writer, rows = None, 0
    for chunk in tqdm(pd.read_csv(src, chunksize=chunksize, low_memory=False),
                      desc=f"filter {src.name}", unit="chunk", leave=False):
        out = normalize_eaglei_chunk(chunk)
        if out.empty:
            continue
        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table.cast(writer.schema))
        rows += len(out)
    if writer:
        writer.close()
        tmp.replace(dest)
    return rows


def get_eaglei(years: list[int], keep_raw: bool) -> None:
    print("\n[EAGLE-I outages]")
    files = figshare_files()
    out_dir = C.PROCESSED_DIR / "eaglei"
    # small reference files: modeled customer counts, coverage, data quality index
    for name in ["MCC.csv", "coverage_history.csv", "DQI.csv"]:
        if name in files:
            download(files[name]["download_url"], out_dir / name, files[name]["size"])
    for year in years:
        name = f"eaglei_outages_{year}.csv"
        dest = out_dir / f"outages_{year}.parquet"
        if dest.exists():
            print(f"  {dest.name} done, skipping")
            continue
        if name not in files:
            print(f"  WARNING: {name} not on figshare yet, skipping")
            continue
        raw = download(files[name]["download_url"], C.RAW_DIR / "eaglei" / name,
                       files[name]["size"])
        n = filter_eaglei_csv(raw, dest)
        print(f"  {year}: {n:,} study-area rows -> {dest.relative_to(C.ROOT)}")
        # Preserve raw source files for reproducible checksums (EDA handoff policy).


# --------------------------------------------------------------------------- Storm Events
def latest_storm_files() -> dict[int, str]:
    """Map year -> newest StormEvents details filename on the NOAA server."""
    html = requests.get(C.STORM_EVENTS_URL, headers=HEADERS, timeout=60).text
    pat = re.compile(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz")
    best: dict[int, tuple[str, str]] = {}
    for m in pat.finditer(html):
        year, created = int(m.group(1)), m.group(2)
        if year not in best or created > best[year][0]:
            best[year] = (created, m.group(0))
    return {y: f for y, (_, f) in best.items()}


def tz_offset_hours(tz: str) -> float:
    if not isinstance(tz, str):
        return float("nan")
    m = re.search(r"([+-]?\d+)$", tz.strip())
    if m:
        return float(m.group(1))
    return float(TZ_FALLBACK.get(tz.strip().upper(), "nan"))


def clean_storm_events(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower() for c in df.columns]
    df["state_fips"] = df["state_fips"].astype("Int64").astype(str).str.zfill(2)
    df = df[df["state_fips"].isin(STATE_FIPS)].copy()

    def stamp(prefix: str) -> pd.Series:
        ym = df[f"{prefix}_yearmonth"].astype(str)
        day = df[f"{prefix}_day"].astype(int).astype(str).str.zfill(2)
        hhmm = df[f"{prefix}_time"].astype(int).astype(str).str.zfill(4)
        return pd.to_datetime(ym + day + hhmm, format="%Y%m%d%H%M", errors="coerce")

    off = pd.to_timedelta(df["cz_timezone"].map(tz_offset_hours), unit="h")
    df["begin_local"] = stamp("begin")
    df["end_local"] = stamp("end")
    df["begin_utc"] = df["begin_local"] - off      # local = UTC + offset
    df["end_utc"] = df["end_local"] - off
    df["cz_fips"] = df["cz_fips"].astype("Int64").astype(str).str.zfill(3)
    df["damage_property_usd"] = df["damage_property"].map(parse_damage)
    df["damage_crops_usd"] = df["damage_crops"].map(parse_damage)
    return df


def parse_damage(x) -> float:
    """NOAA writes damage like '25.00K', '1.5M', '0.00K'."""
    if not isinstance(x, str) or not x.strip():
        return float("nan")
    x = x.strip().upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(x[-1], 1)
    try:
        return float(x[:-1] if x[-1] in "KMB" else x) * mult
    except ValueError:
        return float("nan")


def get_storms(years: list[int]) -> None:
    print("\n[NOAA Storm Events]")
    files = latest_storm_files()
    out_dir = C.PROCESSED_DIR / "storms"
    for year in years:
        dest = out_dir / f"storm_events_{year}.parquet"
        if dest.exists():
            print(f"  {dest.name} done, skipping")
            continue
        if year not in files:
            print(f"  WARNING: no Storm Events file for {year}")
            continue
        raw = download(C.STORM_EVENTS_URL + files[year], C.RAW_DIR / "storms" / files[year])
        df = clean_storm_events(pd.read_csv(raw, compression="gzip", low_memory=False))
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        print(f"  {year}: {len(df):,} events ({files[year]})")


# --------------------------------------------------------------------------- zones -> counties
def get_zone_county() -> pd.DataFrame:
    """NWS zone-county correlation file. Many storms (hurricanes, high wind, winter
    storms) are reported by forecast ZONE, not county, so we need this to map them."""
    print("\n[NWS zone -> county lookup]")
    dest = C.PROCESSED_DIR / "reference" / "zone_county.parquet"
    if dest.exists():
        print("  zone_county.parquet done, skipping")
        return pd.read_parquet(dest)
    html = requests.get(C.ZONE_COUNTY_PAGE, headers=HEADERS, timeout=60).text
    links = re.findall(r"bp(\d{2})([a-z]{2})(\d{2})\.dbx", html)
    if not links:
        raise RuntimeError("Could not find a bp*.dbx link on " + C.ZONE_COUNTY_PAGE)
    months = {"ja": 1, "fe": 2, "mr": 3, "ap": 4, "my": 5, "jn": 6,
              "jl": 7, "au": 8, "se": 9, "oc": 10, "no": 11, "de": 12}
    dd, mm, yy = max(links, key=lambda t: (int(t[2]), months.get(t[1], 0), int(t[0])))
    url = f"https://www.weather.gov/source/gis/Shapefiles/County/bp{dd}{mm}{yy}.dbx"
    raw = download(url, C.RAW_DIR / "reference" / f"bp{dd}{mm}{yy}.dbx")
    cols = ["state", "zone", "cwa", "zone_name", "state_zone", "county",
            "fips_code", "time_zone", "fe_area", "lat", "lon"]
    zc = pd.read_csv(raw, sep="|", names=cols, dtype=str)
    zc["fips_code"] = zc["fips_code"].str.zfill(5)
    zc["zone"] = zc["zone"].str.zfill(3)
    zc = zc[zc["state"].isin(ABBR_TO_FIPS)].drop_duplicates(["state", "zone", "fips_code"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    zc.to_parquet(dest, index=False)
    print(f"  {len(zc):,} zone-county pairs from {raw.name}")
    return zc


def build_storm_counties(zc: pd.DataFrame) -> None:
    """One row per (storm event, affected county). County events map directly;
    zone events are expanded to every county the zone touches."""
    print("\n[Storm event -> county table]")
    parts = sorted((C.PROCESSED_DIR / "storms").glob("storm_events_*.parquet"))
    if not parts:
        print("  no storm files yet, skipping")
        return
    ev = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    ev["state_abbr"] = ev["state_fips"].map(C.STATES)

    county = ev[ev["cz_type"] == "C"].copy()
    county["fips_code"] = county["state_fips"] + county["cz_fips"]
    county["mapped_from"] = "county"

    lookup = zc[["state", "zone", "fips_code"]].rename(
        columns={"state": "zc_state", "zone": "zc_zone"})
    zone = ev[ev["cz_type"] == "Z"].merge(
        lookup, left_on=["state_abbr", "cz_fips"], right_on=["zc_state", "zc_zone"],
        how="left")
    unmatched = zone["fips_code"].isna().mean()
    zone = zone.dropna(subset=["fips_code"]).drop(columns=["zc_state", "zc_zone"])
    zone["mapped_from"] = "zone"

    out = pd.concat([county, zone], ignore_index=True)
    dest = C.PROCESSED_DIR / "storms" / "storm_event_counties.parquet"
    out.to_parquet(dest, index=False)
    print(f"  {len(out):,} event-county rows "
          f"({unmatched:.1%} of zone events had no county match; zones change over time)")


# --------------------------------------------------------------------------- census
def get_census(years: list[int]) -> None:
    print("\n[Census ACS + land area]")
    out_dir = C.PROCESSED_DIR / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "county_census.parquet"
    if dest.exists():
        print("  county_census.parquet done, skipping")
        return
    if not C.CENSUS_API_KEY:
        print("  WARNING: set CENSUS_API_KEY (free: https://api.census.gov/data/key_signup.html), skipping")
        return
    frames = []
    for year in years:
        for st in C.STATES:
            try:
                rows = get_json(C.CENSUS_ACS_URL.format(year=year),
                                get="NAME,B01003_001E,B25001_001E",
                                **{"for": "county:*", "in": f"state:{st}"},
                                key=C.CENSUS_API_KEY)
            except Exception as e:  # newest year may not be released yet
                print(f"  {year} state {st}: not available ({e.__class__.__name__})")
                break
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df["acs_year"] = year
            frames.append(df)
    if not frames:
        print("  WARNING: no ACS data returned (bad CENSUS_API_KEY?), skipping")
        return
    acs = pd.concat(frames, ignore_index=True).rename(columns={
        "NAME": "county_name", "B01003_001E": "population", "B25001_001E": "housing_units"})
    acs["fips_code"] = acs["state"] + acs["county"]
    acs[["population", "housing_units"]] = acs[["population", "housing_units"]].astype(int)

    z = requests.get(C.GAZETTEER_URL, headers=HEADERS, timeout=120).content
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        gaz = pd.read_csv(zf.open(zf.namelist()[0]), sep="\t", dtype={"GEOID": str})
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz = gaz[["GEOID", "ALAND_SQMI"]].rename(columns={"GEOID": "fips_code",
                                                       "ALAND_SQMI": "land_sqmi"})
    acs = acs.merge(gaz, on="fips_code", how="left")
    acs["pop_density"] = acs["population"] / acs["land_sqmi"]
    acs["housing_density"] = acs["housing_units"] / acs["land_sqmi"]
    acs.drop(columns=["state", "county"]).to_parquet(dest, index=False)
    print(f"  {len(acs):,} county-year rows")


# --------------------------------------------------------------------------- EIA-861
def get_eia(years: list[int]) -> None:
    print("\n[EIA-861 utility data, raw zips]")
    for year in years:
        try:
            download(C.EIA861_URL.format(year=year), C.RAW_DIR / "eia861" / f"f861{year}.zip")
        except requests.HTTPError:
            print(f"  {year}: not published yet")


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=C.YEARS)
    p.add_argument("--only", nargs="+",
                   choices=["eaglei", "storms", "zones", "census", "eia"],
                   help="run only these steps (default: all except eia)")
    p.add_argument("--keep-raw", action="store_true",
                   help="keep the ~1.2 GB/yr raw EAGLE-I CSVs after filtering")
    a = p.parse_args()
    steps = set(a.only or ["eaglei", "storms", "zones", "census"])

    if "storms" in steps:
        get_storms(a.years)
    if "zones" in steps or "storms" in steps:
        build_storm_counties(get_zone_county())
    if "census" in steps:
        get_census([y for y in a.years if y <= 2024])
    if "eia" in steps:
        get_eia(a.years)
    if "eaglei" in steps:   # last: the biggest download
        get_eaglei(a.years, a.keep_raw)
    print("\nDone. Run  python scripts/check_data.py  for a quick summary.")


if __name__ == "__main__":
    main()
