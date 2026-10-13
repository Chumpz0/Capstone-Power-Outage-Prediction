# Data

Everything in this folder except this file is git-ignored. The shared copy lives in the team's private Hugging Face dataset; load it with the functions in `src/data.py`. To build it yourself, run `python scripts/download_data.py` (see the main README). The same paths below are used locally and on Hugging Face.

```
data/
  raw/          temporary downloads (EAGLE-I raw CSVs are deleted after filtering)
  processed/
    eaglei/     outages_2014.parquet ... outages_2025.parquet, MCC.csv, coverage_history.csv, DQI.csv
    storms/     storm_events_YYYY.parquet, storm_event_counties.parquet
    reference/  zone_county.parquet, county_census.parquet
```

## Key columns

**EAGLE-I outages** (`eaglei/outages_YYYY.parquet`), one row per county per 15-minute snapshot:

| Column | Meaning |
| --- | --- |
| `fips_code` | 5-digit county FIPS code (string, zero-padded). **Join key.** |
| `county`, `state` | Names |
| `customers_out` | Customers without power at that snapshot |
| `run_start_time` | Snapshot timestamp |

`MCC.csv` holds the modeled total customers per county. Use it to turn `customers_out` into a **% of customers out**, since EAGLE-I coverage grows over the years.

**Storm event counties** (`storms/storm_event_counties.parquet`), one row per storm event per affected county. It has all NOAA columns (lower-case) plus:

| Column | Meaning |
| --- | --- |
| `fips_code` | County the event affected. **Join key.** |
| `event_id`, `episode_id` | NOAA ids (an episode groups related events, e.g. one hurricane) |
| `event_type` | e.g. Thunderstorm Wind, Tropical Storm, Hurricane, Winter Storm, Ice Storm |
| `begin_local` / `end_local` | Event start/end in local standard time |
| `begin_utc` / `end_utc` | Same, converted to UTC using `cz_timezone` |
| `magnitude` | Wind speed (knots) for wind events, hail size for hail |
| `damage_property_usd` | Property damage in dollars (parsed from "25.00K" style text) |
| `mapped_from` | `county` (reported by county) or `zone` (expanded from an NWS forecast zone) |

**County census** (`reference/county_census.parquet`): `fips_code`, `acs_year`, `population`, `housing_units`, `land_sqmi`, `pop_density`, `housing_density`.

## Known issues to check during EDA

1. **Time zones:** confirm whether EAGLE-I `run_start_time` is UTC or local before joining it to storm times. Storms come with both versions.
2. **Zone mapping:** hurricanes, high wind and winter storms are mostly reported by forecast zone, not county. We map zones with the current NWS lookup, but zones were redrawn over the years, so some older events may not match. `download_data.py` prints the unmatched share.
3. **Coverage drift:** EAGLE-I coverage is thinner in early years (see `coverage_history.csv`). Consider starting training in 2017 or 2018 if 2014–2016 look sparse.
4. **Overlapping events:** one hurricane creates many NOAA events across counties. Group by `episode_id` when building one-row-per-storm features.
5. **Gaps and zeros:** EAGLE-I rows are snapshots from utility outage maps. Missing snapshots are not the same as zero outages.
