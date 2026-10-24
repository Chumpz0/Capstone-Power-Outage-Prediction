# Data

## Approved baseline inputs

Use `data/04_model_input/southeast_v1/`, obtained with `scripts/download_eda_data.py` or built by pipeline stages 00–14. See [the dataset card](../docs/dataset_card.md) and [handoff](../docs/teammate_handoff.md). Those inputs use fixed alert windows, historical zone mappings, count-based labels and coverage-aware negatives. Test files contain features only.

The layout and loaders below are **legacy acquisition data**, not the approved labeled population. MCC is invalid for North Carolina and customer totals require validation before percentages. Current NWS zones do not establish historical county mappings. Legacy EAGLE-I downloads now preserve raw files; no source is automatically deleted. Do not run target summaries on 2025.

Everything in this folder except this file is git-ignored. The shared copy lives in the team's private Hugging Face dataset; load it with the functions in `src/data.py`. To build it yourself, run `python scripts/download_data.py` (see the main README). The same paths below are used locally and on Hugging Face.

```
data/
  raw/          preserved legacy downloads
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

`MCC.csv` holds modeled customer totals. Percentage analyses are supporting only and require validated denominators (`mcc_status = 'valid'`) and plausible peaks. Do not use MCC for North Carolina or assume all totals are valid. Primary labels use customer counts.

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
