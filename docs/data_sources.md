# Data sources

Raw files live in `data/01_raw/` and are never modified. Checksums: `pipeline/raw_sources.json` (NOAA, EAGLE-I) and the `manifest.json` in each stage-08 folder. Redistribution terms must be confirmed for every source before any Hugging Face release (plan step 9).

## EDA release attribution review (2026-10-23)

The [Figshare API metadata for article 24237376](https://api.figshare.com/v2/articles/24237376) identifies version 4, DOI [10.6084/m9.figshare.24237376.v4](https://doi.org/10.6084/m9.figshare.24237376.v4), as **CC BY 4.0**. Attribute Christa Brelsford and coauthors, link the original and [license](https://creativecommons.org/licenses/by/4.0/), and identify our changes: filtering, geographic matching, time-window aggregation, labels and derived predictors. Preserve the local archive checksum as the actual build provenance; publication metadata is not a substitute for byte verification.

NOAA/NWS and Census products are credited to their agencies. IEM requests attribution and explains its data-use terms in its [disclaimer](https://mesonet.agron.iastate.edu/disclaimer.php); credit the Iowa Environmental Mesonet for the archived NWS alerts. The prepared private release includes an attribution README and source links. It contains only approved prepared inputs, not a redistribution of the raw archives. A broader/public release should review any additional sources separately.

| Source | Files | Access | Version tracking | Use | Redistribution notes |
|---|---|---|---|---|---|
| NOAA NCEI Storm Events, 2014-2025 | `StormEvents_{details,fatalities,locations}-ftp_v1.0_dYYYY_cYYYYMMDD.csv.gz` | [NCEI CSV files](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/) | The c-date changes when NCEI revises a year; old versions are removed | Storm onsets, types and areas | U.S. government work; cite NCEI |
| EAGLE-I recorded outages, 2014-2025 | `24237376.zip` | [figshare 24237376](https://figshare.com/articles/dataset/24237376); [data descriptor](https://doi.org/10.1038/s41597-024-03095-5) | SHA-256 in `raw_sources.json`; release metadata identifies v4 (see review above) | Customers out by county every 15 minutes; MCC, DQI and coverage history | CC BY 4.0; attribution, license link and transformations included in the prepared release |
| NWS zone-county correlation files | `bpDDmmYY.dbx`, 23 versions | [weather.gov](https://www.weather.gov/gis/ZoneCounty) (April 2019 onward); pinned Internet Archive captures for earlier versions | Effective date in the file name; captures pinned by timestamp | Historical zone-to-county mapping | U.S. government work |
| IEM NWS alert (VTEC) archive | `vtec_southeast_YYYY.csv` | [IEM watch/warning service](https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml) | Retrieval time and SHA-256; IEM can revise recent years | Alert issuance times and areas (before-arrival design) | Derived from NWS products; acknowledge the Iowa Environmental Mesonet |
| Census county population estimates, Vintage 2023 | `co-est2023-alldata.csv` | [Census Bureau](https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/) | SHA-256 | County list, names, size, denominator checks | U.S. government work |
| Census Connecticut crosswalk and town populations | `data/reference/census_ct/` | Stage 03 | — | Connecticut planning regions (national outputs only) | U.S. government work |
| EIA-861 state customers (via EAGLE-I `coverage_history.csv`) | in the EAGLE-I archive | — | — | State totals for denominator validation | As for EAGLE-I |

## Reading notes

- EAGLE-I timestamps are UTC run starts at :00, :15, :30 and :45. Zero-outage rows are omitted in 2014-2021 and intermittent from 2022; an absent row means zero or a collection gap.
- NOAA local times are converted with the recorded offset (for example `EST-5`); daylight-labelled offsets are flagged.
- NOAA zone codes do not always follow NWS effective dates. Mobile's 2015 zone change appears in Storm Events only from late 2017, so zone matching checks names as well as codes.
