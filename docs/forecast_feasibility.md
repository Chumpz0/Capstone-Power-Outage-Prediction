# Can we predict before storm arrival? Archived forecast and warning feasibility

Step 2 of the project plan. Generated tables are in `reports/forecast_feasibility/` (stage 12); all statistics use training years only (2014-2023 issuance or onset).

## Bottom line

- **Feasible for hazards with real lead time.** Archived NWS watches, warnings and advisories (VTEC) exist for every county and year, with product issuance times, and they map to counties for 99.98% of zone records. Before NOAA-recorded tropical onsets the earliest alert came a median 41.5 hours ahead; winter, synoptic-wind and coastal onsets about 25-27 hours ahead.
- **Convective storms give little lead.** Thunderstorm-wind, hail and lightning onsets had a median earliest alert of 0.7-1.2 hours, mostly watches. For them a "before-arrival" design is close to an onset design.
- **The archive includes storms that were never recorded as events**, which the NOAA-onset design cannot see. Only 47% of county severe-thunderstorm warnings and 12% of wind advisories were followed by any recorded storm event in that county.
- **NOAA-onset rows miss many major outages.** Of alert-triggered 24-hour windows that reached at least 5,000 customers out, 28% had no NOAA storm record in the county during the window (46% at 1,000; 23-25% at 10,000-25,000).
- **Label availability is the binding constraint for both designs**, not weather inputs (see [label_feasibility_report.md](label_feasibility_report.md)).

Recommendation for checkpoint A: use **alert-triggered rows** (prediction time = first storm-family NWS watch or warning issued for the county) as the primary design, and keep the NOAA-onset rows as a clearly labeled retrospective comparison. The final choice depends on the audience (decision O-1).

## Sources investigated

| Source | Coverage and access | Timestamps | Geography | Includes non-events | Processing | Verdict |
|---|---|---|---|---|---|---|
| NWS VTEC watches, warnings, advisories (Iowa Environmental Mesonet archive) | County data 1986+, VTEC products 2005+; free CSV/shapefile API, about 11-14 MB per year for the six states | Product issuance (`utc_prodissue`), validity start per county or zone, initial expiry | County (`FLC001`) or forecast-zone (`FLZ123`) codes; storm-based polygons duplicated by county rows | Yes | Light: downloaded and processed for all 12 years in about 20 s | **Tested end to end** (stages 08, 12) |
| SPC convective outlooks, Days 1-8 (IEM archive) | 1987+, 99.9% complete per IEM; shapefile API | Product issuance (`PRODISS`) and valid period | Polygons; need county overlay | Yes | Moderate: spatial join to counties | Strong candidate for convective lead time; not yet tested |
| WPC excessive rainfall outlooks (IEM archive) | November 2015+ | Issuance and valid period | Polygons | Yes | Moderate | Candidate for flood hazards; partial training coverage |
| NHC GIS products | Advisory cone and watches/warnings 2008+; wind-speed probabilities 2009+; arrival time of tropical-storm winds 2018+ | Per advisory | Polygons and grids | Yes (forecast tracks) | Moderate to high | Best source for a fixed lead time before tropical arrival; arrival times cover only 2018+ |
| NDFD gridded forecasts (NCEI) | About ten years online; AWS from April 2020 | Issuance and forecast hour | 2.5 km grids (GRIB2) | Yes | High (large GRIB2 volume) | Not realistic for this project's timeline |

Not investigated: numerical-model archives such as HRRR.

## What the VTEC test shows

**Mapping.** 738,829 alert rows (2014-2025). County codes map directly; 354,314 zone rows map through the versioned correlation files (stage 09), 25,240 of them through the nearest version rather than the effective one, and 112 do not map (marine and other non-county zones).

**Leakage traps.** IEM's `utc_expire`, `status` and `utc_updated` describe the event's final state after later updates, so they are unknown at issuance and are never read. Only issuance time, validity start and initial expiry are used.

**Lead time before NOAA-recorded onsets** (training opportunities from stage 10; earliest storm-family watch, warning or advisory issued within 72 hours before onset and still in effect):

| Hazard at onset | Opportunities | Any prior alert | Prior warning | Median earliest lead (h) | Alert at least 6 h ahead | At least 24 h ahead |
|---|---|---|---|---|---|---|
| All | 33,784 | 83% | 68% | 3.1 | 34% | 20% |
| Tropical | 1,454 | 98% | 89% | 41.5 | 95% | 83% |
| Winter | 4,882 | 87% | 46% | 25.5 | 81% | 49% |
| Synoptic wind | 1,225 | 86% | 36% | 25.7 | 79% | 47% |
| Coastal surge | 439 | 82% | 46% | 27.0 | 75% | 44% |
| Rain and flood | 3,780 | 81% | 60% | 8.3 | 43% | 28% |
| Convective wind | 18,354 | 83% | 78% | 1.2 | 16% | 7% |
| Hail | 3,019 | 82% | 76% | 0.7 | 9% | 2% |
| Lightning | 631 | 37% | 21% | 1.1 | 6% | 3% |

**Alerts that were not followed by a recorded storm** (county-alert pairs; follow-up = any storm-scope NOAA event beginning in the county between 1 hour before issuance and 12 hours after the initial expiry, at most 84 hours after issuance):

| Alert | County alerts | Followed by a NOAA event |
|---|---|---|
| Severe thunderstorm warning (SV.W) | 87,947 | 47% |
| Tornado warning (TO.W) | 10,991 | 59% |
| Flash flood warning (FF.W) | 11,542 | 46% |
| Wind advisory (WI.Y) | 31,128 | 12% |
| Winter storm warning (WS.W) | 3,776 | 73% |
| Tropical storm warning (TR.W) | 2,229 | 66% |
| Hurricane watch (HU.A) | 477 | 86% |
| High wind warning (HW.W) | 1,538 | 41% |

**Alert-triggered windows** (training; greedy 24-hour county windows starting at the first storm-family watch or warning; 95,432 windows):

| Threshold | Majors with a NOAA event in the window | Majors without one | Share missed by NOAA-onset rows |
|---|---|---|---|
| 1,000 | 8,060 | 6,854 | 46% |
| 5,000 | 2,054 | 808 | 28% |
| 10,000 | 844 | 280 | 25% |
| 25,000 | 261 | 80 | 23% |

These shares are upper bounds on what NOAA misses. A watch issued two days ahead can start a window that ends before the storm arrives, and some majors are pre-existing or non-weather outages. Labels are also scarcer in alert windows without a recorded storm: at 5,000 customers only 5% of them have a known label, versus 16% of windows with a NOAA event.

## Design comparison

| | Before-arrival (alert-triggered) | Onset (NOAA-recorded) |
|---|---|---|
| Supports | Preparation ahead of expected impacts: staging crews, mutual aid, public messaging | Early response once a storm has begun |
| Prediction time | First storm-family watch or warning issued for the county | NOAA event begin time |
| Row exists when | An alert was issued, whether or not a storm materialized | Storm Events recorded an event, which is compiled after the fact and often from damage reports |
| Training opportunities (2014-2023) | 105,780 | 33,784 |
| Main risk | Lead time varies by hazard (0.7 to 41 hours); many windows carry no recorded storm | Selection on the outcome: rows exist mainly where impacts were reported; real-time availability not established |
| Operational validity | Inputs genuinely available at prediction time | Retrospective prototype only |

## Information allowed at the prediction time

**Alert-triggered design.** Alert type, significance and validity start of every alert issued for the county (and its neighbors) at or before `t0`; for tropical rows, NHC products issued by `t0` if added; SPC outlooks issued by `t0` if added; calendar fields of `t0`; county context (Census population, validated customer denominators, state); EAGLE-I runs that started at or before `t0 - 15 min`; outage history computed from windows that ended before `t0`.

**Onset design.** All of the above at the onset time, plus the NOAA event types whose recorded begin time equals `t0`, as a retrospective proxy for what is known at onset.

**Never allowed in either design.** Realized duration, magnitude, damage, deaths and injuries, narratives, later NOAA records in the window, final alert status or expiry, and any outcome-window summary.

## Questions for checkpoint A

1. Who uses the prediction, and for what action?
2. Which prediction time defines a row: first alert issuance, recorded onset, or both with alerts as the primary design?
3. Should heat and cold hazards be in scope? Cold-driven load shedding would otherwise be missed.

## Limitations

- Forecast products other than VTEC alerts were assessed from documentation, not downloaded.
- The zone-county correlation archive has three unrecovered releases affecting Mobile (2015), Wilmington (December 2017) and Jacksonville (May 2018). Mapping in those periods relies on name-verified fallback.
- Follow-up rates depend on NOAA's recording practice, which differs between WFOs and years.
- IEM regenerates recent years daily; retrieval dates and checksums are recorded in `data/01_raw/iem_vtec/manifest.json`.

Sources: [IEM watch/warning archive](https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml), [IEM outlook archive](https://mesonet.agron.iastate.edu/request/gis/outlooks.phtml), [NHC GIS](https://www.nhc.noaa.gov/gis/), [NDFD at NCEI](https://www.ncei.noaa.gov/products/weather-climate-models/national-digital-forecast-database), [NWS zone-county correlation](https://www.weather.gov/gis/ZoneCounty).
