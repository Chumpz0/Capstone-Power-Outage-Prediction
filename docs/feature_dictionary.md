# Feature dictionary

Predictors for the major-outage task, built by `pipeline/13_build_major_outage_dataset.py`. Every feature uses only information with a timestamp at or before the prediction time `t0`. For alert rows, `t0` is the issuance of the first storm-family NWS watch or warning for the county; for onset rows it is the NOAA-recorded begin time. Logs are `log10(1 + x)`. Missing values stay missing (`NULL`); the teammate must define missing-value handling in the baseline protocol.

## Common to both designs

| Feature | Source | Timestamp rule | Transformation | Why |
|---|---|---|---|---|
| `month`, `hour_utc`, `day_of_week` | `t0` | known at `t0` | integers | Seasonality and diurnal storm timing |
| `state` | Census county | static | one-hot (categorical) | Utility and reporting differences between states |
| `log_population` | Census 2020 base population | static | log10 | County size; 2014-2019 rows use the 2020 base (slowly changing attribute) |
| `log_mcc_valid` | EAGLE-I modeled county customers (2022) | static | log10; missing unless `mcc_status = 'valid'` | Customer base where the denominator validated (never North Carolina) |
| `log_at_t0_value`, `at_t0_row_present` | EAGLE-I | last run starting at or before `t0 - 15 min` | log; presence flag | Outages already underway; an absent row is not zero |
| `log_last_hour_max` | EAGLE-I | runs in `[t0 - 1 h, t0 - 15 min]` | log | Recent level when the last run is absent |
| `log_pre_max_24h`, `pre_observed`, `pre_null` | EAGLE-I | runs in `[t0 - 24 h, t0 - 15 min]` | log; counts | Recent conditions and reporting coverage |
| `hours_since_last_observation` | EAGLE-I | as above | hours | Staleness of the latest known value |
| `hist_days_observed`, `hist_days_major`, `log_hist_max_daily`, `hist30_days_ge_1000`, `hist_span_days` | EAGLE-I daily maxima | days `[date(t0) - 365, date(t0) - 1]` (30 days for `hist30_*`) | counts; log | County outage climatology from earlier days only; `hist_span_days` marks short early history |
| `act_<family>_<W/A/Y>` | IEM VTEC county alert events | issued at or before `t0`, within the previous 72 h, still valid at `t0` | counts by family and significance (warning, watch, advisory) | Hazards in effect for the county |
| `alerts_prior_72h` | VTEC | issued in `[t0 - 72 h, t0)` | count | Recent alert activity |
| `hours_since_first_active_alert` | VTEC | as for `act_*` | hours | Lead of the earliest active watch or warning |
| `max_hours_to_valid_start` | VTEC | as for `act_*` | hours (0 if already valid) | How far ahead the hazard is expected to start |
| `region_warned_counties`, `state_warned_counties` | VTEC | warnings issued by `t0` and valid at `t0` | counts of counties | Spatial extent of the event (a proxy for system scale) |
| `region_tropical_alert_counties` | VTEC | tropical watches and warnings valid at `t0` | count | Tropical system extent |

## Alert rows only

| Feature | Source | Rule | Why |
|---|---|---|---|
| `trigger_has_warning` | VTEC | alerts issued exactly at `t0` | Warning versus watch-only trigger |
| `trig_fam_<family>` | VTEC | as above | Hazard family of the trigger |
| `trig_<code>` (for example `trig_HU_W`, `trig_SV_W`) | VTEC | as above | Specific products: tornado, severe thunderstorm, flood, hurricane, tropical storm, storm surge, high wind, winter, ice, coastal |
| `trigger_hours_to_valid_start` | VTEC | issuance to validity start | Lead of the trigger |
| `trigger_validity_hours` | VTEC | initial validity, capped at 72 h | Expected duration as issued |

## Onset rows only

| Feature | Source | Rule | Why |
|---|---|---|---|
| `noaa_<group>` | NOAA Storm Events | event types whose recorded begin equals `t0` | Retrospective proxy for what is known at onset; real-time availability is assumed, not established |

## Never used as predictors

| Field | Reason |
|---|---|
| Outcome columns: `label`, `y_major`, `peak_customers`, `n_observed`, `max_gap_slots`, `single_run_crossing`, `bracketed_gap_slots`, `window_peak_exceeds_customers`, `covered_window`, `peak_complete` | Describe the outcome window (72 h for alert rows, 24 h for onset rows) |
| `retro_*` (later NOAA records in the window, realized duration, magnitude) | Realized after `t0` |
| IEM `utc_expire`, `status`, `utc_updated` | Final alert state after later updates |
| `outage_plausibility` | Computed from every year's peaks, including future ones; subgroup reporting only |
| `size_bin`, `primary_family`, `trigger_partial_county` | Analysis groupings (their information is already present as features or is retrospective) |
| Damage, injuries, deaths, narratives | Post-event outcomes |
