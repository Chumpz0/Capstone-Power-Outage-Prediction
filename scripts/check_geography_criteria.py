"""Audit existing storm geography; write evidence overlay without replacing sources."""
from pathlib import Path
import hashlib
import json
import re
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C
from src.forecast_criteria import split_origins


def normalize(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper().split(" COUNTY")[0])


def supported_candidate(row, lookup, county_names):
    """Require exact zone name, unique county, and matching Census county name."""
    candidates = lookup[lookup.state.eq(row.state_abbr) & lookup.zone.eq(row.cz_fips)]
    if len(candidates) != 1:
        return None
    candidate = candidates.iloc[0]
    if normalize(row.cz_name) != normalize(candidate.zone_name):
        return None
    if normalize(candidate.county) != county_names.get(candidate.fips_code):
        return None
    return candidate.fips_code


def main():
    p = C.PROCESSED_DIR
    paths = [p / f"storms/storm_events_{y}.parquet" for y in C.YEARS]
    columns = ["event_id", "year", "state_fips", "cz_type", "cz_fips", "cz_name", "event_type"]
    events = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    events["state_abbr"] = events.state_fips.map(C.STATES)
    mapping_path = p / "storms/storm_event_counties.parquet"
    mapped = pd.read_parquet(mapping_path, columns=["event_id", "fips_code", "mapped_from"])
    census_path, zone_path = p / "reference/county_census.parquet", p / "reference/zone_county.parquet"
    census = pd.read_parquet(census_path).drop_duplicates("fips_code")
    zone = pd.read_parquet(zone_path)
    names = census.set_index("fips_code").county_name.map(lambda x: normalize(x.split(",")[0])).to_dict()
    assert events.event_id.is_unique and not mapped.duplicated(["event_id", "fips_code"]).any()
    full = mapped.merge(events, on="event_id", validate="many_to_one")
    direct = full[full.cz_type.eq("C")].copy()
    conflicts = direct[~direct.cz_name.map(normalize).eq(direct.fips_code.map(names))].copy()
    corrections = []
    for row in conflicts.itertuples():
        candidate = supported_candidate(row, zone, names)
        corrections.append(dict(event_id=int(row.event_id), year=int(row.year), state=row.state_abbr,
            event_type=row.event_type, original_cz_type=row.cz_type, original_cz_fips=row.cz_fips,
            original_cz_name=row.cz_name, original_fips=row.fips_code,
            original_fips_valid=row.fips_code in names, corrected_fips=candidate,
            disposition="supported_local_overlay" if candidate else "quarantine_unresolved",
            evidence="Exact state/zone code and zone-name match; unique lookup county with matching Census name" if candidate else "Insufficient unambiguous local evidence"))
    unmatched = events[~events.event_id.isin(mapped.event_id)].copy()
    unmatched["disposition"] = "unresolved_historical_zone; exclude from county storm attribution; retain county-hour targets"
    assert unmatched.cz_type.eq("Z").all()
    # An overlay is explicit and reversible; the original data stays intact.
    overlay = pd.DataFrame(corrections)
    out = p / "eda/geography_criteria"
    out.mkdir(parents=True, exist_ok=True)
    overlay.to_parquet(out / "mapping_overlay.parquet", index=False)
    unmatched.to_parquet(out / "unmatched_events.parquet", index=False)
    audited = mapped.merge(overlay[["event_id", "original_fips", "corrected_fips", "disposition"]],
                           left_on=["event_id", "fips_code"], right_on=["event_id", "original_fips"], how="left", validate="one_to_one")
    audited["original_mapped_from"] = audited.mapped_from
    audited.loc[audited.corrected_fips.notna(), "mapped_from"] = "zone_evidence_override"
    audited["original_fips"] = audited.fips_code
    audited.loc[audited.corrected_fips.notna(), "fips_code"] = audited.loc[audited.corrected_fips.notna(), "corrected_fips"]
    audited = audited[audited.disposition.ne("quarantine_unresolved")]
    assert audited.fips_code.isin(names).all()
    assert not audited.duplicated(["event_id", "fips_code"]).any()
    audited.to_parquet(out / "audited_event_counties.parquet", index=False)
    original = pd.read_parquet(mapping_path)
    corrected = original.rename(columns={"fips_code": "original_fips", "mapped_from": "original_mapped_from"}).merge(
        audited[["event_id", "original_fips", "fips_code", "mapped_from", "disposition"]],
        on=["event_id", "original_fips"], how="inner", validate="one_to_one")
    corrected.to_parquet(out / "storm_event_counties.parquet", index=False)
    groups = ["year", "state_abbr", "event_type"]
    losses = events.groupby(groups).size().rename("reports").to_frame().join(unmatched.groupby(groups).size().rename("unmatched"), how="left").fillna(0).astype(int).reset_index()
    zone_losses = unmatched.groupby(["state_abbr", "cz_fips", "cz_name"]).size().rename("unmatched").reset_index().sort_values("unmatched", ascending=False)
    yearly = []
    for year, group in events.groupby("year"):
        u = unmatched[unmatched.year.eq(year)]
        yearly.append(dict(year=int(year), reports=len(group), zone_reports=int(group.cz_type.eq("Z").sum()),
                           unmatched=len(u), name_conflicts=int(conflicts.year.eq(year).sum())))
    boundary = []
    for year in [2023, 2024, 2025]:
        times = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", inclusive="left", freq="h")
        for h in [6, 12]:
            assigned = split_origins(times, h)
            boundary.append(dict(year=year, horizon=h, excluded_origins_per_county=int(assigned.eq("excluded").sum()),
                                 excluded_county_origins=int(assigned.eq("excluded").sum()) * 534))
    source_paths = paths + [mapping_path, census_path, zone_path]
    result = dict(checkpoint=3, scope="2014-2025 geography integrity only; no holdout outage outcomes read",
        historical_lookup_limitation="One local lookup vintage; matches do not certify historical zone boundaries",
        totals=dict(reports=len(events), zone_reports=int(events.cz_type.eq("Z").sum()), unmatched=len(unmatched),
                    name_conflicts=len(conflicts), invalid_original_ids=int((~overlay.original_fips_valid).sum()),
                    supported_corrections=int(overlay.corrected_fips.notna().sum()), audited_rows=len(audited)),
        by_year=yearly, exclusions_by_year_state_type=losses.to_dict("records"), unmatched_zones=zone_losses.to_dict("records"),
        mapping_dispositions=corrections, split_boundary_exclusions=boundary,
        files=[dict(path=x.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in source_paths])
    result["code_files"] = [dict(path=name, sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest()) for name in ["scripts/check_geography_criteria.py", "src/forecast_criteria.py"]]
    (ROOT / "docs/checkpoint_3_geography_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["totals"], indent=2))
    print(pd.DataFrame(yearly).to_string(index=False))


if __name__ == "__main__":
    main()
