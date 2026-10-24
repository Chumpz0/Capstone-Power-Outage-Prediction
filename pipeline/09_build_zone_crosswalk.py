"""Build a versioned NWS zone-to-county crosswalk and map Southeast NOAA events to counties.

Run: python pipeline/09_build_zone_crosswalk.py   (after stages 04 and 08)
Outputs:
  data/reference/nws_zone_county_versions.parquet        all versions, one row per version/zone/county
  data/03_cleaned/southeast/zone_event_counties.parquet  one row per zone-mapped Southeast event and county
  data/03_cleaned/southeast/county_event_checks.parquet  name verification of county-coded events
  reports/zone_crosswalk/   version inventory, zone changes, mapping status, unmapped events

Zone events (CZ_TYPE=Z) are mapped with the correlation version in effect on their UTC
begin date. Zone numbers are reused and renamed, and NOAA did not always switch codes
when NWS did (Mobile zones changed in 2015; Storm Events used the old codes until 2017),
so the NOAA zone name is checked against the version's zone name and county names:
  verified  zone-name similarity >= 0.85, or a mapped county's name appears in the NOAA name
  weak      zone-name similarity >= 0.6
  conflict  otherwise
County names come from the Census county list plus every label the correlation files use
for that FIPS (older files put labels such as "Mainland Dare" in the county field while
the FIPS stays correct). NOAA's own names are never used to verify NOAA records.
If the zone is absent from the effective version, or its name conflicts, the nearest
other version with a verified name is used. Archive gaps (stage 08 MISSING_VERSIONS)
make that fallback necessary for some Mobile, Wilmington and Jacksonville zones. A zone
code found in no version is mapped only when exactly one zone in the state has a
verified name (NOAA codes North Walton FL as FL107; NWS uses FL007). Conflicts and
remaining unmatched zones are not mapped.

County-coded events (CZ_TYPE=C) must match the Census/NWS name of their FIPS. Some are
zone records miscoded as counties (Georgia's January 2025 winter storm uses zone numbers
such as 13044 "SOUTH FULTON"; 13037 "OCONEE" would otherwise land in Calhoun County).
A failing county-coded event is reinterpreted as a zone only when the zone name verifies;
otherwise it is left unmapped.

A zone maps to every county it touches, and a county can be split among several zones.
Mapped rows carry zones_in_county so partial-county coverage stays visible: an event in
one of three zones of a county does not mean the whole county was affected.
"""
import argparse
import json
import re
from difflib import SequenceMatcher
import pandas as pd

from common import SE_ABBR_BY_FIPS, SE_STATE_FIPS, SOUTHEAST_STATES, connect, copy_query, lit, sql_list, write_json
from paths import CENSUS_COUNTY_POP, CLEANED, RAW_ZONE_COUNTY, SOUTHEAST, ZONE_COUNTY, ZONE_REPORTS

FIELDS = ['state', 'zone', 'cwa', 'zone_name', 'state_zone', 'county_name', 'county_fips', 'time_zone', 'fe_area', 'lat', 'lon']
VERIFIED, WEAK = 0.85, 0.6
ACCEPTED = ('effective_version_verified', 'effective_version_weak', 'other_version_verified', 'zone_code_unmatched_name_verified')


def norm(name):
    name = (name or '').upper().replace('&', ' AND ')
    name = re.sub(r'[^A-Z0-9]+', ' ', name)
    return re.sub(r'\b(COUNTY|COUNTIES|PARISH)\b', ' ', name).split()


def names_match(record_name, known_names):
    """True if a known county name appears in the record name (DE KALB = DeKalb)."""
    tokens = norm(record_name)
    joined = ''.join(tokens)
    for known in known_names:
        k = norm(known)
        if k and tokens and (set(k) <= set(tokens) or ''.join(k) in joined):
            return True
    return False


def name_check(noaa_name, zone_name, county_names):
    """Return (similarity, verdict) for a NOAA zone name against an NWS zone. Verdict: verified, weak, conflict."""
    a, b = norm(noaa_name), norm(zone_name)
    similarity = SequenceMatcher(None, ' '.join(a), ' '.join(b)).ratio()
    if a and b and (set(a) <= set(b) or set(b) <= set(a)):
        similarity = max(similarity, 0.9)
    verdict = ('verified' if similarity >= VERIFIED or names_match(noaa_name, county_names)
               else 'weak' if similarity >= WEAK else 'conflict')
    return round(similarity, 3), verdict


def read_versions():
    manifest = json.loads((RAW_ZONE_COUNTY / 'manifest.json').read_text())
    frames = []
    se_abbr = set(SE_ABBR_BY_FIPS.values())
    for name, info in sorted(manifest['files'].items(), key=lambda kv: kv[1]['effective_date']):
        lines = (RAW_ZONE_COUNTY / name).read_text(encoding='latin-1').splitlines()
        rows = [[v.strip() for v in line.split('|')] for line in lines if line.strip()]
        valid = [r for r in rows if len(r) == len(FIELDS) and re.fullmatch(r'\d{5}', r[6])]
        malformed = [r for r in rows if r not in valid]
        # A few archived lines elsewhere lack or shift a field (VA515, NM402, CA552); none may be in the study area.
        assert not [r for r in malformed if r[0] in se_abbr], f'{name}: malformed Southeast lines {malformed[:3]}'
        frame = pd.DataFrame(valid, columns=FIELDS)
        # STATE_ZONE is the identifier NOAA and VTEC use; a few rows disagree with the separate
        # STATE/ZONE fields (e.g. FL108 tagged GA in 2014-2015 files), so derive both from it.
        derived = frame.state_zone.str[:2], frame.state_zone.str[2:]
        frame['fields_disagree'] = (frame.state != derived[0]) | (frame.zone != derived[1])
        frame['state'], frame['zone'] = derived
        frame['version_tag'] = name.removesuffix('.dbx')
        frame['effective_date'] = info['effective_date']
        frame['source'] = info['source']
        frame['raw_lines'] = len(rows)
        frame['malformed_lines'] = len(malformed)
        frames.append(frame)
    return manifest, pd.concat(frames, ignore_index=True)


class ZoneMatcher:
    """Find the correlation version and counties for a NOAA zone code, name and date."""

    def __init__(self, con, se_abbr, labels, gaps):
        se = con.execute(f"""SELECT version_tag, effective_date, state_zone, any_value(zone_name) AS zone_name,
                list(county_fips ORDER BY county_fips) AS counties
            FROM versions WHERE state IN {sql_list(se_abbr)} GROUP BY ALL""").df()
        se['county_names'] = [sorted(set().union(*(labels.get(f, set()) for f in fips))) for fips in se.counties]
        tags = con.execute('SELECT DISTINCT version_tag, effective_date FROM versions ORDER BY effective_date').fetchall()
        self.tag_dates = pd.Series({t: pd.Timestamp(d) for t, d in tags})
        self.by_zone = {z: g.set_index('version_tag') for z, g in se.groupby('state_zone')}
        self.by_state = {s: g for s, g in se.groupby(se.state_zone.str[:2])}
        self.gaps = gaps

    def effective_tag(self, day):
        prior = self.tag_dates[self.tag_dates <= day]
        return prior.index[-1] if len(prior) else self.tag_dates.index[0]

    def in_gap(self, wfo, day):
        for info in self.gaps.values():
            start = pd.Timestamp(info['effective_date'])
            if wfo in info['wfos'] and start <= day < self.tag_dates[self.tag_dates > start].min():
                return True
        return False

    def match(self, state_zone, cz_name, wfo, day):
        tag = self.effective_tag(day)
        out = {'effective_tag': tag, 'mapped_tag': None, 'mapped_zone': None, 'match_status': 'unmatched',
               'name_similarity': None, 'in_missing_version_gap': self.in_gap(wfo, day)}
        options = self.by_zone.get(state_zone)
        if options is None:
            # Zone code in no version: accept only a unique, name-verified zone in the same state.
            pool = self.by_state.get(state_zone[:2])
            if pool is None:
                return out
            hits = pool[[name_check(cz_name, r.zone_name, [])[0] >= VERIFIED for r in pool.itertuples()]]
            if len(hits) and hits.state_zone.nunique() == 1:
                hits = hits.assign(distance=(hits.effective_date.map(pd.Timestamp) - day).abs()).sort_values('distance')
                out.update(mapped_tag=hits.version_tag.iloc[0], mapped_zone=hits.state_zone.iloc[0],
                           match_status='zone_code_unmatched_name_verified',
                           name_similarity=name_check(cz_name, hits.zone_name.iloc[0], [])[0])
            return out
        checks = {t: name_check(cz_name, r.zone_name, r.county_names) for t, r in options.iterrows()}
        if tag in checks and checks[tag][1] != 'conflict':
            chosen, status = tag, 'effective_version_' + checks[tag][1]
        else:
            ranked = sorted(checks, key=lambda t: abs((self.tag_dates[t] - day).days))
            verified = [t for t in ranked if checks[t][1] == 'verified']
            chosen = verified[0] if verified else None
            status = ('other_version_verified' if verified else
                      'effective_version_conflict' if tag in checks else 'other_version_conflict_or_weak')
        out['match_status'] = status
        if chosen is not None:
            out.update(mapped_tag=chosen, mapped_zone=state_zone, name_similarity=checks[chosen][0])
        return out


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    ZONE_REPORTS.mkdir(parents=True, exist_ok=True)
    SOUTHEAST.mkdir(parents=True, exist_ok=True)
    manifest, raw = read_versions()
    con = connect()
    con.register('raw_versions', raw)
    # One row per version, zone and county; duplicate lines describe separate polygon parts.
    con.execute("""CREATE TABLE pairs AS SELECT version_tag, effective_date::DATE AS effective_date, source, state, zone, state_zone,
            any_value(cwa) AS cwa, any_value(zone_name) AS zone_name, county_fips, any_value(county_name) AS county_name,
            any_value(time_zone) AS time_zone, any_value(fe_area) AS fe_area, count(*) AS polygon_parts
        FROM raw_versions GROUP BY version_tag, effective_date, source, state, zone, state_zone, county_fips""")
    con.execute("""CREATE TABLE versions AS
        WITH tags AS (SELECT version_tag, lead(effective_date) OVER (ORDER BY effective_date) AS next_effective_date
                      FROM (SELECT DISTINCT version_tag, effective_date FROM pairs))
        SELECT p.*, t.next_effective_date,
            count(*) OVER (PARTITION BY p.version_tag, p.state_zone) AS counties_in_zone,
            count(*) OVER (PARTITION BY p.version_tag, p.county_fips) AS zones_in_county,
            count(*) OVER (PARTITION BY p.version_tag, p.county_fips) > 1 AS partial_county
        FROM pairs p JOIN tags t USING (version_tag)""")
    assert con.execute('SELECT count(*) - count(DISTINCT (version_tag, state_zone, county_fips)) FROM versions').fetchone()[0] == 0
    copy_query(con, 'SELECT * FROM versions ORDER BY effective_date, state_zone, county_fips', ZONE_COUNTY)

    se_abbr = list(SE_ABBR_BY_FIPS.values())
    copy_query(con, f"""SELECT version_tag, effective_date, next_effective_date, any_value(source) AS source,
            any_value(raw_lines) AS raw_lines, any_value(malformed_lines) AS malformed_lines_skipped,
            any_value(disagreeing_rows) AS state_zone_field_disagreements, count(*) AS zone_county_pairs,
            count(*) FILTER (WHERE state IN {sql_list(se_abbr)}) AS southeast_pairs,
            count(DISTINCT state_zone) FILTER (WHERE state IN {sql_list(se_abbr)}) AS southeast_zones,
            count(DISTINCT county_fips) FILTER (WHERE state IN {sql_list(se_abbr)} AND partial_county) AS southeast_split_counties
        FROM versions v JOIN (SELECT version_tag, any_value(raw_lines) AS raw_lines, any_value(malformed_lines) AS malformed_lines,
                count(*) FILTER (WHERE fields_disagree) AS disagreeing_rows FROM raw_versions GROUP BY 1) USING (version_tag)
        GROUP BY ALL ORDER BY effective_date""", ZONE_REPORTS / 'version_inventory.csv', fmt='csv')
    write_json({'missing_versions': manifest.get('missing_versions', {}),
                'note': 'Releases that changed Southeast zones but were never archived. Events in these periods rely on name-verified fallback.'},
               ZONE_REPORTS / 'missing_versions.json')

    # Zone membership changes between consecutive versions in the six states.
    con.execute(f"""CREATE TABLE se_zone_sets AS SELECT version_tag, effective_date, state_zone, any_value(zone_name) AS zone_name,
            string_agg(county_fips, ';' ORDER BY county_fips) AS counties
        FROM versions WHERE state IN {sql_list(se_abbr)} GROUP BY ALL""")
    copy_query(con, """WITH tags AS (SELECT version_tag, effective_date, lag(version_tag) OVER (ORDER BY effective_date) AS prev_tag
                      FROM (SELECT DISTINCT version_tag, effective_date FROM se_zone_sets)),
        pairs AS (SELECT prev_tag AS from_tag, version_tag AS to_tag, effective_date FROM tags WHERE prev_tag IS NOT NULL),
        a AS (SELECT p.from_tag, p.to_tag, p.effective_date, s.state_zone, s.zone_name, s.counties
              FROM pairs p JOIN se_zone_sets s ON s.version_tag = p.from_tag),
        b AS (SELECT p.from_tag, p.to_tag, p.effective_date, s.state_zone, s.zone_name, s.counties
              FROM pairs p JOIN se_zone_sets s ON s.version_tag = p.to_tag)
        SELECT coalesce(a.from_tag, b.from_tag) AS from_tag, coalesce(a.to_tag, b.to_tag) AS to_tag,
            coalesce(a.effective_date, b.effective_date) AS effective_date, coalesce(a.state_zone, b.state_zone) AS state_zone,
            CASE WHEN a.state_zone IS NULL THEN 'added' WHEN b.state_zone IS NULL THEN 'removed'
                 WHEN a.counties<>b.counties THEN 'counties_changed' ELSE 'renamed' END AS change_type,
            a.zone_name AS name_before, b.zone_name AS name_after, a.counties AS counties_before, b.counties AS counties_after
        FROM a FULL JOIN b ON a.from_tag = b.from_tag AND a.state_zone = b.state_zone
        WHERE a.state_zone IS NULL OR b.state_zone IS NULL OR a.counties<>b.counties OR upper(a.zone_name)<>upper(b.zone_name)
        ORDER BY effective_date, state_zone""", ZONE_REPORTS / 'zone_changes_southeast.csv', fmt='csv')

    # Independent county names: Census names plus all correlation-file labels for the FIPS.
    labels = {}
    for fips, name in con.execute(f"""SELECT STATE || COUNTY, regexp_replace(CTYNAME, ' County$', '')
            FROM read_csv({lit(CENSUS_COUNTY_POP)}, all_varchar=true, encoding='latin-1')
            WHERE SUMLEV = '050' AND STATE IN {sql_list(SE_STATE_FIPS)}
            UNION SELECT DISTINCT county_fips, county_name FROM versions WHERE state IN {sql_list(se_abbr)}""").fetchall():
        labels.setdefault(fips, set()).add(name)
    census_fips = {r[0] for r in con.execute(f"""SELECT STATE || COUNTY FROM read_csv({lit(CENSUS_COUNTY_POP)}, all_varchar=true, encoding='latin-1')
        WHERE SUMLEV = '050' AND STATE IN {sql_list(SE_STATE_FIPS)}""").fetchall()}
    matcher = ZoneMatcher(con, se_abbr, labels, manifest.get('missing_versions', {}))

    print('Mapping Southeast NOAA zone events and verifying county-coded events', flush=True)
    events = con.execute(f"""SELECT event_id_clean AS event_id, episode_id_clean AS episode_id, STATE AS state_name, CZ_TYPE AS source_cz_type,
            state_fips_clean AS state_fips, cz_code_clean AS zone, county_fips AS noaa_county_fips, CZ_NAME AS cz_name, WFO AS wfo,
            EVENT_TYPE AS event_type, begin_utc, CAST(begin_utc AS DATE) AS begin_date
        FROM read_parquet({lit(CLEANED / 'storm_details.parquet')})
        WHERE CZ_TYPE IN ('Z', 'C') AND STATE IN {sql_list(SOUTHEAST_STATES)}""").df()
    events['state_zone'] = events.state_fips.map(SE_ABBR_BY_FIPS) + events.zone
    zone_records, county_checks = [], []
    for ev in events.itertuples(index=False):
        day = pd.Timestamp(ev.begin_date)
        if ev.source_cz_type == 'C':
            valid = ev.noaa_county_fips in census_fips
            if valid and names_match(ev.cz_name, labels.get(ev.noaa_county_fips, ())):
                county_checks.append({'event_id': ev.event_id, 'county_check': 'county_code_verified'})
                continue
            m = matcher.match(ev.state_zone, ev.cz_name, ev.wfo, day)
            # Reinterpret as a zone only on a verified zone name, never on the weak tier.
            if m['match_status'] in ('effective_version_verified', 'other_version_verified', 'zone_code_unmatched_name_verified'):
                county_checks.append({'event_id': ev.event_id, 'county_check': 'county_code_reinterpreted_as_zone'})
                zone_records.append({'event_id': ev.event_id, 'state_zone': ev.state_zone, **m})
            else:
                county_checks.append({'event_id': ev.event_id, 'county_check': 'county_code_invalid' if not valid else 'county_name_mismatch'})
            continue
        zone_records.append({'event_id': ev.event_id, 'state_zone': ev.state_zone, **matcher.match(ev.state_zone, ev.cz_name, ev.wfo, day)})
    con.register('events_df', events)
    con.register('matches_df', pd.DataFrame(zone_records))
    con.register('checks_df', pd.DataFrame(county_checks))
    con.execute(f"""CREATE TABLE zone_matches AS SELECT e.*, m.* EXCLUDE (event_id, state_zone),
            m.match_status IN {sql_list(ACCEPTED)} AND m.mapped_tag IS NOT NULL AS mapping_accepted
        FROM events_df e JOIN matches_df m USING (event_id)""")
    con.execute('CREATE TABLE county_checks AS SELECT e.*, c.county_check FROM events_df e JOIN checks_df c USING (event_id)')
    n_zone = int((events.source_cz_type == 'Z').sum())
    assert con.execute("SELECT count(*) FROM zone_matches WHERE source_cz_type = 'Z'").fetchone()[0] == n_zone
    assert con.execute('SELECT count(*) FROM county_checks').fetchone()[0] == len(events) - n_zone
    copy_query(con, """SELECT event_id, noaa_county_fips, state_name, cz_name, event_type, begin_utc, county_check
        FROM county_checks ORDER BY event_id""", SOUTHEAST / 'county_event_checks.parquet')
    copy_query(con, f"""SELECT z.event_id, z.episode_id, z.source_cz_type, z.state_name, z.state_zone, z.mapped_zone, z.cz_name, z.wfo,
            z.event_type, z.begin_utc, z.effective_tag, z.mapped_tag, z.match_status, z.name_similarity, z.in_missing_version_gap,
            v.county_fips, v.county_name, v.zone_name AS nws_zone_name, v.counties_in_zone, v.zones_in_county, v.partial_county
        FROM zone_matches z JOIN versions v ON v.version_tag=z.mapped_tag AND v.state_zone=z.mapped_zone
        WHERE z.mapping_accepted ORDER BY z.event_id, v.county_fips""", SOUTHEAST / 'zone_event_counties.parquet')
    copy_query(con, """SELECT state_name, year(begin_utc) AS year, source_cz_type, match_status, in_missing_version_gap, count(*) AS events
        FROM zone_matches GROUP BY ALL ORDER BY ALL""", ZONE_REPORTS / 'match_status_by_state_year.csv', fmt='csv')
    copy_query(con, """SELECT state_name, year(begin_utc) AS year, county_check, count(*) AS events
        FROM county_checks GROUP BY ALL ORDER BY ALL""", ZONE_REPORTS / 'county_code_checks_by_state_year.csv', fmt='csv')
    copy_query(con, """SELECT 'zone_event' AS record, match_status AS status, state_name, state_zone AS code, cz_name, wfo,
            min(begin_utc) AS first_begin, max(begin_utc) AS last_begin, count(*) AS events, string_agg(DISTINCT event_type, '; ') AS event_types
        FROM zone_matches WHERE NOT mapping_accepted AND source_cz_type = 'Z' GROUP BY ALL
        UNION ALL
        SELECT 'county_event', county_check, state_name, noaa_county_fips, cz_name, NULL, min(begin_utc), max(begin_utc), count(*),
            string_agg(DISTINCT event_type, '; ')
        FROM county_checks WHERE county_check NOT IN ('county_code_verified', 'county_code_reinterpreted_as_zone') GROUP BY ALL
        ORDER BY events DESC""", ZONE_REPORTS / 'unmapped_events.csv', fmt='csv')
    copy_query(con, """SELECT z.source_cz_type, z.match_status, z.state_zone, z.mapped_zone, z.cz_name, v.zone_name AS nws_zone_name,
            z.mapped_tag, z.name_similarity, count(*) AS events
        FROM zone_matches z JOIN (SELECT DISTINCT version_tag, state_zone, zone_name FROM versions) v
          ON v.version_tag=z.mapped_tag AND v.state_zone=z.mapped_zone
        WHERE z.match_status <> 'effective_version_verified' OR z.source_cz_type = 'C' GROUP BY ALL ORDER BY events DESC""",
               ZONE_REPORTS / 'fallback_and_reinterpreted_matches.csv', fmt='csv')
    mapped = f"read_parquet({lit(SOUTHEAST / 'zone_event_counties.parquet')})"
    summary = {
        'versions': len(matcher.tag_dates),
        'southeast_zone_events': n_zone,
        'zone_match_status': dict(con.execute("SELECT match_status, count(*) FROM zone_matches WHERE source_cz_type = 'Z' GROUP BY 1 ORDER BY 2 DESC").fetchall()),
        'county_coded_events': len(events) - n_zone,
        'county_code_checks': dict(con.execute('SELECT county_check, count(*) FROM county_checks GROUP BY 1 ORDER BY 2 DESC').fetchall()),
        'events_in_missing_version_gaps': con.execute('SELECT count(*) FROM zone_matches WHERE in_missing_version_gap').fetchone()[0],
        'mapped_event_county_rows': con.execute(f'SELECT count(*) FROM {mapped}').fetchone()[0],
        'mapped_rows_partial_county': con.execute(f'SELECT count(*) FILTER (WHERE partial_county) FROM {mapped}').fetchone()[0],
        'rules': {'verified_similarity': VERIFIED, 'weak_similarity': WEAK, 'accepted': list(ACCEPTED),
                  'county_reinterpretation': 'verified zone names only'},
    }
    write_json(summary, ZONE_REPORTS / 'summary.json')
    print(json.dumps(summary, indent=2), flush=True)
    con.close()


if __name__ == '__main__':
    main()
