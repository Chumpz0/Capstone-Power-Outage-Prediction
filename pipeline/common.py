"""Shared rules for pipeline stages.

Defines the study area, DuckDB session settings, the SQL macros that standardize
county FIPS codes, UTC times and EAGLE-I quarter-hour slots, atomic output writers,
and the content-based build hash used for cache invalidation.

Stages import these instead of restating rules. Every build hash includes the macro
text and the bytes of each source file, so changing either a rule or a source
invalidates the cached outputs built from it.
"""
import hashlib
import json
import os
from pathlib import Path
import duckdb

from paths import CACHE, COMBINED, ROOT

# Study area: NOAA STATE name -> (two-digit state FIPS, USPS abbreviation).
from settings import SOUTHEAST_STATES, SPLIT_YEARS, SPLIT_BOUNDARIES
SE_STATE_FIPS = tuple(fips for fips, _ in SOUTHEAST_STATES.values())
SE_ABBR_BY_FIPS = dict(SOUTHEAST_STATES.values())

# Chronological splits by UTC year of the prediction time. 2025 is the final held-out test.


def split_sql(t):
    return (f"CASE WHEN year({t})<={SPLIT_YEARS['train'][1]} THEN 'train' "
            f"WHEN year({t})={SPLIT_YEARS['validation'][0]} THEN 'validation' "
            f"WHEN year({t})={SPLIT_YEARS['test'][0]} THEN 'test' ELSE 'out_of_scope' END")


# NOAA event types grouped by hazard. PROVISIONAL scope pending the event-type decision
# (docs/decision_log.md): groups in STORM_GROUPS create prediction opportunities;
# 'temperature' and 'other' are reported but excluded.
HAZARD_GROUPS = {
    'convective_wind': ['Thunderstorm Wind', 'Tornado'],
    'hail': ['Hail'],
    'lightning': ['Lightning'],
    'rain_flood': ['Heavy Rain', 'Flash Flood', 'Flood', 'Debris Flow'],
    'coastal_surge': ['Coastal Flood', 'Storm Surge/Tide'],
    'tropical': ['Hurricane (Typhoon)', 'Hurricane', 'Tropical Storm', 'Tropical Depression'],
    'synoptic_wind': ['High Wind', 'Strong Wind'],
    'winter': ['Winter Storm', 'Ice Storm', 'Heavy Snow', 'Winter Weather', 'Sleet', 'Blizzard'],
    'temperature': ['Heat', 'Excessive Heat', 'Cold/Wind Chill', 'Extreme Cold/Wind Chill', 'Frost/Freeze'],
}
STORM_GROUPS = ('convective_wind', 'hail', 'lightning', 'rain_flood', 'coastal_surge', 'tropical', 'synoptic_wind', 'winter')


# NWS VTEC phenomena codes by hazard family, mirroring HAZARD_GROUPS for alerts.
ALERT_FAMILIES = {
    'tropical': ['HU', 'TR', 'TY', 'HI', 'TI', 'SS'],
    'convective': ['TO', 'SV', 'EW'],
    'wind': ['HW', 'WI', 'LW'],
    'winter': ['WS', 'IS', 'BZ', 'WW', 'ZR', 'LE', 'SQ', 'HS', 'SN', 'IP', 'SB', 'LB'],
    'flood': ['FF', 'FA', 'FL', 'HY'],
    'coastal': ['CF', 'SU', 'LS'],
    'temperature': ['EC', 'WC', 'EH', 'HT', 'XH', 'CW', 'FR', 'FZ', 'HZ'],
}
STORM_ALERT_FAMILIES = ('tropical', 'convective', 'wind', 'winter', 'flood', 'coastal')


def alert_family_sql(phenomena):
    cases = ' '.join(f"WHEN {phenomena} IN {sql_list(codes)} THEN '{fam}'" for fam, codes in ALERT_FAMILIES.items())
    return f"CASE {cases} ELSE 'other' END"


def hazard_group_sql(event_type):
    cases = ' '.join(f'WHEN {event_type} IN {sql_list(types)} THEN {lit(group)}' for group, types in HAZARD_GROUPS.items())
    return f"CASE {cases} ELSE 'other' END"


def greedy_window_starts(rows, hours):
    """Non-overlapping county windows from (county, time) pairs sorted by county, then time.

    The first time starts a window; later times before start + hours belong to it; the
    next time after that starts the next window. Returns the (county, start) pairs.
    """
    starts, last = [], {}
    for county, t in rows:
        if county not in last or (t - last[county]).total_seconds() >= hours * 3600:
            starts.append((county, t))
            last[county] = t
    return starts


def combined_csv(filename):
    """Read a combined CSV with every field as text and conventional NA markers as null."""
    return (f"read_csv({lit(COMBINED / filename)}, header=true, all_varchar=true, strict_mode=true, ignore_errors=false, "
            "nullstr=['','NA','N/A','NULL','null','NaN','nan','None','<NA>','#N/A'])")


MACROS = (
    # Zero-padded numeric code; NULL for non-numeric or over-long input.
    """CREATE OR REPLACE MACRO padded_code(x, width) AS CASE
        WHEN regexp_full_match(trim(x), '[0-9]+') AND length(trim(x))<=width
        THEN lpad(trim(x), width, '0') END""",
    # NOAA local time to UTC using the recorded numeric offset (EST-5 -> +5 h). No DST localization.
    """CREATE OR REPLACE MACRO utc_from_offset(t, z) AS
        (t - try_cast(nullif(regexp_extract(z, '([+-]?[0-9]+)$', 1), '') AS INTEGER) * INTERVAL '1 hour') AT TIME ZONE 'UTC'""",
    # EAGLE-I collection runs start at :00, :15, :30 and :45 UTC.
    "CREATE OR REPLACE MACRO slot_floor(t) AS time_bucket(INTERVAL 15 MINUTE, t)",
    "CREATE OR REPLACE MACRO slot_ceil(t) AS CASE WHEN slot_floor(t)=t THEN t ELSE slot_floor(t) + INTERVAL 15 MINUTE END",
    "CREATE OR REPLACE MACRO on_slot(t) AS slot_floor(t)=t",
)


def lit(value):
    return "'" + str(value).replace("'", "''") + "'"


def sql_list(values):
    return '(' + ', '.join(lit(v) for v in values) + ')'


def connect(database=':memory:'):
    """DuckDB session in UTC with the shared macros.

    Resources default to 2 threads and 4GB; override with the POWER_DUCKDB_THREADS
    and POWER_DUCKDB_MEMORY environment variables.
    """
    con = duckdb.connect(database)
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit={lit(os.environ.get('POWER_DUCKDB_MEMORY', '4GB'))}")
    con.execute(f"SET threads={int(os.environ.get('POWER_DUCKDB_THREADS', '2'))}")
    con.execute('SET preserve_insertion_order=false')
    for macro in MACROS:
        con.execute(macro)
    return con


def file_sha256(path, cache_dir=None):
    """SHA-256 of a file, cached by size and modification time.

    Large sources are hashed once; any rewrite changes the modification time and
    forces a fresh hash. Cache keys are relative to the project root, so a moved
    project keeps its cache when timestamps are preserved.
    """
    path = Path(path).resolve()
    cache_file = Path(cache_dir or CACHE) / 'file_sha256.json'
    stat = path.stat()
    key = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    entry = cache.get(key)
    if entry and entry['size'] == stat.st_size and entry['mtime_ns'] == stat.st_mtime_ns:
        return entry['sha256']
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    cache[key] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': digest}
    write_json(cache, cache_file)
    return digest


def build_hash(version, query, sources=(), upstream=(), cache_dir=None):
    """Content hash of everything that determines an output.

    Covers the rule version, the shared SQL macros, the query text with the project
    root replaced (moving the whole project does not force a rebuild), the bytes of
    each source file, and the sidecar metadata of each upstream output.
    """
    digest = hashlib.sha256(f'version:{version}\n'.encode())
    for macro in MACROS:
        digest.update(macro.encode())
    digest.update(query.replace(str(ROOT), '<ROOT>').encode())
    for source in sources:
        digest.update(f'\nsource:{file_sha256(source, cache_dir)}'.encode())
    for meta in upstream:
        digest.update(b'\nupstream:' + Path(meta).read_bytes())
    return digest.hexdigest()


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(obj, indent=2, default=str) + '\n', encoding='utf-8')
    pending.replace(path)


def copy_query(con, query, path, fmt='parquet'):
    """Write query results atomically: a pending file replaces the target only when complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = 'FORMAT PARQUET, COMPRESSION ZSTD' if fmt == 'parquet' else 'FORMAT CSV, HEADER'
    pending = path.with_name(f'{path.stem}.pending{path.suffix}')
    con.execute(f'COPY ({query}) TO {lit(pending)} ({options})')
    pending.replace(path)


class ParquetCache:
    """Rebuild a Parquet output only when its build hash or rule version changes.

    Each output has a sidecar <name>.meta.json holding {version_key: version,
    'build_hash': ...}. With check=True nothing is written: statuses are recorded,
    and outputs downstream of a stale output are reported as stale too.
    """

    def __init__(self, con, version, version_key, force=False, check=False):
        self.con, self.version, self.version_key = con, version, version_key
        self.force, self.check = force, check
        self.meta, self.status = {}, {}
        self._stale = set()

    def plan(self, query, path, sources=(), upstream=()):
        """Record the output's status; return True when it must be (re)built."""
        path = Path(path)
        meta_path = path.with_suffix('.meta.json')
        upstream = [Path(u) for u in upstream]
        if self.check and any(u in self._stale or not u.exists() for u in upstream):
            self.status[path.name] = 'rebuild: upstream output changes'
            self._stale.add(meta_path)
            return True
        meta = {self.version_key: self.version, 'build_hash': build_hash(self.version, query, sources, upstream)}
        self.meta[path.name] = meta
        current = json.loads(meta_path.read_text()) if path.exists() and meta_path.exists() else None
        if current == meta and not self.force:
            self.status[path.name] = 'reuse'
            return False
        self.status[path.name] = ('rebuild: forced' if current == meta else
                                  'rebuild: no output' if current is None else 'rebuild: hash or version changed')
        self._stale.add(meta_path)
        return True

    def build(self, query, path, sources=(), upstream=()):
        """Write the query to path unless the cached output still matches (or check mode)."""
        path = Path(path)
        if not self.plan(query, path, sources, upstream):
            print(f'Reusing {path.name} ({self.version_key} {self.version}, hash matches)', flush=True)
            return
        if self.check:
            return
        print(f'Building {path.name}', flush=True)
        meta_path = path.with_suffix('.meta.json')
        meta_path.unlink(missing_ok=True)
        copy_query(self.con, query, path)
        meta_path.write_text(json.dumps(self.meta[path.name], indent=2))
