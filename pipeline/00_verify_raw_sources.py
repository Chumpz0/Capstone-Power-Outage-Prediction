"""Verify the raw NOAA Storm Events and EAGLE-I files, and say where to get missing ones.

Run: python pipeline/00_verify_raw_sources.py           check data/01_raw/ against pipeline/raw_sources.json
     python pipeline/00_verify_raw_sources.py --record  write the manifest from the files present
The manifest is tracked in version control; the data are not. Stage 08 fetches the NWS,
IEM and Census sources itself and keeps its own manifests next to those files.

NCEI replaces Storm Events files when it revises them (the c-date in the file name
changes), so the exact files used here may no longer be downloadable. Keep the local
copies; a different c-date means different source data and a rebuild.
"""
import argparse
import json
import sys
from pathlib import Path

from common import file_sha256, write_json
from paths import RAW

MANIFEST = Path(__file__).with_name('raw_sources.json')
SOURCES = {
    'StormEvents_': 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/',
    '24237376.zip': 'EAGLE-I recorded outages, figshare article 24237376 (https://figshare.com/articles/dataset/24237376); '
                    'see Brelsford et al. 2024, https://doi.org/10.1038/s41597-024-03095-5',
}


def source_of(name):
    return next(url for prefix, url in SOURCES.items() if name.startswith(prefix))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true', help='Write the manifest from the files present.')
    args = parser.parse_args()
    if args.record:
        files = sorted(p for p in RAW.iterdir() if p.is_file() and (p.name.startswith('StormEvents_') or p.name == '24237376.zip'))
        manifest = {p.name: {'bytes': p.stat().st_size, 'sha256': file_sha256(p), 'source': source_of(p.name)} for p in files}
        write_json(manifest, MANIFEST)
        print(f'Recorded {len(manifest)} files in {MANIFEST}')
        return
    manifest = json.loads(MANIFEST.read_text())
    problems = []
    for name, info in manifest.items():
        path = RAW / name
        if not path.exists():
            problems.append(f'missing  {name}  <- {info["source"]}')
        elif path.stat().st_size != info['bytes'] or file_sha256(path) != info['sha256']:
            problems.append(f'changed  {name}  (size or SHA-256 differs from the manifest)')
    for line in problems:
        print(line)
    print(f'{len(manifest) - len(problems)} of {len(manifest)} raw files verified.')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
