"""Prepare an allowlisted EDA release; --upload publishes to the private team dataset.

Uploads no raw observations or test outcomes, replaces no existing release, and
leaves the legacy dataset/card intact. Credentials come from hf auth login.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from src import config as C
from src.major_outage import manifest, feature_columns

FILES = [f'{design}_{split}.parquet' for design in ('alert', 'onset')
         for split in ('train', 'validation', 'test_features')] + ['manifest.json']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(folder):
    meta = manifest()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for design in ('alert', 'onset'):
        feature_columns(design)
        names = set(pq.read_schema(C.MAJOR_INPUT_DIR / f'{design}_test_features.parquet').names)
        if names & set(meta['outcomes']):
            raise ValueError('Refusing a release with test outcome columns')
    for name in FILES:
        shutil.copyfile(C.MAJOR_INPUT_DIR / name, folder / name)
    card = (C.ROOT / 'docs/dataset_card.md').read_text(encoding='utf-8')
    sources = (C.ROOT / 'docs/data_sources.md').read_text(encoding='utf-8')
    features = (C.ROOT / 'docs/feature_dictionary.md').read_text(encoding='utf-8')
    text = ('# Southeast EDA release: 2026-10-23-eda-v2\n\n'
            'Derived from EAGLE-I recorded outages by Christa Brelsford et al., '
            'https://doi.org/10.6084/m9.figshare.24237376.v4, licensed CC BY 4.0 '
            '(https://creativecommons.org/licenses/by/4.0/). Changes: six-state filtering, '
            'historical geography matching, county prediction windows, coverage-based labels '
            'and pre-prediction features. Also credits NOAA/NWS, Iowa Environmental Mesonet '
            'and U.S. Census Bureau. No endorsement is implied.\n\n'
            'This release contains development labels and test features only. Earlier 2025 '
            'exploration and prior 2024 development remain disclosed. Do not create test labels '
            'until the final protocol is frozen.\n\n' + card + '\n\n' + features + '\n\n' + sources)
    (folder / 'README.md').write_text(text, encoding='utf-8')
    inventory = {name: {'sha256': digest(folder / name), 'bytes': (folder / name).stat().st_size}
                 for name in FILES + ['README.md']}
    checks = {'release': C.HF_EDA_RELEASE, 'files': inventory}
    (folder / 'checksums.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--repo-id', default=C.HF_REPO_ID)
    args = parser.parse_args()
    folder = C.ROOT / 'tmp/hf-eda-release'
    checks = prepare(folder)
    print(f"Prepared {len(checks['files']) + 1} files, {sum(f['bytes'] for f in checks['files'].values()):,} bytes")
    print(f'Destination: {args.repo_id}/{C.HF_EDA_RELEASE}')
    if not args.upload:
        print('No upload performed. Add --upload after local hf auth login.')
        return
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    info = api.repo_info(args.repo_id, repo_type='dataset')
    if not info.private:
        raise ValueError('This uploader requires an existing private team dataset')
    existing = api.list_repo_files(args.repo_id, repo_type='dataset', revision=info.sha)
    if any(p.startswith(C.HF_EDA_RELEASE + '/') for p in existing):
        raise ValueError('Release already exists; refusing to overwrite it. Choose a new reviewed release identifier.')
    names = FILES + ['README.md', 'checksums.json']
    commit = api.create_commit(repo_id=args.repo_id, repo_type='dataset', parent_commit=info.sha,
                              commit_message='Add versioned Southeast EDA inputs; test features only',
                              operations=[CommitOperationAdd(path_in_repo=f'{C.HF_EDA_RELEASE}/{n}',
                                                             path_or_fileobj=str(folder / n)) for n in names])
    receipt = {'repo_id': args.repo_id, 'revision': commit.oid, 'release': C.HF_EDA_RELEASE,
               'checksums_sha256': digest(folder / 'checksums.json')}
    (C.ROOT / 'docs/eda_release.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print('Uploaded immutable release at revision:', commit.oid)


if __name__ == '__main__':
    main()
