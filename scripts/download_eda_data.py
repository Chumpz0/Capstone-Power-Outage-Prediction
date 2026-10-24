"""Download and verify the pinned EDA release without overwriting different local data."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C


def main():
    from huggingface_hub import hf_hub_download
    receipt_path = C.ROOT / 'docs/eda_release.json'
    if not receipt_path.exists():
        raise SystemExit('Release not published yet: no docs/eda_release.json. Build stage 13 or ask the data owner to publish.')
    receipt = json.loads(receipt_path.read_text())
    def fetch(name):
        return Path(hf_hub_download(repo_id=receipt['repo_id'], repo_type='dataset',
                                   revision=receipt['revision'], filename=f"{receipt['release']}/{name}"))
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    checks_path = fetch('checksums.json')
    if sha(checks_path) != receipt['checksums_sha256']:
        raise ValueError('Release checksum manifest does not match the pinned receipt')
    checks = json.loads(checks_path.read_text())
    expected = {f'{d}_{s}.parquet' for d in ('alert', 'onset') for s in ('train', 'validation', 'test_features')}
    expected |= {'manifest.json', 'README.md'}
    if set(checks['files']) != expected or checks['release'] != receipt['release']:
        raise ValueError('Unexpected release files or identifier')
    downloads = {}
    for name, record in checks['files'].items():
        source = fetch(name)
        if source.stat().st_size != record['bytes'] or sha(source) != record['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
        dest = C.MAJOR_INPUT_DIR / name
        if dest.exists() and sha(dest) != record['sha256']:
            raise FileExistsError(f'Refusing to replace different local data: {dest}')
        downloads[name] = source
    C.MAJOR_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, source in downloads.items():
        shutil.copyfile(source, C.MAJOR_INPUT_DIR / name)
    print(f'Verified {len(downloads)} files from revision {receipt["revision"]}')


if __name__ == '__main__':
    main()
