"""Export schemas, null counts, fingerprints and environment for the EDA handoff."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    tables = []
    for path in sorted((ROOT / "data/processed/eda").rglob("*.parquet")):
        table = pq.read_table(path)
        tables.append({**fingerprint(path), "rows": table.num_rows,
                       "fields": [dict(name=f.name, type=str(f.type), nulls=table.column(f.name).null_count)
                                  for f in table.schema]})
    sources = json.loads((ROOT / "docs/data_audit.json").read_text())["inventory"]
    packages = ["pandas", "numpy", "pyarrow", "duckdb", "matplotlib", "seaborn", "nbformat", "nbclient", "ipykernel", "jupyter-client"]
    artifacts = [p for folder in ["src", "scripts", "tests", "notebooks", "docs"] for p in (ROOT / folder).rglob("*")
                 if p.is_file() and p.suffix in {".py", ".ipynb", ".md", ".json", ".png"}
                 and p.name != "eda_metadata.json"]
    artifacts += [ROOT / "README.md", ROOT / "data/README.md", ROOT / "requirements.txt"]
    result = dict(schema_version=1, source_inventory=sources, derived_tables=tables,
                  artifact_files=[fingerprint(p) for p in sorted(artifacts)],
                  python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in packages},
                  base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  commit_note="Working-tree handoff; base commit does not include uncommitted checkpoint work",
                  semantics="docs/eda_metadata.md", validation="docs/eda_validation.json",
                  reproducibility="Source data hashes identify existing inputs; no data refresh is performed")
    (ROOT / "docs/eda_metadata.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Metadata: {len(sources)} source files, {len(tables)} derived tables, {len(artifacts)} artifacts", flush=True)


if __name__ == "__main__":
    main()
