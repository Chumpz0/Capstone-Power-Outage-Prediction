"""Execute and audit the EDA; no downloads, fitting, commits or uploads."""
from pathlib import Path
import json
import subprocess
import sys
import time
import unittest
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.time()
    subprocess.run([sys.executable, str(ROOT / "scripts/check_geography_criteria.py")], cwd=ROOT, check=True)
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise RuntimeError("Calculation tests failed")
    records = []
    for path in sorted((ROOT / "notebooks").glob("0[1-4]_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        try:
            NotebookClient(notebook, km=km, timeout=1200,
                resources={"metadata": {"path": str(path.parent)}}).execute()
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
        records.append(dict(notebook=path.relative_to(ROOT).as_posix(),
                            executed_code_cells=sum(c.cell_type == "code" for c in notebook.cells)))
        print(f"Executed {path.name}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_data_review.py")], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    result = dict(status="passed", tests_command="python -m unittest discover -s tests -v", tests=tests.testsRun,
                  notebooks=records, elapsed_seconds=round(time.time()-start, 1),
                  scope="All notebooks re-executed, including previously exposed 2024/2025 descriptive EDA; forecast target rules remain training-only")
    (ROOT / "docs/eda_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts/build_eda_metadata.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
