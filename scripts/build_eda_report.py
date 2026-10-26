"""Build reports/EDA_report.pdf: the EDA report with the dataset card attached.

Renders reports/eda_report.md and reports/dataset_card.md to one HTML page, then prints it to PDF
with a headless Chromium browser (Microsoft Edge or Google Chrome, whichever is installed).
Run after notebook 06 so the figures are current:  python scripts/build_eda_report.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
HTML_OUT = REPORTS / "EDA_report.html"
PDF_OUT = REPORTS / "EDA_report.pdf"

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chrome",
]

# Print styles: same palette as the charts (ink, blue accent, light surface)
CSS = """
@page { size: Letter; margin: 0.6in 0.65in; }
body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #0b0b0b; }
h1 { font-size: 22pt; margin: 0 0 4pt; color: #0b0b0b; }
h2 { font-size: 15pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 2px solid #2a78d6; }
h3 { font-size: 12pt; margin: 12pt 0 4pt; color: #2a78d6; }
p.subtitle { color: #52514e; margin-top: 0; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f1f0ec; padding: 0 2pt; border-radius: 2px; }
pre { background: #f6f5f1; border: 1px solid #e3e2dd; padding: 8pt; font-size: 8pt; line-height: 1.3; overflow: hidden; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: auto; }
th { background: #eef4fc; text-align: left; }
th, td { border: 1px solid #d6d5d0; padding: 3pt 5pt; vertical-align: top; }
tr { page-break-inside: avoid; }
img { max-width: 100%; display: block; margin: 6pt auto; page-break-inside: avoid; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6pt; }
.grid2 p { margin: 0; }
.pagebreak { page-break-after: always; }
.attachment { color: #52514e; font-size: 9.5pt; border: 1px solid #d6d5d0; background: #fcfcfb; padding: 6pt 8pt; }
.card table { font-size: 8pt; }
.card th, .card td { padding: 2pt 4pt; }
"""


def render(md_path):
    """Markdown file -> HTML fragment (tables, code blocks, markdown inside HTML blocks)."""
    return markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "md_in_html"])


def find_browser():
    for b in BROWSERS:
        path = shutil.which(b) or (b if Path(b).exists() else None)
        if path:
            return path
    sys.exit("No Edge or Chrome found: open reports/EDA_report.html in a browser and print it to PDF instead.")


def main():
    body = render(REPORTS / "eda_report.md")
    card = render(REPORTS / "dataset_card.md")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Storm-to-Outage EDA</title>
<base href="{REPORTS.as_uri()}/"><style>{CSS}</style></head>
<body>
{body}
<div class="pagebreak"></div>
<p class="attachment"><b>Attachment: dataset card.</b> Also available as <code>reports/dataset_card.md</code> in the repository and as the Hugging Face dataset README.</p>
<div class="card">{card}</div>
</body></html>"""
    HTML_OUT.write_text(html, encoding="utf-8")

    # Headless print; a throwaway profile keeps it independent of any open browser window
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={PDF_OUT}", HTML_OUT.as_uri()],
                       check=True, capture_output=True, timeout=180)
    print(f"wrote {PDF_OUT.relative_to(ROOT)} ({PDF_OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
