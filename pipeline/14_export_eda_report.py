"""Execute the training-only EDA notebook and export a standalone HTML report.

No models are fitted. The notebook reads training inputs and the stage-12 training
timing audit only. Embedded figures and tables work offline without Jupyter.
"""
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import re

import nbformat
from nbclient import NotebookClient

from paths import EDA_NOTEBOOK, EDA_REPORT, ROOT


def inline(text):
    text = escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)


def markdown(text):
    blocks = []
    for block in text.strip().split('\n\n'):
        match = re.match(r'^(#{1,3}) (.*)$', block)
        if match:
            level = len(match[1])
            blocks.append(f'<h{level}>{inline(match[2])}</h{level}>')
        else:
            blocks.append(f'<p>{inline(block)}</p>')
    return '\n'.join(blocks)


def main():
    notebook = nbformat.read(EDA_NOTEBOOK, as_version=4)
    NotebookClient(notebook, timeout=300, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.validate(notebook)
    body = []
    charts = tables = 0
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            body.append(markdown(cell.source))
            continue
        for output in cell.get('outputs', []):
            if output.output_type == 'error':
                raise RuntimeError(output.evalue)
            data = output.get('data', {})
            if 'image/png' in data:
                charts += 1
                body.append(f'<figure><img alt="EDA figure {charts}; associated values follow in a table" '
                            f'src="data:image/png;base64,{data["image/png"]}"></figure>')
            elif 'text/html' in data:
                tables += 1
                body.append('<div class="table-wrap">' + data['text/html'] + '</div>')
            elif output.output_type == 'stream' and output.get('name') == 'stdout':
                body.append('<pre>' + escape(output.text) + '</pre>')
            elif 'text/plain' in data:
                body.append('<pre>' + escape(data['text/plain']) + '</pre>')
    if charts != 5 or tables != 10:
        raise RuntimeError(f'Unexpected EDA output inventory: {charts} charts, {tables} tables')
    lab = '''<aside><strong>Reading guide.</strong> Coverage and selection are the central findings:
    only a small subset of alert windows meet the reporting rule, with strong county-size differences.
    Observed peaks are heavily right-skewed, missing values require explicit treatment, and storm-family
    representation varies across years. These are descriptive findings, not model results or causal effects.
    A peak from an incomplete window is a lower bound; the severity completeness policy remains open.
    Prior 2025 exploration is disclosed; this report uses no 2025 outcomes.</aside>'''
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    source_hash = sha256(EDA_NOTEBOOK.read_bytes()).hexdigest()
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Southeast major-outage EDA report</title><style>
    body{margin:0;background:#f5f6f8;color:#18232f;font:16px/1.65 system-ui,sans-serif}
    main{max-width:1050px;margin:32px auto;padding:40px;background:white;border-radius:12px}
    h1{font-size:32px;line-height:1.2}h2{margin-top:48px;padding-top:20px;border-top:1px solid #dae0e6}
    aside{background:#eef4fa;border-left:4px solid #2a78d6;padding:20px;margin:24px 0}
    figure{margin:24px 0}img{max-width:100%;height:auto}.table-wrap{overflow-x:auto;margin:20px 0}
    table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #dde2e7;text-align:right}
    th{background:#f1f4f7}code,pre{font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere}
    footer{margin-top:40px;color:#556270;font-size:13px;overflow-wrap:anywhere}
    @media(max-width:700px){main{margin:0;padding:20px}h1{font-size:26px}}
    @media print{body{background:white}main{margin:0;padding:0}figure{break-inside:avoid}}
    </style></head><body><main>'''
    html += lab + '\n'.join(body)
    html += ('<h2>Handoff and sources</h2><p>The teammate owns baseline modeling. Preserve the approved '
             '72-hour alert windows, 5,000-customer target, covered population and chronological evaluation. '
             'Baseline design, severity completeness and implausible-peak county treatment remain open.</p>'
             '<p><a href="../../docs/dataset_card.md">Dataset card</a> · '
             '<a href="../../docs/feature_dictionary.md">Feature dictionary</a> · '
             '<a href="../../docs/evaluation_protocol.md">Evaluation safeguards</a></p>')
    html += f'<footer>Generated {stamp}. Source: notebooks/06_major_outage_eda.ipynb. '
    html += f'Notebook SHA-256 before execution: {source_hash}. {charts} charts; {tables} tables. '
    html += 'Rebuild: .venv/Scripts/python.exe pipeline/14_export_eda_report.py</footer></main></body></html>'
    EDA_REPORT.parent.mkdir(parents=True, exist_ok=True)
    EDA_REPORT.write_text(html, encoding='utf-8')
    nbformat.write(notebook, EDA_NOTEBOOK)
    print(f'Exported {EDA_REPORT}: {charts} charts and {tables} tables; notebook executed successfully.')


if __name__ == '__main__':
    main()
