"""Single source of project paths. Pipeline scripts import this module.

Running any script in this folder puts the folder on sys.path, so
`python pipeline/04_clean_datasets.py` works from any working directory.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent

DATA = Path(os.environ.get('POWER_DATA_ROOT', ROOT / 'data')).expanduser().resolve()
RAW = DATA / '01_raw'
COMBINED = DATA / '02_combined'
CLEANED = DATA / '03_cleaned'
MODEL_INPUT = DATA / '04_model_input'
REFERENCE = DATA / 'reference'
CENSUS_CT = REFERENCE / 'census_ct'
EAGLEI_ARCHIVE = REFERENCE / 'eaglei_archive'
CT_WEIGHTS = REFERENCE / 'ct_region_to_county_weights.csv'
MCC = EAGLEI_ARCHIVE / 'MCC.csv'
COVERAGE_HISTORY = EAGLEI_ARCHIVE / 'coverage_history.csv'
# Content-hash cache for large source files (see common.file_sha256).
CACHE = DATA / '.cache'

# Southeast major-outage pipeline (stages 08-12).
RAW_ZONE_COUNTY = RAW / 'nws_zone_county'
RAW_IEM_VTEC = RAW / 'iem_vtec'
RAW_CENSUS_COUNTIES = RAW / 'census_counties'
CENSUS_COUNTY_POP = RAW_CENSUS_COUNTIES / 'co-est2023-alldata.csv'
ZONE_COUNTY = REFERENCE / 'nws_zone_county_versions.parquet'
SOUTHEAST = CLEANED / 'southeast'

REPORTS = ROOT / 'reports'
QUALITY_REPORTS = REPORTS / 'quality'
PROFILE_REPORTS = REPORTS / 'full_profile'
ZONE_REPORTS = REPORTS / 'zone_crosswalk'
LABEL_REPORTS = REPORTS / 'label_audit'
FORECAST_REPORTS = REPORTS / 'forecast_feasibility'
DOCS = ROOT / 'docs'
EDA_NOTEBOOK = ROOT / 'notebooks' / '06_major_outage_eda.ipynb'
EDA_REPORT = REPORTS / 'eda' / 'major_outage_eda.html'
