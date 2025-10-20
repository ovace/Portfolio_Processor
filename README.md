Portfolio Processor

A small, configurable Python toolkit to extract portfolio holdings from heterogeneous CSV and Excel reports, normalize them to a canonical schema, clean and deduplicate the results, and compute simple portfolio metrics (CAGR per row and optional portfolio Beta).
Designed for reliability with multiple vendor formats, configurable column-mapping, and clean architecture (reusable utilities + thin orchestration script).

Table of contents

Features

Repository layout

Requirements

Install

Configuration

config/default_settings.json example

Column mapping

Usage

Command line examples

How files are detected & processed

Metrics

Logging & audit

Testing locally

Contributing

Troubleshooting

License

Features

Detects and extracts one or more tables from CSV or Excel (.xls, .xlsx) files.

Normalizes column names via configurable mapping file (config/column_mapping.json) into canonical fields:

Account, Symbol, Quantity, Acquisition Date, Cost per Unit, Total Cost, Value, Type (customizable).

Supports a hybrid format where a summary row contains the Symbol and subsequent rows are lots — the symbol is propagated to lot rows and summary rows are skipped.

Supports Excel workbooks with multiple sheets and allows selecting which sheets to read per-file via INPUT_FILES config.

If Broker is requested in output fields, the caller extracts it from the filename prefix (e.g. ML_10022025.csv -> ML).

Cleans the combined output: removes blank rows and duplicates automatically.

Appends across multiple input files instead of overwriting.

Post-processing metrics:

Per-row CAGR (using acquisition date, cost and value).

Portfolio Beta (optional — computed from provided returns CSVs).

Good logging: debug, audit messages and error messages that explain failures.

Repository layout
ProjectRoot/
│
├─ config/
│  ├─ default_settings.json      # main config (INPUT_FILES, OUTPUT_FIELDS, METRICS, etc.)
│  └─ column_mapping.json       # canonical -> [aliases]
│
├─ utils/
│  ├─ portfolio_utils.py        # extraction/normalization/IO functions
│  ├─ portfolio_metrics.py      # CAGR/Beta utility functions
│  └─ __init__.py
│
├─ run_portfolio_processor.py   # main orchestrator / CLI
├─ requirements.txt
├─ README.md
├─ logs/                        # runtime logs (created at runtime)
└─ out/                         # output CSVs (created at runtime)

Requirements

Python 3.8+ (3.10 recommended)

Recommended Python packages (see requirements.txt):

pandas
numpy
openpyxl
xlrd==1.2.0        # only if you need old-style .xls support


Note: For .xls support you may need xlrd==1.2.0. Newer xlrd releases removed .xlsx support. The code also falls back to reading mislabelled Excel files as CSV when appropriate.

Install
# create venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
python.exe -m pip install --upgrade pip
pip install --upgrade -r requirements.txt

Configuration

The main configuration file is config/default_settings.json. It controls:

INPUT_FILES — list of input files (path + optional tabs list for Excel).

OUTPUT_DIR — destination for results (default ./out).

OUTPUT_FIELDS — comma-separated canonical columns to output (defaults include Account,Symbol,...).

METRICS — enable/disable metrics and configure returns files (for Beta).

DEBUG — enable detailed logging.

config/default_settings.json example
{
  "OUTPUT_DIR": "./out",
  "DEBUG": true,
  "OUTPUT_FIELDS": "Account,Symbol,Quantity,Acquisition Date,Cost per Unit,Total Cost,Value,Type,Broker",
  "INPUT_FILES": [
    { "path": "in/ET_10022025.csv" },
    { "path": "in/ML_10022025.csv" },
    { "path": "in/CW-100502025.xls", "tabs": ["Holdings", "Summary"] }
  ],
  "METRICS": {
    "ENABLE": true,
    "COMPUTE_CAGR_PER_ROW": true,
    "RETURNS": {
      "PORTFOLIO_FILE": null,
      "BENCHMARK_FILE": null,
      "COLUMN": "return",
      "PERIODS_PER_YEAR": 252
    }
  }
}


INPUT_FILES entries may be strings or objects. If an entry is an object it must include path and may include tabs (list of sheet names) for Excel workbooks. If tabs is null or omitted, all sheets are processed.

OUTPUT_FIELDS can include Broker. If present, the caller extracts the broker code from the filename prefix before the underscore.

Column mapping

config/column_mapping.json maps canonical field names to aliases used in vendor files. Format:

{
  "Account": ["account", "acct", "description", "portfolio"],
  "Symbol": ["symbol", "ticker"],
  "Quantity": ["qty", "quantity"],
  "...": ["..."]
}


Keep this file current with the variations you encounter.

Usage

Run the orchestrator to process files, clean data, calculate metrics and write the final CSV:

python run_portfolio_processor.py            # uses config/default_settings.json


Or pass filenames on the command line (overrides config INPUT_FILES):

python run_portfolio_processor.py in/A_01012025.csv in/B_01252025.csv --debug


Options:

--config PATH       Path to settings JSON (default: config/default_settings.json)
--outdir PATH       Override output directory
--timestamp STR     Override timestamp (used in output file name)
--debug             Enable debug logging
--show-audit        Print audit log at the end


Output naming

Default output: out/portfolio-<timestamp>.csv. Timestamp is generated automatically (YYYYMMDD_HHMMSS) unless overridden.

How files are detected & processed

CSV and Excel files are scanned and tables are detected heuristically.

For hybrid/complex vendor reports:

Account Summary table (if present) yields account names.

Holdings table(s) are normalized; if a symbol appears at an aggregate row only, that symbol is propagated to the lot rows below and the aggregate row is skipped.

Description column is used to extract account/lots details where appropriate.

Excel files: if tabs provided in config for a file, only those sheets are processed. Otherwise all sheets are scanned.

Files that appear to be Excel but raise workbook errors are automatically attempted to be read as CSV.

Metrics

After extraction and cleanup, the script will:

Add Years Held and CAGR columns (if COMPUTE_CAGR_PER_ROW is enabled). CAGR is computed only when:

Acquisition Date is a valid date,

Initial (Total Cost or Quantity × Cost per Unit) > 0,

Value > 0, and

Years Held > 0.

Optionally compute a portfolio Beta if you provide CSV files containing returns: a portfolio returns CSV and a benchmark returns CSV (configured via the METRICS.RETURNS block). Beta is computed as:

beta = cov(portfolio_returns, benchmark_returns) / var(benchmark_returns)


Beta, when computed, is written as a single column Beta (same value for all rows).

Logging & audit

Logs are written to console and to the logs/ directory (created at runtime).

DEBUG mode produces verbose logs including:

the number of tables detected,

sample rows per detected table,

mapping/renaming decisions,

how summary rows and lot rows were interpreted.

--show-audit prints the developer audit trail captured during processing.

Testing locally

Put sample input files in in/ (or update config/default_settings.json).

Run with --debug to view parsing decisions:

python run_portfolio_processor.py --debug


Inspect out/portfolio-<timestamp>.csv and logs in logs/.

Contributing

Fork the repo.

Create a feature branch: git checkout -b feature/my-change.

Make your changes with tests and run them locally.

Open a pull request describing the change.

Please follow the coding conventions (PEP8) and include unit tests for parsing logic if you add new formats.

Troubleshooting

TypeError: stat: path should be string... — that means INPUT_FILES entries are not normalized to {"path":..., "tabs":...}. Update your default_settings.json entries accordingly.

Error reading Excel file ... File contains no valid workbook part — the file might be a CSV with .xls extension. The processor will attempt a CSV fallback; check logs for fallback actions.

Incomplete data — enable --debug to see how tables were detected and how column mapping was applied.

If a symbol only appears as an aggregate row and lot rows under it have blanks, the code will propagate the symbol to the lot rows. Ensure the aggregate row is present in the CSV in the format the parser expects.

License

This project is provided under the MIT License — see LICENSE for details.