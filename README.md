# PortfolioProcessor

**PortfolioProcessor** is a robust ETL (Extract, Transform, Load) toolkit designed to normalize, cleanse, and analyze investment portfolio data. It supports various input formats (CSV, Excel), handles "hybrid" vendor reports (summary + lots), and calculates key performance metrics like **CAGR (Compound Annual Growth Rate)** and **Beta**.

## Features

-   **Unified CLI**: Single entry point `caller.py` for all operations.
-   **Multi-Format Support**: Reads `.csv`, `.xls`, and `.xlsx`.
-   **Hybrid Parsing**: Automatically handles files containing both account summaries and holding lots.
-   **Configurable**: driven by `config/default_settings.json` and `config/column_mapping.json`.
-   **Metrics**:
    -   **CAGR**: Vectorized calculation for high performance.
    -   **Beta**: Portfolio volatility relative to a benchmark.
-   **Reporting**: Detailed per-file audit logs and summary statistics.
-   **Clean Architecture**: Separation of concerns between orchestration (`caller.py`) and logic (`utils/`).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://your-repo-url/portfolioProcessor.git
    cd portfolioProcessor
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project is controlled via `caller.py`. It supports three main subcommands: `process`, `metrics`, and `all` (default).

### 1. Process Files Only
Extracts data, normalizes columns, cleanses rows, and saves to `out/`.

**PowerShell 7+**
```powershell
# Process specific files
.venv\Scripts\python caller.py process "in/my_portfolio.xlsx" "in/old_data.csv"

# Process files defined in config/default_settings.json (INPUT_FILES)
.venv\Scripts\python caller.py process
```

**Bash**
```bash
# Process specific files
./.venv/Scripts/python caller.py process "in/my_portfolio.xlsx" "in/old_data.csv"

# Process files defined in config/default_settings.json (INPUT_FILES)
./.venv/Scripts/python caller.py process
```

### 2. Calculate Metrics Only
Appends CAGR and Beta columns to an already processed CSV file.

**PowerShell 7+**
```powershell
.venv\Scripts\python caller.py metrics "out/portfolio-20231027_120000.csv"
```

**Bash**
```bash
./.venv/Scripts/python caller.py metrics "out/portfolio-20231027_120000.csv"
```

### 3. Full Workflow (Process + Metrics)
Runs extraction followed immediately by metric calculation. This is the default behavior if no subcommand is specified.

**PowerShell 7+**
```powershell
# Explicit command
.venv\Scripts\python caller.py all "in/my_portfolio.xlsx"

# Default (uses config inputs)
.venv\Scripts\python caller.py
```

**Bash**
```bash
# Explicit command
./.venv/Scripts/python caller.py all "in/my_portfolio.xlsx"

# Default (uses config inputs)
./.venv/Scripts/python caller.py
```

### Common Flags

-   `--outdir "my_output"`: Specify custom output directory.
-   `--timestamp "2023_Q3"`: Force a specific timestamp suffix for the output file.
-   `--debug`: Enable verbose debug logging in console.
-   `--show-audit`: Print the audit log at the end of execution.

## Configuration

-   **`config/default_settings.json`**:
    -   `INPUT_FILES`: List of default files to process.
    -   `OUTPUT_FIELDS`: List of columns to include in the final CSV.
    -   `METRICS`: Toggle CAGR/Beta and configure benchmark files.
-   **`config/column_mapping.json`**: Maps vendor-specific column names (e.g., "Current Val") to canonical names (e.g., "Value").

## Running Tests

The project uses `pytest` for testing.

```bash
# Run all tests
pytest

# Run verbose
pytest -v
```

## Structure

```
portfolioProcessor/
├── caller.py               # Main CLI entry point
├── utils/
│   ├── portfolio_utils.py  # Core ETL logic (parsing, cleaning)
│   ├── portfolio_metrics.py# Financial math (CAGR, Beta)
│   └── reporting_utils.py  # Audit and summary reporting
├── config/                 # Configuration JSONs
├── tests/                  # Unit and integration tests
├── in/                     # Default input directory
└── out/                    # Default output directory
```