# PRFS Unified — Pakistan Revenue Forecasting System

## Overview
A unified Streamlit app that combines:
- **Multi-model engine** (ARDL / ARIMAX / ElasticNet) from `app_multimodel_v2.py`
- **Dynamic 2-Step PRFS engine** (ForecastingPipeline + ScenarioEngine) from `main_macro.py`

Under a **single sidebar with ONLY 4 macro scenario sliders**:
1. Nominal GDP Growth Target (%)
2. Inflation Target (%)
3. Exchange Rate Depreciation (%)
4. Policy Rate Target (%)

No multi-model growth sliders (GDP non-agr, LSM, imports, etc.) appear anywhere.

## How It Works

### Mapping Layer (`prfs_unified/mapping.py`)
The 4 dynamic sliders are converted into a full exogenous future dataframe via deterministic mapping:
- **GDP Growth** drives all log-level bases (GDP, LSM, imports, consumption, dutiable imports) through configurable elasticities
- **Exchange Rate Depreciation** drives `log_exrate` via compound log-space growth
- **Inflation** is passed through as a level
- **Policy Rate** is passed through as a level

### Advanced Overrides
Users can optionally fine-tune mapping elasticities:
- `imports ↔ GDP elasticity` (default 1.0)
- `consumption ↔ GDP elasticity` (default 1.0)
- `LSM ↔ GDP elasticity` (default 1.0)

## Required Files
Place these files in the same directory as `app_prfs_unified.py` (or in `PRFS/`):
- `tax_prepared_data.xlsx` (or `.csv`)
- `buoyancy_estimates.xlsx` (optional, for FY2027 benchmark)
- `tax_models_bundle.pkl` (for multi-model engine)
- `tax_models_meta.json` (for multi-model engine)
- `PRFS/engine/` folder (for dynamic engine)

## Run
```bash
streamlit run PRFS_UNIFIED/app_prfs_unified.py
```

## Project Structure
```
PRFS_UNIFIED/
├── app_prfs_unified.py          # Entry point
├── README.md
└── prfs_unified/
    ├── __init__.py
    ├── data_io.py               # Data loading (xlsx/csv/pkl)
    ├── scenario_inputs.py       # Sidebar (4 sliders only)
    ├── mapping.py               # dynamic → multimodel bridge
    ├── buoyancy_benchmark.py    # FY27 buoyancy comparison
    ├── plots.py                 # Reusable Plotly charts
    ├── utils.py                 # Diagnostics & coef tables
    └── adapters/
        ├── __init__.py
        ├── multimodel_adapter.py  # Wraps app_multimodel_v2 logic
        └── dynamic_adapter.py     # Wraps PRFS engine
```
