
from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "UK EXPENDITURE-SIDE GDP MODEL"

PROJECT_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = PROJECT_DIR / "RAW_DATA"
CLEAN_DATA_DIR = PROJECT_DIR / "CLEAN_DATA"
GDP_IDENTITY_DIR = PROJECT_DIR / "GDP_IDENTITY"
INDICATORS_DIR = PROJECT_DIR / "INDICATORS"
ECONOMETRICS_DIR = PROJECT_DIR / "ECONOMETRICS"
FORECAST_DIR = PROJECT_DIR / "FORECAST"
DASHBOARD_DIR = PROJECT_DIR / "DASHBOARD"
REPORT_DIR = PROJECT_DIR / "REPORT"

RAW_DATA_FILE = RAW_DATA_DIR / "uk_gdp_expenditure.csv"


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from SRC.data_loader import (
    load_gdp_data,
    print_data_loader_report,
)

from SRC.data_quality import (
    run_quality_checks,
    print_quality_report,
)

from SRC.cleaning import (
    clean_gdp_data,
    print_cleaning_report,
)

from SRC.gdp_identity import (
    run_gdp_identity_analysis,
    print_gdp_identity_report,
)

from SRC.indicators import (
    calculate_macro_indicators,
    print_macro_indicator_report,
    run_indicator_quality_checks,
    print_indicator_quality_report,
)

from SRC.econometrics import (
    run_econometric_analysis,
    print_econometric_report,
    run_econometric_quality_checks,
    print_econometric_quality_report,
)

from SRC.forecasting import (
    run_forecasting_analysis,
    print_forecast_report,
    run_forecast_quality_checks,
    print_forecast_quality_report,
    save_forecast_outputs,
)

from SRC.dashboard import (
    create_economic_dashboard,
    print_dashboard_report,
)

from SRC.report import (
    create_economist_report,
    print_report_summary,
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [
    RAW_DATA_DIR,
    CLEAN_DATA_DIR,
    GDP_IDENTITY_DIR,
    INDICATORS_DIR,
    ECONOMETRICS_DIR,
    FORECAST_DIR,
    DASHBOARD_DIR,
    REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PROJECT HEADER
# ============================================================

print()
print("=" * 80)
print(PROJECT_NAME)
print("=" * 80)

print("PROFESSIONAL ECONOMIST VERSION")
print()

print("Project directory:")
print(f"  {PROJECT_DIR}")

print()

print("Model:")
print("  Y = C + I + G + NX")
print("  NX = X - M")

print()

print("Workflow:")
print("  RAW DATA")
print("      ↓")
print("  DATA QUALITY")
print("      ↓")
print("  DATA CLEANING")
print("      ↓")
print("  GDP IDENTITY")
print("      ↓")
print("  MACRO INDICATORS")
print("      ↓")
print("  ECONOMETRICS")
print("      ↓")
print("  FORECAST")
print("      ↓")
print("  DASHBOARD")
print("      ↓")
print("  ECONOMIST REPORT")

print("=" * 80)


# ============================================================
# PHASE 01 — DATA LOADING
# ============================================================

print()
print("=" * 80)
print("PHASE 01 — DATA LOADING")
print("=" * 80)

df_raw = load_gdp_data(
    RAW_DATA_FILE
)

print_data_loader_report(
    df_raw
)


# ============================================================
# PHASE 01 — DATA QUALITY CONTROL
# ============================================================

print()
print("=" * 80)
print("PHASE 01 — DATA QUALITY CHECKS")
print("=" * 80)

quality_results = run_quality_checks(
    df_raw
)

print_quality_report(
    quality_results
)


# Save Phase 01 dataset
phase01_file = (
    CLEAN_DATA_DIR / "phase01_gdp_data.csv"
)

df_raw.to_csv(
    phase01_file,
    index=False,
)

print()
print("Phase 01 dataset saved:")
print(f"  → {phase01_file}")


# ============================================================
# PHASE 02 — DATA CLEANING
# ============================================================

print()
print("=" * 80)
print("PHASE 02 — DATA CLEANING")
print("=" * 80)

clean_df = clean_gdp_data(
    df_raw
)

print_cleaning_report(
    clean_df
)


# Save cleaned dataset
clean_file = (
    CLEAN_DATA_DIR / "clean_gdp_data.csv"
)

clean_df.to_csv(
    clean_file,
    index=False,
)

print()
print("Clean dataset saved:")
print(f"  → {clean_file}")


# ============================================================
# PHASE 03 — GDP IDENTITY & GROWTH DECOMPOSITION
# ============================================================

print()
print("=" * 80)
print("PHASE 03 — GDP IDENTITY ANALYSIS")
print("=" * 80)

gdp_identity_df = run_gdp_identity_analysis(
    clean_df
)

print_gdp_identity_report(
    gdp_identity_df
)


# Save GDP identity analysis
gdp_identity_file = (
    CLEAN_DATA_DIR / "gdp_identity_analysis.csv"
)

gdp_identity_df.to_csv(
    gdp_identity_file,
    index=False,
)


# Save professional dataset
professional_dataset_file = (
    CLEAN_DATA_DIR /
    "uk_gdp_professional_dataset.csv"
)

gdp_identity_df.to_csv(
    professional_dataset_file,
    index=False,
)

print()
print("GDP identity analysis saved:")
print(f"  → {gdp_identity_file}")

print()
print("Professional GDP dataset saved:")
print(f"  → {professional_dataset_file}")


# ============================================================
# PHASE 04 — MACROECONOMIC INDICATORS
# ============================================================

print()
print("=" * 80)
print("PHASE 04 — MACROECONOMIC INDICATORS")
print("=" * 80)

macro_indicators_df = calculate_macro_indicators(
    gdp_identity_df
)

print_macro_indicator_report(
    macro_indicators_df
)


# Indicator quality control
indicator_quality_results = (
    run_indicator_quality_checks(
        macro_indicators_df
    )
)

print_indicator_quality_report(
    indicator_quality_results
)


# Save macro indicators
macro_indicators_file = (
    CLEAN_DATA_DIR /
    "macro_indicators.csv"
)

macro_indicators_df.to_csv(
    macro_indicators_file,
    index=False,
)

print()
print("Macro indicators dataset saved:")
print(f"  → {macro_indicators_file}")


# ============================================================
# PHASE 05 — ECONOMETRIC ANALYSIS
# ============================================================

print()
print("=" * 80)
print("PHASE 05 — ECONOMETRIC ANALYSIS")
print("=" * 80)

econometric_results = run_econometric_analysis(
    macro_indicators_df
)

print_econometric_report(
    econometric_results
)


# Econometric quality control
econometric_quality_results = (
    run_econometric_quality_checks(
        econometric_results
    )
)

print_econometric_quality_report(
    econometric_quality_results
)


# ============================================================
# SAVE ECONOMETRIC OUTPUTS
# ============================================================

# Correlation matrix
correlation_matrix = (
    econometric_results.get(
        "correlation_matrix"
    )
)

if isinstance(
    correlation_matrix,
    pd.DataFrame,
):
    correlation_matrix.to_csv(
        ECONOMETRICS_DIR /
        "correlation_matrix.csv"
    )


# GDP growth correlations
gdp_correlations = (
    econometric_results.get(
        "gdp_correlations"
    )
)

if isinstance(
    gdp_correlations,
    pd.DataFrame,
):
    gdp_correlations.to_csv(
        ECONOMETRICS_DIR /
        "gdp_growth_correlations.csv",
        index=False,
    )


# Lag relationships
lag_relationships = (
    econometric_results.get(
        "lag_relationships"
    )
)

if isinstance(
    lag_relationships,
    pd.DataFrame,
):
    lag_relationships.to_csv(
        ECONOMETRICS_DIR /
        "lag_relationships.csv",
        index=False,
    )


# Regression summary
regression_summary = (
    econometric_results.get(
        "regression_summary"
    )
)

if isinstance(
    regression_summary,
    pd.DataFrame,
):
    regression_summary.to_csv(
        ECONOMETRICS_DIR /
        "regression_summary.csv",
        index=False,
    )


# Regression coefficients
regression_coefficients = (
    econometric_results.get(
        "regression_coefficients"
    )
)

if isinstance(
    regression_coefficients,
    pd.DataFrame,
):
    regression_coefficients.to_csv(
        ECONOMETRICS_DIR /
        "regression_coefficients.csv",
        index=False,
    )


print()
print("Econometric outputs saved to:")
print(f"  → {ECONOMETRICS_DIR}")


# ============================================================
# PHASE 06 — SCENARIO FORECASTING
# ============================================================

print()
print("=" * 80)
print("PHASE 06 — SCENARIO FORECASTING")
print("=" * 80)

forecast_results = run_forecasting_analysis(
    macro_indicators_df
)

print_forecast_report(
    forecast_results
)


# Forecast quality control
forecast_quality_results = (
    run_forecast_quality_checks(
        forecast_results
    )
)

print_forecast_quality_report(
    forecast_quality_results
)


# ============================================================
# SAVE FORECAST OUTPUTS
# ============================================================

save_forecast_outputs(
    forecast_results,
    FORECAST_DIR,
    CLEAN_DATA_DIR,
)


# ============================================================
# PHASE 07 — PROFESSIONAL ECONOMIST DASHBOARD
# ============================================================

print()
print("=" * 80)
print("PHASE 07 — PROFESSIONAL ECONOMIST DASHBOARD")
print("=" * 80)

dashboard_outputs = create_economic_dashboard(
    macro_indicators_df,
    forecast_results,
    DASHBOARD_DIR,
)

print_dashboard_report(
    dashboard_outputs
)


# ============================================================
# PHASE 08 — AUTOMATED ECONOMIST REPORT
# ============================================================

print()
print("=" * 80)
print("PHASE 08 — AUTOMATED ECONOMIST REPORT")
print("=" * 80)

report_outputs = create_economist_report(
    macro_indicators_df=macro_indicators_df,
    econometric_results=econometric_results,
    forecast_results=forecast_results,
    output_dir=REPORT_DIR,
)

print_report_summary(
    report_outputs
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

print()
print("=" * 80)
print("COMPLETE PIPELINE")
print("=" * 80)

print()

print("Completed phases:")

print(
    "  ✓ Phase 01 — Data Loading & Quality Control"
)

print(
    "  ✓ Phase 02 — Data Cleaning & GDP Preparation"
)

print(
    "  ✓ Phase 03 — GDP Identity & Growth Decomposition"
)

print(
    "  ✓ Phase 04 — Macroeconomic Indicators"
)

print(
    "  ✓ Phase 05 — Econometric Analysis"
)

print(
    "  ✓ Phase 06 — Scenario Forecasting"
)

print(
    "  ✓ Phase 07 — Professional Economist Dashboard"
)

print(
    "  ✓ Phase 08 — Automated Economist Report"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

print()

print("Output directories:")

print(
    f"  RAW DATA:          {RAW_DATA_DIR}"
)

print(
    f"  CLEAN DATA:        {CLEAN_DATA_DIR}"
)

print(
    f"  GDP IDENTITY:      {GDP_IDENTITY_DIR}"
)

print(
    f"  INDICATORS:        {INDICATORS_DIR}"
)

print(
    f"  ECONOMETRICS:      {ECONOMETRICS_DIR}"
)

print(
    f"  FORECAST:          {FORECAST_DIR}"
)

print(
    f"  DASHBOARD:         {DASHBOARD_DIR}"
)

print(
    f"  REPORT:            {REPORT_DIR}"
)


# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

print()
print("=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)

print()

print("Key outputs:")

print()
print(
    "  Dashboard:"
)

print(
    f"    → {DASHBOARD_DIR / 'economic_dashboard.html'}"
)

print()
print(
    "  Economist HTML report:"
)

print(
    f"    → {REPORT_DIR / 'economist_report.html'}"
)

print()
print(
    "  Economist text report:"
)

print(
    f"    → {REPORT_DIR / 'economist_report.txt'}"
)

print()
print(
    "  Economic assessment:"
)

print(
    f"    → {REPORT_DIR / 'economic_assessment.csv'}"
)

print()
print(
    "  Risk assessment:"
)

print(
    f"    → {REPORT_DIR / 'risk_assessment.csv'}"
)

print()
print("Important:")

print(
    "  The current RAW_DATA dataset is an "
    "illustrative/test dataset."
)

print(
    "  Replace it with official UK national "
    "accounts data before using the model "
    "for substantive economic conclusions."
)

print()
print("=" * 80)
print("END OF MODEL")
print("=" * 80)

