from pathlib import Path
from datetime import datetime
import pandas as pd


# ============================================================
# PHASE 08 — AUTOMATED ECONOMIST REPORT
# ============================================================

PROJECT_NAME = "UK EXPENDITURE-SIDE GDP MODEL"


# ============================================================
# HELPERS
# ============================================================

def safe_value(df, column, default=None):
    """Return latest non-null value from a dataframe column."""
    if column not in df.columns:
        return default

    series = df[column].dropna()

    if series.empty:
        return default

    return series.iloc[-1]


def format_pct(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_number(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def latest_row(df):
    """Return the latest observation based on Year."""
    if df.empty:
        return pd.Series(dtype=object)

    temp = df.sort_values("Year").copy()
    return temp.iloc[-1]


def previous_row(df):
    """Return the observation immediately before the latest."""
    if len(df) < 2:
        return pd.Series(dtype=object)

    temp = df.sort_values("Year").copy()
    return temp.iloc[-2]


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def generate_executive_summary(df):
    latest = latest_row(df)

    year = latest.get("Year", "N/A")
    gdp_growth = latest.get("GDP_Growth_Pct")
    domestic_growth = latest.get("Domestic_Demand_Growth_Pct")
    investment_growth = latest.get("Investment_Growth_Pct")
    consumption_growth = latest.get("Consumption_Growth_Pct")
    nx = latest.get("NX")
    macro_regime = latest.get("Macro_Regime", "N/A")

    paragraphs = []

    if pd.notna(gdp_growth):
        if gdp_growth < 0:
            growth_assessment = (
                f"The economy recorded a contraction of "
                f"{abs(gdp_growth):.2f}% in {year}."
            )
        elif gdp_growth < 1:
            growth_assessment = (
                f"Economic growth remained weak at {gdp_growth:.2f}% "
                f"in {year}."
            )
        elif gdp_growth < 2:
            growth_assessment = (
                f"Economic activity expanded by {gdp_growth:.2f}% "
                f"in {year}, indicating moderate growth."
            )
        else:
            growth_assessment = (
                f"Economic activity expanded strongly by {gdp_growth:.2f}% "
                f"in {year}."
            )
    else:
        growth_assessment = "GDP growth could not be assessed."

    paragraphs.append(growth_assessment)

    if pd.notna(domestic_growth):
        if domestic_growth > gdp_growth:
            paragraphs.append(
                f"Domestic demand grew by {domestic_growth:.2f}%, "
                "outperforming headline GDP growth and providing a "
                "positive domestic contribution to activity."
            )
        else:
            paragraphs.append(
                f"Domestic demand grew by {domestic_growth:.2f}%, "
                "indicating comparatively subdued domestic momentum."
            )

    if pd.notna(investment_growth):
        paragraphs.append(
            f"Investment growth was {investment_growth:.2f}%, while "
            f"household consumption grew by "
            f"{consumption_growth:.2f}%."
        )

    if pd.notna(nx):
        if nx < 0:
            paragraphs.append(
                f"The external sector remained in deficit, with net exports "
                f"at {nx:.2f} in the model dataset."
            )
        else:
            paragraphs.append(
                f"The external sector recorded positive net exports of "
                f"{nx:.2f}."
            )

    paragraphs.append(
        f"The model classifies the latest macroeconomic regime as "
        f"'{macro_regime}'."
    )

    return "\n\n".join(paragraphs)


# ============================================================
# GDP AND BUSINESS CYCLE
# ============================================================

def generate_gdp_assessment(df):
    latest = latest_row(df)

    growth = latest.get("GDP_Growth_Pct")
    momentum = latest.get("GDP_Momentum", "N/A")
    regime = latest.get("GDP_Regime", "N/A")

    if pd.isna(growth):
        assessment = "GDP growth is unavailable for the latest observation."
    elif growth < 0:
        assessment = (
            f"Latest GDP growth was negative at {growth:.2f}%, "
            "indicating contraction."
        )
    elif growth < 1:
        assessment = (
            f"Latest GDP growth was {growth:.2f}%, indicating weak "
            "economic expansion."
        )
    elif growth < 2:
        assessment = (
            f"Latest GDP growth was {growth:.2f}%, indicating moderate "
            "economic expansion."
        )
    else:
        assessment = (
            f"Latest GDP growth was {growth:.2f}%, indicating relatively "
            "strong expansion."
        )

    return (
        f"{assessment} The current GDP regime is '{regime}' and "
        f"momentum is classified as '{momentum}'."
    )


# ============================================================
# EXPENDITURE COMPONENTS
# ============================================================

def generate_component_assessment(df):
    latest = latest_row(df)

    components = {
        "Consumption": "Consumption_Growth_Pct",
        "Investment": "Investment_Growth_Pct",
        "Government demand": "Government_Growth_Pct",
        "Exports": "Exports_Growth_Pct",
        "Imports": "Imports_Growth_Pct",
    }

    lines = []

    for name, column in components.items():
        value = latest.get(column)

        if pd.notna(value):
            direction = "increased" if value >= 0 else "contracted"
            lines.append(
                f"{name}: {direction} by {abs(value):.2f}%."
            )

    return " ".join(lines)


# ============================================================
# GROWTH CONTRIBUTIONS
# ============================================================

def generate_contribution_assessment(df):
    latest = latest_row(df)

    contribution_columns = {
        "Consumption": "Consumption_Growth_Contribution_Ppt",
        "Investment": "Investment_Growth_Contribution_Ppt",
        "Government demand": "Government_Growth_Contribution_Ppt",
        "Net exports": "Net_Exports_Growth_Contribution_Ppt",
    }

    contributions = []

    for name, column in contribution_columns.items():
        value = latest.get(column)

        if pd.notna(value):
            contributions.append((name, value))

    if not contributions:
        return "Growth contribution data are unavailable."

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    strongest = contributions[0]

    text = (
        f"The largest absolute contribution to GDP growth came from "
        f"{strongest[0]}, contributing {strongest[1]:.2f} percentage points."
    )

    positive = [x for x in contributions if x[1] > 0]
    negative = [x for x in contributions if x[1] < 0]

    if positive:
        text += (
            " Positive contributors included "
            + ", ".join(
                f"{name} ({value:.2f} pp)"
                for name, value in positive
            )
            + "."
        )

    if negative:
        text += (
            " Negative contributors included "
            + ", ".join(
                f"{name} ({value:.2f} pp)"
                for name, value in negative
            )
            + "."
        )

    return text


# ============================================================
# DOMESTIC DEMAND
# ============================================================

def generate_domestic_demand_assessment(df):
    latest = latest_row(df)

    growth = latest.get("Domestic_Demand_Growth_Pct")
    regime = latest.get("Domestic_Demand_Regime", "N/A")
    momentum = latest.get("Domestic_Demand_Momentum", "N/A")

    if pd.isna(growth):
        return "Domestic demand could not be assessed."

    return (
        f"Domestic demand grew by {growth:.2f}% in the latest observation. "
        f"The domestic demand regime is classified as '{regime}', with "
        f"momentum assessed as '{momentum}'."
    )


# ============================================================
# EXTERNAL SECTOR
# ============================================================

def generate_external_sector_assessment(df):
    latest = latest_row(df)

    nx = latest.get("NX")
    regime = latest.get("External_Sector_Regime", "N/A")
    coverage = latest.get("Export_Import_Coverage_Ratio")

    if pd.isna(nx):
        return "External-sector data are unavailable."

    if nx < 0:
        balance_text = (
            f"The economy recorded a net export deficit of "
            f"{abs(nx):.2f}."
        )
    else:
        balance_text = (
            f"The economy recorded positive net exports of {nx:.2f}."
        )

    if pd.notna(coverage):
        coverage_text = (
            f" Export/import coverage was {coverage:.2f}%."
        )
    else:
        coverage_text = ""

    return (
        f"{balance_text}{coverage_text} The external-sector regime is "
        f"classified as '{regime}'."
    )


# ============================================================
# FINANCIAL CONDITIONS
# ============================================================

def generate_financial_assessment(df):
    latest = latest_row(df)

    rate = latest.get("Rate")
    direction = latest.get("Interest_Rate_Direction", "N/A")
    financial_regime = latest.get("Financial_Conditions_Regime", "N/A")
    fx = latest.get("FX")
    fx_momentum = latest.get("FX_Momentum", "N/A")

    parts = []

    if pd.notna(rate):
        parts.append(f"The policy rate is {rate:.2f}%.")

    parts.append(f"Interest-rate direction is classified as '{direction}'.")

    parts.append(
        f"Financial conditions are classified as '{financial_regime}'."
    )

    if pd.notna(fx):
        parts.append(
            f"The model FX rate is {fx:.2f}, with FX momentum classified "
            f"as '{fx_momentum}'."
        )

    return " ".join(parts)


# ============================================================
# ECONOMETRIC INTERPRETATION
# ============================================================

def generate_econometric_assessment(econometric_results):
    if not econometric_results:
        return "Econometric results are unavailable."

    sample = econometric_results.get("sample_assessment", {})

    n = sample.get("observations")
    assessment = sample.get("assessment", "UNKNOWN")

    if n is not None:
        sample_text = (
            f"The econometric analysis uses {n} observations and is "
            f"classified as '{assessment}'."
        )
    else:
        sample_text = (
            "The econometric sample size could not be established."
        )

    correlations = econometric_results.get("gdp_correlations")

    correlation_text = ""

    if isinstance(correlations, pd.DataFrame) and not correlations.empty:
        try:
            corr_series = correlations.iloc[:, 0].dropna()

            if not corr_series.empty:
                strongest_variable = corr_series.abs().idxmax()
                strongest_value = corr_series.loc[strongest_variable]

                correlation_text = (
                    f" The strongest reported correlation with GDP growth "
                    f"is {strongest_variable}, at "
                    f"{strongest_value:.3f}."
                )
        except Exception:
            correlation_text = ""

    warning = (
        " These relationships should be treated as exploratory rather "
        "than causal because the sample is very small and expenditure "
        "components are mechanically linked to GDP through the national "
        "accounts identity."
    )

    return sample_text + correlation_text + warning


# ============================================================
# FORECAST ASSESSMENT
# ============================================================

def generate_forecast_assessment(forecast_results):
    if forecast_results is None:
        return "Forecast results are unavailable."

    summary = None

    if isinstance(forecast_results, dict):
        for key in [
            "scenario_summary",
            "forecast_summary",
            "summary",
        ]:
            candidate = forecast_results.get(key)

            if isinstance(candidate, pd.DataFrame):
                summary = candidate
                break

    if summary is None:
        return (
            "Forecast scenarios are available, but no scenario summary "
            "table was supplied to the report generator."
        )

    if summary.empty:
        return "Forecast scenario summary is empty."

    text_parts = []

    for _, row in summary.iterrows():
        scenario = row.get("Scenario", "N/A")
        final_growth = row.get("Final_GDP_Growth_Pct")

        if pd.notna(final_growth):
            text_parts.append(
                f"{scenario}: final projected GDP growth "
                f"{final_growth:.2f}%."
            )

    return (
        "The scenario analysis produces illustrative projections. "
        + " ".join(text_parts)
        + " These should not be interpreted as high-confidence forecasts."
    )


# ============================================================
# RISK ASSESSMENT
# ============================================================

def generate_risk_assessment(df):
    latest = latest_row(df)

    risks = []

    gdp_growth = latest.get("GDP_Growth_Pct")
    domestic_growth = latest.get("Domestic_Demand_Growth_Pct")
    investment_growth = latest.get("Investment_Growth_Pct")
    nx = latest.get("NX")
    rate_direction = latest.get("Interest_Rate_Direction")
    financial_regime = latest.get("Financial_Conditions_Regime")
    macro_regime = latest.get("Macro_Regime")

    if pd.notna(gdp_growth) and gdp_growth < 1:
        risks.append(
            ("Growth", "MEDIUM", "GDP growth is weak.")
        )

    if pd.notna(domestic_growth) and domestic_growth < 1:
        risks.append(
            ("Domestic demand", "MEDIUM",
             "Domestic demand growth is weak.")
        )

    if pd.notna(investment_growth) and investment_growth < 0:
        risks.append(
            ("Investment", "HIGH",
             "Investment is contracting.")
        )

    if pd.notna(nx) and nx < 0:
        risks.append(
            ("External sector", "MEDIUM",
             "Net exports remain negative.")
        )

    if rate_direction == "TIGHTENING":
        risks.append(
            ("Monetary policy", "MEDIUM",
             "Interest rates are moving higher.")
        )

    if financial_regime == "STRONG TIGHTENING":
        risks.append(
            ("Financial conditions", "HIGH",
             "Financial conditions are strongly restrictive.")
        )

    if macro_regime == "BROAD CONTRACTION":
        risks.append(
            ("Business cycle", "HIGH",
             "Multiple parts of the economy are contracting.")
        )

    if not risks:
        risks.append(
            ("General", "LOW",
             "No major automated risk trigger was identified.")
        )

    return pd.DataFrame(
        risks,
        columns=["Risk_Area", "Risk_Level", "Assessment"]
    )


# ============================================================
# DATA QUALITY / LIMITATIONS
# ============================================================

def generate_limitations(df):
    observations = len(df)

    return (
        f"The analysis contains {observations} observations. "
        "The current dataset is an illustrative/test dataset rather than "
        "an official ONS UK national accounts dataset. Consequently, "
        "numerical findings should be treated as demonstrations of the "
        "analytical framework rather than estimates of the actual UK "
        "economy. The expenditure identity may also contain residuals "
        "because real national accounts can contain statistical "
        "discrepancies and chain-volume measures may not be strictly "
        "additive. Econometric results are particularly sensitive to the "
        "very small sample and should not be interpreted as causal evidence."
    )


# ============================================================
# ECONOMIC ASSESSMENT TABLE
# ============================================================

def create_economic_assessment(df):
    latest = latest_row(df)

    rows = [
        ["GDP Growth", latest.get("GDP_Growth_Pct"), "%"],
        ["Domestic Demand Growth",
         latest.get("Domestic_Demand_Growth_Pct"), "%"],
        ["Consumption Growth",
         latest.get("Consumption_Growth_Pct"), "%"],
        ["Investment Growth",
         latest.get("Investment_Growth_Pct"), "%"],
        ["Government Growth",
         latest.get("Government_Growth_Pct"), "%"],
        ["Exports Growth",
         latest.get("Exports_Growth_Pct"), "%"],
        ["Imports Growth",
         latest.get("Imports_Growth_Pct"), "%"],
        ["Net Exports",
         latest.get("NX"), "model units"],
        ["Interest Rate",
         latest.get("Rate"), "%"],
        ["FX",
         latest.get("FX"), "index/rate"],
    ]

    return pd.DataFrame(
        rows,
        columns=["Indicator", "Value", "Unit"]
    )


# ============================================================
# TEXT REPORT
# ============================================================

def build_text_report(
    macro_indicators_df,
    econometric_results=None,
    forecast_results=None,
):
    latest = latest_row(macro_indicators_df)

    year = latest.get("Year", "N/A")

    sections = []

    sections.append(
        f"""
{PROJECT_NAME}
PHASE 08 — AUTOMATED ECONOMIST REPORT
============================================================

Report date: {datetime.now().strftime("%Y-%m-%d")}
Latest observation: {year}

IMPORTANT DATA NOTE
-------------------
The current project uses an illustrative/test dataset.
The numerical results below demonstrate the analytical framework
and should not be presented as official UK economic statistics.
"""
    )

    sections.append(
        f"""
1. EXECUTIVE SUMMARY
--------------------
{generate_executive_summary(macro_indicators_df)}
"""
    )

    sections.append(
        f"""
2. CURRENT ECONOMIC POSITION
----------------------------
{generate_gdp_assessment(macro_indicators_df)}
"""
    )

    sections.append(
        f"""
3. GDP GROWTH AND BUSINESS CYCLE
---------------------------------
{generate_gdp_assessment(macro_indicators_df)}
"""
    )

    sections.append(
        f"""
4. GDP GROWTH DECOMPOSITION
---------------------------
{generate_contribution_assessment(macro_indicators_df)}
"""
    )

    sections.append(
        f"""
5. EXPENDITURE COMPONENTS
-------------------------
{generate_component_assessment(macro_indicators_df)}
"""
    )

    sections.append(
        f"""
6. DOMESTIC DEMAND
------------------
{generate_domestic_demand_assessment(macro_indicators_df)}
"""
    )

    sections.append(
        f"""
7. EXTERNAL SECTOR
------------------
{generate_external_sector_assessment(macro_indicators_df)}
"""
    )

    sections.append(
        f"""
8. MONETARY AND FINANCIAL CONDITIONS
------------------------------------
{generate_financial_assessment(macro_indicators_df)}
"""
    )

    sections.append(
        f"""
9. ECONOMETRIC EVIDENCE
-----------------------
{generate_econometric_assessment(econometric_results)}
"""
    )

    sections.append(
        f"""
10. FORECAST SCENARIOS
----------------------
{generate_forecast_assessment(forecast_results)}
"""
    )

    risk_df = generate_risk_assessment(macro_indicators_df)

    risk_text = "\n".join(
        f"- {row.Risk_Area}: {row.Risk_Level} — {row.Assessment}"
        for _, row in risk_df.iterrows()
    )

    sections.append(
        f"""
11. ECONOMIC RISKS
------------------
{risk_text}
"""
    )

    sections.append(
        f"""
12. DATA QUALITY AND LIMITATIONS
--------------------------------
{generate_limitations(macro_indicators_df)}
"""
    )

    sections.append(
        """
13. ECONOMIST CONCLUSION
------------------------
The model provides a structured expenditure-side framework for
monitoring UK economic activity. The appropriate interpretation is
based on the interaction between GDP growth, domestic demand,
individual expenditure components, the external sector and financial
conditions.

The framework is designed to support repeated updates as new data
become available. Before the model is used for real economic
forecasting or policy analysis, the illustrative dataset should be
replaced with official UK national accounts and financial-market data,
with the relevant ONS definitions and methodological adjustments
applied.
"""
    )

    sections.append(
        """
============================================================
END OF AUTOMATED ECONOMIST REPORT
============================================================
"""
    )

    return "\n".join(sections)


# ============================================================
# HTML REPORT
# ============================================================

def build_html_report(
    macro_indicators_df,
    econometric_results=None,
    forecast_results=None,
):
    text_report = build_text_report(
        macro_indicators_df,
        econometric_results,
        forecast_results,
    )

    escaped = (
        text_report
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{PROJECT_NAME} — Economist Report</title>

<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background: #f5f6f8;
    color: #222;
}}

.container {{
    max-width: 1100px;
    margin: auto;
    background: white;
    padding: 40px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}}

h1 {{
    margin-bottom: 5px;
}}

.subtitle {{
    color: #666;
    margin-bottom: 30px;
}}

pre {{
    white-space: pre-wrap;
    font-family: Arial, sans-serif;
    line-height: 1.6;
    font-size: 15px;
}}

.warning {{
    padding: 15px;
    margin-bottom: 25px;
    background: #fff3cd;
    border-left: 5px solid #f0ad4e;
}}

.footer {{
    margin-top: 40px;
    color: #777;
    font-size: 12px;
}}
</style>
</head>

<body>

<div class="container">

<h1>{PROJECT_NAME}</h1>

<div class="subtitle">
Phase 08 — Automated Economist Report
</div>

<div class="warning">
<strong>Important:</strong>
This report currently uses an illustrative/test dataset and is not
an official UK economic forecast.
</div>

<pre>{escaped}</pre>

<div class="footer">
Generated automatically by the UK expenditure-side GDP model.
</div>

</div>

</body>
</html>
"""

    return html


# ============================================================
# MASTER REPORT PIPELINE
# ============================================================

def create_economist_report(
    macro_indicators_df,
    econometric_results,
    forecast_results,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_report = build_text_report(
        macro_indicators_df,
        econometric_results,
        forecast_results,
    )

    html_report = build_html_report(
        macro_indicators_df,
        econometric_results,
        forecast_results,
    )

    assessment_df = create_economic_assessment(
        macro_indicators_df
    )

    risk_df = generate_risk_assessment(
        macro_indicators_df
    )

    text_path = output_dir / "economist_report.txt"
    html_path = output_dir / "economist_report.html"
    assessment_path = output_dir / "economic_assessment.csv"
    risk_path = output_dir / "risk_assessment.csv"

    text_path.write_text(
        text_report,
        encoding="utf-8"
    )

    html_path.write_text(
        html_report,
        encoding="utf-8"
    )

    assessment_df.to_csv(
        assessment_path,
        index=False
    )

    risk_df.to_csv(
        risk_path,
        index=False
    )

    return {
        "text_report": text_path,
        "html_report": html_path,
        "economic_assessment": assessment_path,
        "risk_assessment": risk_path,
        "assessment_df": assessment_df,
        "risk_df": risk_df,
    }


# ============================================================
# REPORT OUTPUT
# ============================================================

def print_report_summary(report_outputs):
    print()
    print("=" * 80)
    print("PHASE 08 — AUTOMATED ECONOMIST REPORT")
    print("=" * 80)

    print()
    print("REPORT OUTPUTS")
    print("-" * 80)

    for key, value in report_outputs.items():
        if isinstance(value, Path):
            print(f"{key:<25}: {value}")

    print()
    print("Phase 08 completed successfully.")