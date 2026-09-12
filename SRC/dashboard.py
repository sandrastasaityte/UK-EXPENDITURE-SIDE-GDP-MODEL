from pathlib import Path

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True

except ImportError:
    PLOTLY_AVAILABLE = False


# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

DASHBOARD_TITLE = "UK EXPENDITURE-SIDE GDP MODEL"
DASHBOARD_SUBTITLE = "Professional Economist Dashboard"

MATERIAL_DISCREPANCY_THRESHOLD = 1.0
MINOR_DISCREPANCY_THRESHOLD = 0.5


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dashboard_inputs(
    macro_df: pd.DataFrame,
    forecast_results: dict,
):
    """
    Validate datasets required for dashboard construction.
    """

    required_macro = [
        "Year",
        "Y",
        "C",
        "I",
        "G",
        "X",
        "M",
        "NX",
        "Domestic_Demand",
        "GDP_Growth_Pct",
        "Consumption_Growth_Pct",
        "Investment_Growth_Pct",
        "Government_Growth_Pct",
        "Exports_Growth_Pct",
        "Imports_Growth_Pct",
        "Domestic_Demand_Growth_Pct",
        "Rate",
        "FX",
    ]

    missing = [
        col
        for col in required_macro
        if col not in macro_df.columns
    ]

    if missing:
        raise ValueError(
            "Dashboard is missing required macroeconomic columns: "
            + ", ".join(missing)
        )

    if not isinstance(forecast_results, dict):
        raise TypeError(
            "forecast_results must be a dictionary."
        )

    return True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _latest_value(df, column):

    if column not in df.columns:
        return np.nan

    series = df[column].dropna()

    if series.empty:
        return np.nan

    return series.iloc[-1]


def _previous_value(df, column):

    if column not in df.columns:
        return np.nan

    series = df[column].dropna()

    if len(series) < 2:
        return np.nan

    return series.iloc[-2]


def _format_pct(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:.2f}%"


def _format_number(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:,.2f}"


def _risk_label(score):

    if score >= 3:
        return "HIGH"

    elif score == 2:
        return "MEDIUM"

    else:
        return "LOW"


# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

def create_dashboard_summary(
    macro_df: pd.DataFrame,
    forecast_results: dict,
):

    latest = macro_df.iloc[-1]

    summary = {
        "Latest_Year": int(latest["Year"]),

        "GDP": latest["Y"],

        "GDP_Growth_Pct": latest["GDP_Growth_Pct"],

        "GDP_Regime": latest.get(
            "GDP_Regime",
            "N/A",
        ),

        "GDP_Momentum": latest.get(
            "GDP_Momentum",
            "N/A",
        ),

        "Consumption_Growth_Pct":
            latest["Consumption_Growth_Pct"],

        "Investment_Growth_Pct":
            latest["Investment_Growth_Pct"],

        "Government_Growth_Pct":
            latest["Government_Growth_Pct"],

        "Domestic_Demand_Growth_Pct":
            latest["Domestic_Demand_Growth_Pct"],

        "Exports_Growth_Pct":
            latest["Exports_Growth_Pct"],

        "Imports_Growth_Pct":
            latest["Imports_Growth_Pct"],

        "Interest_Rate":
            latest["Rate"],

        "FX":
            latest["FX"],

        "Macro_Regime":
            latest.get(
                "Macro_Regime",
                "N/A",
            ),
    }

    return pd.DataFrame([summary])


# =============================================================================
# RISK DASHBOARD
# =============================================================================

def create_risk_dashboard(
    macro_df: pd.DataFrame,
):

    latest = macro_df.iloc[-1]

    risks = []

    # -------------------------------------------------------------------------
    # GDP RISK
    # -------------------------------------------------------------------------

    gdp_growth = latest["GDP_Growth_Pct"]

    if gdp_growth < 0:
        gdp_risk = 3

    elif gdp_growth < 1:
        gdp_risk = 2

    else:
        gdp_risk = 1

    risks.append(
        {
            "Risk_Category": "GDP Growth",
            "Score": gdp_risk,
            "Risk_Level": _risk_label(gdp_risk),
            "Indicator": gdp_growth,
            "Unit": "%",
            "Interpretation": (
                "Negative GDP growth"
                if gdp_growth < 0
                else "Weak GDP growth"
                if gdp_growth < 1
                else "Positive GDP growth"
            ),
        }
    )

    # -------------------------------------------------------------------------
    # DOMESTIC DEMAND
    # -------------------------------------------------------------------------

    demand_growth = latest[
        "Domestic_Demand_Growth_Pct"
    ]

    if demand_growth < 0:
        demand_risk = 3

    elif demand_growth < 1:
        demand_risk = 2

    else:
        demand_risk = 1

    risks.append(
        {
            "Risk_Category": "Domestic Demand",
            "Score": demand_risk,
            "Risk_Level": _risk_label(demand_risk),
            "Indicator": demand_growth,
            "Unit": "%",
            "Interpretation": (
                "Domestic demand contracting"
                if demand_growth < 0
                else "Weak domestic demand"
                if demand_growth < 1
                else "Domestic demand expanding"
            ),
        }
    )

    # -------------------------------------------------------------------------
    # INVESTMENT
    # -------------------------------------------------------------------------

    investment_growth = latest[
        "Investment_Growth_Pct"
    ]

    if investment_growth < 0:
        investment_risk = 3

    elif investment_growth < 1:
        investment_risk = 2

    else:
        investment_risk = 1

    risks.append(
        {
            "Risk_Category": "Investment",
            "Score": investment_risk,
            "Risk_Level": _risk_label(investment_risk),
            "Indicator": investment_growth,
            "Unit": "%",
            "Interpretation": (
                "Investment contracting"
                if investment_growth < 0
                else "Weak investment"
                if investment_growth < 1
                else "Investment expanding"
            ),
        }
    )

    # -------------------------------------------------------------------------
    # MONETARY POLICY
    # -------------------------------------------------------------------------

    rate = latest["Rate"]

    if rate >= 5:
        rate_risk = 3

    elif rate >= 3:
        rate_risk = 2

    else:
        rate_risk = 1

    risks.append(
        {
            "Risk_Category": "Interest Rates",
            "Score": rate_risk,
            "Risk_Level": _risk_label(rate_risk),
            "Indicator": rate,
            "Unit": "%",
            "Interpretation": (
                "Restrictive interest-rate environment"
                if rate >= 3
                else "Moderate interest-rate environment"
            ),
        }
    )

    # -------------------------------------------------------------------------
    # FX
    # -------------------------------------------------------------------------

    fx_change = latest.get(
        "FX_Change_Pct",
        np.nan,
    )

    if pd.isna(fx_change):

        fx_risk = 1
        fx_interpretation = "FX change unavailable"

    elif abs(fx_change) >= 5:

        fx_risk = 3
        fx_interpretation = "High FX volatility"

    elif abs(fx_change) >= 2:

        fx_risk = 2
        fx_interpretation = "Moderate FX movement"

    else:

        fx_risk = 1
        fx_interpretation = "Limited FX movement"

    risks.append(
        {
            "Risk_Category": "FX",
            "Score": fx_risk,
            "Risk_Level": _risk_label(fx_risk),
            "Indicator": fx_change,
            "Unit": "%",
            "Interpretation": fx_interpretation,
        }
    )

    # -------------------------------------------------------------------------
    # EXTERNAL SECTOR
    # -------------------------------------------------------------------------

    nx = latest["NX"]

    if nx < -0.05 * latest["Y"]:

        external_risk = 3

    elif nx < 0:

        external_risk = 2

    else:

        external_risk = 1

    risks.append(
        {
            "Risk_Category": "External Sector",
            "Score": external_risk,
            "Risk_Level": _risk_label(external_risk),
            "Indicator": nx,
            "Unit": "GDP units",
            "Interpretation": (
                "Large net export deficit"
                if nx < -0.05 * latest["Y"]
                else "Net export deficit"
                if nx < 0
                else "Net export surplus"
            ),
        }
    )

    return pd.DataFrame(risks)


# =============================================================================
# GDP IDENTITY DIAGNOSTICS
# =============================================================================

def create_gdp_identity_dashboard(
    macro_df: pd.DataFrame,
):

    df = macro_df.copy()

    df["GDP_Expenditure_Calculated"] = (
        df["C"]
        + df["I"]
        + df["G"]
        + df["X"]
        - df["M"]
    )

    df["GDP_Identity_Residual_Dashboard"] = (
        df["GDP_Expenditure_Calculated"]
        - df["Y"]
    )

    df["GDP_Identity_Error_Pct_Dashboard"] = (
        df["GDP_Identity_Residual_Dashboard"]
        / df["Y"]
        * 100
    )

    def classify(error):

        absolute_error = abs(error)

        if absolute_error < MINOR_DISCREPANCY_THRESHOLD:
            return "PASS"

        if absolute_error < MATERIAL_DISCREPANCY_THRESHOLD:
            return "MINOR DISCREPANCY"

        return "MATERIAL DISCREPANCY"

    df["GDP_Identity_Status_Dashboard"] = (
        df[
            "GDP_Identity_Error_Pct_Dashboard"
        ].apply(classify)
    )

    return df


# =============================================================================
# FORECAST EXTRACTION
# =============================================================================

def _validate_forecast_dataframe(
    forecast_df: pd.DataFrame,
):
    """
    Validate the professional forecast dataset.

    The forecasting phase produces:
        Year
        Scenario
        GDP_Expenditure
        GDP_Growth_Pct
        C
        I
        G
        X
        M
        NX
    """

    required_columns = [
        "Year",
        "Scenario",
        "GDP_Expenditure",
        "GDP_Growth_Pct",
        "C",
        "I",
        "G",
        "X",
        "M",
        "NX",
    ]

    missing = [
        column
        for column in required_columns
        if column not in forecast_df.columns
    ]

    if missing:

        raise ValueError(
            "Forecast dataset is missing required columns: "
            + ", ".join(missing)
        )

    if forecast_df.empty:

        raise ValueError(
            "Forecast dataset is empty."
        )

    return forecast_df.copy()


def extract_forecast_dataframe(
    forecast_results: dict,
    fallback_path=None,
):
    """
    Locate the forecast dataframe.

    Priority:

    1. DataFrame contained inside forecast_results.
    2. forecast_results['forecast_results']
    3. forecast_results['forecast_dataset']
    4. forecast_results['forecasts']
    5. forecast_results['data']
    6. CLEAN_DATA/forecast_dataset.csv fallback.

    This makes the dashboard independent of the exact
    internal structure returned by the forecasting module.
    """

    possible_keys = [
        "forecast_results",
        "forecast_dataset",
        "forecasts",
        "data",
    ]

    # -------------------------------------------------------------------------
    # SEARCH DICTIONARY
    # -------------------------------------------------------------------------

    if isinstance(forecast_results, dict):

        for key in possible_keys:

            value = forecast_results.get(key)

            if isinstance(value, pd.DataFrame):

                if "Year" in value.columns:

                    return _validate_forecast_dataframe(
                        value
                    )

        # ---------------------------------------------------------------------
        # SEARCH ALL DICTIONARY VALUES
        # ---------------------------------------------------------------------

        for value in forecast_results.values():

            if isinstance(value, pd.DataFrame):

                if "Year" in value.columns:

                    try:

                        return _validate_forecast_dataframe(
                            value
                        )

                    except ValueError:

                        continue

    # -------------------------------------------------------------------------
    # FALLBACK CSV
    # -------------------------------------------------------------------------

    if fallback_path is not None:

        fallback_path = Path(
            fallback_path
        )

        if fallback_path.exists():

            forecast_df = pd.read_csv(
                fallback_path
            )

            return _validate_forecast_dataframe(
                forecast_df
            )

    raise ValueError(
        "Could not locate forecast dataframe in "
        "forecast_results and no valid fallback "
        "forecast_dataset.csv was found."
    )


# =============================================================================
# CHART 1 — GDP GROWTH
# =============================================================================

def create_gdp_growth_chart(
    macro_df: pd.DataFrame,
    output_path: Path,
):

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=macro_df["Year"],
            y=macro_df["GDP_Growth_Pct"],
            name="GDP growth",
            hovertemplate=(
                "Year: %{x}<br>"
                "GDP growth: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_width=1,
        line_dash="solid",
    )

    fig.update_layout(
        title="UK GDP Growth",
        xaxis_title="Year",
        yaxis_title="Annual GDP growth (%)",
        template="plotly_white",
        height=450,
    )

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )


# =============================================================================
# CHART 2 — GDP COMPONENTS
# =============================================================================

def create_gdp_components_chart(
    macro_df: pd.DataFrame,
    output_path: Path,
):

    fig = go.Figure()

    components = {
        "Consumption": "Consumption_Growth_Pct",
        "Investment": "Investment_Growth_Pct",
        "Government": "Government_Growth_Pct",
        "Exports": "Exports_Growth_Pct",
        "Imports": "Imports_Growth_Pct",
    }

    for label, column in components.items():

        fig.add_trace(
            go.Scatter(
                x=macro_df["Year"],
                y=macro_df[column],
                mode="lines+markers",
                name=label,
            )
        )

    fig.add_hline(
        y=0,
        line_width=1,
    )

    fig.update_layout(
        title="Expenditure Component Growth",
        xaxis_title="Year",
        yaxis_title="Growth (%)",
        template="plotly_white",
        height=450,
    )

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )


# =============================================================================
# CHART 3 — DOMESTIC DEMAND
# =============================================================================

def create_domestic_demand_chart(
    macro_df: pd.DataFrame,
    output_path: Path,
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=macro_df["Year"],
            y=macro_df[
                "Domestic_Demand_Growth_Pct"
            ],
            mode="lines+markers",
            name="Domestic demand",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=macro_df["Year"],
            y=macro_df[
                "GDP_Growth_Pct"
            ],
            mode="lines+markers",
            name="GDP",
        )
    )

    fig.add_hline(
        y=0,
        line_width=1,
    )

    fig.update_layout(
        title="GDP vs Domestic Demand",
        xaxis_title="Year",
        yaxis_title="Growth (%)",
        template="plotly_white",
        height=450,
    )

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )


# =============================================================================
# CHART 4 — EXTERNAL SECTOR
# =============================================================================

def create_external_sector_chart(
    macro_df: pd.DataFrame,
    output_path: Path,
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=macro_df["Year"],
            y=macro_df["Exports_Growth_Pct"],
            mode="lines+markers",
            name="Exports",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=macro_df["Year"],
            y=macro_df["Imports_Growth_Pct"],
            mode="lines+markers",
            name="Imports",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=macro_df["Year"],
            y=macro_df["NX"],
            mode="lines+markers",
            name="Net exports",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="External Sector",
        xaxis_title="Year",
        yaxis_title="Growth (%)",
        yaxis2=dict(
            title="Net exports",
            overlaying="y",
            side="right",
        ),
        template="plotly_white",
        height=450,
    )

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )


# =============================================================================
# CHART 5 — FINANCIAL CONDITIONS
# =============================================================================

def create_financial_conditions_chart(
    macro_df: pd.DataFrame,
    output_path: Path,
):

    fig = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True
                }
            ]
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=macro_df["Year"],
            y=macro_df["Rate"],
            mode="lines+markers",
            name="Interest rate",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=macro_df["Year"],
            y=macro_df["FX"],
            mode="lines+markers",
            name="FX",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Financial Conditions",
        xaxis_title="Year",
        template="plotly_white",
        height=450,
    )

    fig.update_yaxes(
        title_text="Interest rate (%)",
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text="FX",
        secondary_y=True,
    )

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )


# =============================================================================
# CHART 6 — FORECAST SCENARIOS
# =============================================================================

def create_forecast_chart(
    forecast_df: pd.DataFrame,
    output_path: Path,
):
    """
    Create forecast scenario chart.

    IMPORTANT:
    The forecasting module uses GDP_Expenditure,
    not GDP.
    """

    forecast_df = _validate_forecast_dataframe(
        forecast_df
    )

    fig = go.Figure()

    for scenario in forecast_df[
        "Scenario"
    ].dropna().unique():

        scenario_df = forecast_df[
            forecast_df["Scenario"] == scenario
        ].copy()

        scenario_df = scenario_df.sort_values(
            "Year"
        )

        fig.add_trace(
            go.Scatter(
                x=scenario_df["Year"],
                y=scenario_df[
                    "GDP_Expenditure"
                ],
                mode="lines+markers",
                name=scenario,
                hovertemplate=(
                    "Year: %{x}<br>"
                    "Scenario: "
                    + str(scenario)
                    + "<br>"
                    "GDP expenditure: %{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="GDP Forecast Scenarios",
        xaxis_title="Year",
        yaxis_title="GDP expenditure",
        template="plotly_white",
        height=450,
    )

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )


# =============================================================================
# CREATE COMPLETE DASHBOARD
# =============================================================================

def create_economic_dashboard(
    macro_df: pd.DataFrame,
    forecast_results: dict,
    output_dir: Path,
):
    """
    Create the complete Phase 07 dashboard.
    """

    if not PLOTLY_AVAILABLE:

        raise ImportError(
            "Plotly is required for Phase 07. "
            "Install it with: pip install plotly"
        )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    charts_dir = (
        output_dir / "charts"
    )

    charts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_dashboard_inputs(
        macro_df,
        forecast_results,
    )

    # =========================================================================
    # DATASETS
    # =========================================================================

    summary_df = create_dashboard_summary(
        macro_df,
        forecast_results,
    )

    risk_df = create_risk_dashboard(
        macro_df,
    )

    identity_df = create_gdp_identity_dashboard(
        macro_df,
    )

    # =========================================================================
    # FORECAST DATA
    # =========================================================================
    #
    # IMPORTANT FIX:
    #
    # The forecasting module already saves:
    #
    # CLEAN_DATA/forecast_dataset.csv
    #
    # We therefore use that file as a robust fallback when the forecast
    # dataframe is not embedded inside forecast_results.
    #
    # =========================================================================

    forecast_file = (
        output_dir.parent
        / "CLEAN_DATA"
        / "forecast_dataset.csv"
    )

    forecast_df = extract_forecast_dataframe(
        forecast_results,
        fallback_path=forecast_file,
    )

    # =========================================================================
    # SAVE DATASETS
    # =========================================================================

    summary_file = (
        output_dir
        / "dashboard_summary.csv"
    )

    risk_file = (
        output_dir
        / "risk_dashboard.csv"
    )

    identity_file = (
        output_dir
        / "gdp_identity_dashboard.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    risk_df.to_csv(
        risk_file,
        index=False,
    )

    identity_df.to_csv(
        identity_file,
        index=False,
    )

    # =========================================================================
    # CREATE CHARTS
    # =========================================================================

    chart_files = {}

    # -------------------------------------------------------------------------
    # GDP GROWTH
    # -------------------------------------------------------------------------

    chart_files["GDP Growth"] = (
        charts_dir
        / "gdp_growth.html"
    )

    create_gdp_growth_chart(
        macro_df,
        chart_files["GDP Growth"],
    )

    # -------------------------------------------------------------------------
    # GDP COMPONENTS
    # -------------------------------------------------------------------------

    chart_files["GDP Components"] = (
        charts_dir
        / "gdp_components.html"
    )

    create_gdp_components_chart(
        macro_df,
        chart_files["GDP Components"],
    )

    # -------------------------------------------------------------------------
    # DOMESTIC DEMAND
    # -------------------------------------------------------------------------

    chart_files["Domestic Demand"] = (
        charts_dir
        / "domestic_demand.html"
    )

    create_domestic_demand_chart(
        macro_df,
        chart_files["Domestic Demand"],
    )

    # -------------------------------------------------------------------------
    # EXTERNAL SECTOR
    # -------------------------------------------------------------------------

    chart_files["External Sector"] = (
        charts_dir
        / "external_sector.html"
    )

    create_external_sector_chart(
        macro_df,
        chart_files["External Sector"],
    )

    # -------------------------------------------------------------------------
    # FINANCIAL CONDITIONS
    # -------------------------------------------------------------------------

    chart_files["Financial Conditions"] = (
        charts_dir
        / "financial_conditions.html"
    )

    create_financial_conditions_chart(
        macro_df,
        chart_files["Financial Conditions"],
    )

    # -------------------------------------------------------------------------
    # FORECAST
    # -------------------------------------------------------------------------

    chart_files["Forecast"] = (
        charts_dir
        / "forecast_scenarios.html"
    )

    create_forecast_chart(
        forecast_df,
        chart_files["Forecast"],
    )

    # =========================================================================
    # EXECUTIVE HTML DASHBOARD
    # =========================================================================

    latest = macro_df.iloc[-1]

    identity_latest = identity_df.iloc[-1]

    risk_high = (
        risk_df["Risk_Level"] == "HIGH"
    ).sum()

    risk_medium = (
        risk_df["Risk_Level"] == "MEDIUM"
    ).sum()

    dashboard_file = (
        output_dir
        / "economic_dashboard.html"
    )

    html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
{DASHBOARD_TITLE}
</title>

<style>

body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 0;
    background: #f4f6f8;
    color: #222;
}}

.header {{
    background: #1f2937;
    color: white;
    padding: 30px 40px;
}}

.header h1 {{
    margin: 0;
    font-size: 30px;
}}

.header p {{
    margin-top: 8px;
    color: #d1d5db;
}}

.container {{
    padding: 25px 40px;
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(210px, 1fr));
    gap: 18px;
    margin-bottom: 25px;
}}

.card {{
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow:
        0 2px 7px rgba(0,0,0,0.08);
}}

.card-title {{
    color: #6b7280;
    font-size: 13px;
    text-transform: uppercase;
    margin-bottom: 8px;
}}

.card-value {{
    font-size: 25px;
    font-weight: bold;
}}

.section {{
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 25px;
    box-shadow:
        0 2px 7px rgba(0,0,0,0.08);
}}

.section h2 {{
    margin-top: 0;
}}

iframe {{
    width: 100%;
    height: 500px;
    border: none;
}}

.warning {{
    background: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 15px;
    margin-top: 15px;
}}

.info {{
    background: #eff6ff;
    border-left: 5px solid #3b82f6;
    padding: 15px;
    margin-top: 15px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th, td {{
    padding: 10px;
    border-bottom: 1px solid #e5e7eb;
    text-align: left;
}}

th {{
    background: #f9fafb;
}}

.footer {{
    color: #6b7280;
    font-size: 12px;
    padding: 20px 40px;
}}

</style>

</head>

<body>

<div class="header">

<h1>
{DASHBOARD_TITLE}
</h1>

<p>
{DASHBOARD_SUBTITLE}
</p>

<p>
Latest observation: {int(latest["Year"])}
</p>

</div>

<div class="container">

<!-- EXECUTIVE SUMMARY -->

<div class="grid">

<div class="card">

<div class="card-title">
GDP
</div>

<div class="card-value">
{latest["Y"]:.2f}
</div>

</div>


<div class="card">

<div class="card-title">
GDP Growth
</div>

<div class="card-value">
{latest["GDP_Growth_Pct"]:.2f}%
</div>

</div>


<div class="card">

<div class="card-title">
GDP Regime
</div>

<div class="card-value">
{latest.get("GDP_Regime", "N/A")}
</div>

</div>


<div class="card">

<div class="card-title">
Domestic Demand
</div>

<div class="card-value">
{latest["Domestic_Demand_Growth_Pct"]:.2f}%
</div>

</div>


<div class="card">

<div class="card-title">
Investment
</div>

<div class="card-value">
{latest["Investment_Growth_Pct"]:.2f}%
</div>

</div>


<div class="card">

<div class="card-title">
Interest Rate
</div>

<div class="card-value">
{latest["Rate"]:.2f}%
</div>

</div>


<div class="card">

<div class="card-title">
FX
</div>

<div class="card-value">
{latest["FX"]:.2f}
</div>

</div>


<div class="card">

<div class="card-title">
Macro Regime
</div>

<div class="card-value">
{latest.get("Macro_Regime", "N/A")}
</div>

</div>

</div>


<!-- GDP -->

<div class="section">

<h2>
GDP & Business Cycle
</h2>

<iframe
src="charts/gdp_growth.html">
</iframe>

</div>


<!-- COMPONENTS -->

<div class="section">

<h2>
GDP Expenditure Components
</h2>

<iframe
src="charts/gdp_components.html">
</iframe>

</div>


<!-- DOMESTIC DEMAND -->

<div class="section">

<h2>
Domestic Demand
</h2>

<iframe
src="charts/domestic_demand.html">
</iframe>

</div>


<!-- EXTERNAL -->

<div class="section">

<h2>
External Sector
</h2>

<iframe
src="charts/external_sector.html">
</iframe>

</div>


<!-- FINANCIAL -->

<div class="section">

<h2>
Financial Conditions
</h2>

<iframe
src="charts/financial_conditions.html">
</iframe>

</div>


<!-- FORECAST -->

<div class="section">

<h2>
Forecast Scenarios
</h2>

<iframe
src="charts/forecast_scenarios.html">
</iframe>

</div>


<!-- RISK DASHBOARD -->

<div class="section">

<h2>
Economic Risk Dashboard
</h2>

<p>
High-risk indicators:
<b>{risk_high}</b>
</p>

<p>
Medium-risk indicators:
<b>{risk_medium}</b>
</p>

<table>

<tr>

<th>
Risk Category
</th>

<th>
Level
</th>

<th>
Indicator
</th>

<th>
Interpretation
</th>

</tr>
"""

    for _, row in risk_df.iterrows():

        indicator = row["Indicator"]

        if pd.isna(indicator):

            indicator_text = "N/A"

        else:

            indicator_text = (
                f"{indicator:.2f} {row['Unit']}"
            )

        html += f"""
<tr>

<td>
{row["Risk_Category"]}
</td>

<td>
<b>
{row["Risk_Level"]}
</b>
</td>

<td>
{indicator_text}
</td>

<td>
{row["Interpretation"]}
</td>

</tr>
"""

    html += f"""

</table>

</div>


<!-- GDP IDENTITY -->

<div class="section">

<h2>
GDP Identity Reconciliation
</h2>

<p>
Observed GDP:
<b>
{identity_latest["Y"]:.2f}
</b>
</p>

<p>
Calculated expenditure GDP:
<b>
{identity_latest["GDP_Expenditure_Calculated"]:.2f}
</b>
</p>

<p>
Residual:
<b>
{identity_latest["GDP_Identity_Residual_Dashboard"]:.2f}
</b>
</p>

<p>
Identity error:
<b>
{identity_latest["GDP_Identity_Error_Pct_Dashboard"]:.2f}%
</b>
</p>

<p>
Status:
<b>
{identity_latest["GDP_Identity_Status_Dashboard"]}
</b>
</p>

<div class="warning">

<b>Methodological note:</b>

The current dataset is an illustrative/test dataset.
The expenditure identity does not reconcile perfectly in
all periods. The latest observation contains a material
discrepancy.

Official UK national accounts should be used for
production analysis.

</div>

</div>


<!-- METHODOLOGY -->

<div class="section">

<h2>
Data & Methodology
</h2>

<div class="info">

<ul>

<li>
Historical observations:
<b>{len(macro_df)}</b>
annual observations
</li>

<li>
Dataset status:
<b>Illustrative / Test Dataset</b>
</li>

<li>
Econometric sample:
<b>Very Small</b>
</li>

<li>
Forecast type:
<b>Scenario-based</b>
</li>

<li>
Forecast horizon:
<b>2026–2028</b>
</li>

<li>
Econometric results should be interpreted as
<b>exploratory associations</b>, not causal estimates.
</li>

<li>
Forecasts are illustrative scenarios rather than
official statistical forecasts.
</li>

</ul>

</div>

</div>

</div>

<div class="footer">

UK Expenditure-Side GDP Model —
Professional Economist Version

</div>

</body>

</html>
"""

    with open(
        dashboard_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(html)

    return {
        "dashboard": dashboard_file,
        "summary": summary_file,
        "risk_dashboard": risk_file,
        "identity_dashboard": identity_file,
        "charts": chart_files,
    }


# =============================================================================
# REPORT
# =============================================================================

def print_dashboard_report(
    dashboard_outputs: dict,
):

    print()

    print("=" * 80)
    print("PHASE 07 — PROFESSIONAL ECONOMIST DASHBOARD")
    print("=" * 80)

    print()

    print("✓ Executive summary created")

    print("✓ GDP growth dashboard created")

    print("✓ GDP expenditure component dashboard created")

    print("✓ Domestic demand dashboard created")

    print("✓ External sector dashboard created")

    print("✓ Financial conditions dashboard created")

    print("✓ Forecast scenario dashboard created")

    print("✓ Economic risk dashboard created")

    print("✓ GDP identity reconciliation created")

    print()

    print("Dashboard:")

    print(
        f"  → {dashboard_outputs['dashboard']}"
    )

    print()

    print("Summary:")

    print(
        f"  → {dashboard_outputs['summary']}"
    )

    print()

    print("Risk dashboard:")

    print(
        f"  → {dashboard_outputs['risk_dashboard']}"
    )

    print()

    print("GDP identity:")

    print(
        f"  → {dashboard_outputs['identity_dashboard']}"
    )

    print()

    print("Charts:")

    for name, path in dashboard_outputs[
        "charts"
    ].items():

        print(
            f"  → {name}: {path}"
        )

    print()

    print("=" * 80)