from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 06 — FORECASTING & ECONOMIC SCENARIOS
# UK EXPENDITURE-SIDE GDP MODEL
# =============================================================================

FORECAST_HORIZON = 3

COMPONENTS = [
    "C",
    "I",
    "G",
    "X",
    "M",
]


# =============================================================================
# SAMPLE SIZE WARNING
# =============================================================================

def assess_forecast_sample(df):
    """
    Assess whether the historical dataset is large enough for forecasting.
    """

    observations = len(df)

    if observations < 10:
        assessment = "INSUFFICIENT SAMPLE"
    elif observations < 20:
        assessment = "VERY SMALL SAMPLE"
    elif observations < 30:
        assessment = "SMALL SAMPLE"
    else:
        assessment = "ADEQUATE SAMPLE"

    return {
        "observations": observations,
        "assessment": assessment,
    }


# =============================================================================
# RECENT GROWTH
# =============================================================================

def calculate_recent_growth(df, window=3):
    """
    Calculate average recent growth for GDP expenditure components.
    """

    result = {}

    for component in ["Y"] + COMPONENTS:

        growth_column = f"{component}_Growth_Pct"

        if growth_column in df.columns:

            values = (
                pd.to_numeric(
                    df[growth_column],
                    errors="coerce",
                )
                .dropna()
            )

            if len(values) == 0:

                result[component] = 0.0

            else:

                recent_values = values.tail(window)

                result[component] = float(
                    recent_values.mean()
                )

        else:

            growth = (
                df[component]
                .pct_change()
                .mul(100)
                .dropna()
            )

            result[component] = float(
                growth.tail(window).mean()
            )

    return result


# =============================================================================
# SCENARIO ASSUMPTIONS
# =============================================================================

def create_scenario_assumptions(recent_growth):
    """
    Create baseline, optimistic and downside assumptions.

    The assumptions are deliberately simple because the historical sample
    contains only a small number of annual observations.
    """

    baseline = {}
    optimistic = {}
    downside = {}

    for component in ["Y"] + COMPONENTS:

        recent = recent_growth.get(
            component,
            0.0,
        )

        # Baseline:
        # recent average growth.
        baseline[component] = recent

        # Optimistic:
        # 0.75 percentage points above baseline.
        optimistic[component] = recent + 0.75

        # Downside:
        # 1.00 percentage point below baseline.
        downside[component] = recent - 1.00

    return {
        "BASELINE": baseline,
        "OPTIMISTIC": optimistic,
        "DOWNSIDE": downside,
    }


# =============================================================================
# COMPONENT FORECAST
# =============================================================================

def forecast_component(
    latest_value,
    growth_rate,
    periods,
):
    """
    Forecast a component using compound growth.
    """

    forecasts = []

    value = float(latest_value)

    for _ in range(periods):

        value = value * (
            1 + growth_rate / 100
        )

        forecasts.append(value)

    return forecasts


# =============================================================================
# SCENARIO FORECAST
# =============================================================================

def generate_scenario_forecast(
    df,
    scenario_name,
    assumptions,
    horizon=FORECAST_HORIZON,
):
    """
    Generate expenditure-side GDP forecasts.
    """

    latest = df.iloc[-1]

    forecast_years = [
        int(latest["Year"]) + i
        for i in range(1, horizon + 1)
    ]

    rows = []

    previous_values = {
        component: float(latest[component])
        for component in COMPONENTS
    }

    for step, year in enumerate(
        forecast_years,
        start=1,
    ):

        row = {
            "Year": year,
            "Scenario": scenario_name,
        }

        # -------------------------------------------------------------
        # Forecast expenditure components
        # -------------------------------------------------------------

        for component in COMPONENTS:

            growth_rate = assumptions[component]

            value = previous_values[component] * (
                1 + growth_rate / 100
            )

            row[component] = value

            row[
                f"{component}_Growth_Pct"
            ] = growth_rate

            previous_values[component] = value

        # -------------------------------------------------------------
        # Net exports
        # -------------------------------------------------------------

        row["NX"] = (
            row["X"] - row["M"]
        )

        # -------------------------------------------------------------
        # Domestic demand
        # -------------------------------------------------------------

        row["Domestic_Demand"] = (
            row["C"]
            + row["I"]
            + row["G"]
        )

        # -------------------------------------------------------------
        # Expenditure-side GDP
        # -------------------------------------------------------------

        row["GDP_Expenditure"] = (
            row["C"]
            + row["I"]
            + row["G"]
            + row["NX"]
        )

        # -------------------------------------------------------------
        # GDP growth
        # -------------------------------------------------------------

        previous_gdp = float(
            df.iloc[-1]["Y"]
        )

        if step > 1:

            previous_forecast = rows[-1][
                "GDP_Expenditure"
            ]

            row["GDP_Growth_Pct"] = (
                (
                    row["GDP_Expenditure"]
                    / previous_forecast
                ) - 1
            ) * 100

        else:

            row["GDP_Growth_Pct"] = (
                (
                    row["GDP_Expenditure"]
                    / previous_gdp
                ) - 1
            ) * 100

        # -------------------------------------------------------------
        # Forecast interpretation
        # -------------------------------------------------------------

        if row["GDP_Growth_Pct"] < -1:

            row["GDP_Regime"] = "CONTRACTION"

        elif row["GDP_Growth_Pct"] < 1:

            row["GDP_Regime"] = "WEAK GROWTH"

        elif row["GDP_Growth_Pct"] < 2:

            row["GDP_Regime"] = "MODERATE GROWTH"

        else:

            row["GDP_Regime"] = "STRONG GROWTH"

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# ALL SCENARIOS
# =============================================================================

def generate_all_scenarios(
    df,
    horizon=FORECAST_HORIZON,
):
    """
    Generate baseline, optimistic and downside forecasts.
    """

    recent_growth = calculate_recent_growth(df)

    assumptions = create_scenario_assumptions(
        recent_growth
    )

    forecast_frames = []

    for scenario in [
        "BASELINE",
        "OPTIMISTIC",
        "DOWNSIDE",
    ]:

        scenario_df = generate_scenario_forecast(
            df=df,
            scenario_name=scenario,
            assumptions=assumptions[scenario],
            horizon=horizon,
        )

        forecast_frames.append(
            scenario_df
        )

    forecast_df = pd.concat(
        forecast_frames,
        ignore_index=True,
    )

    return {
        "forecast": forecast_df,
        "recent_growth": recent_growth,
        "assumptions": assumptions,
    }


# =============================================================================
# FORECAST SUMMARY
# =============================================================================

def create_forecast_summary(
    forecast_df,
):
    """
    Create a compact scenario summary.
    """

    rows = []

    for scenario in [
        "BASELINE",
        "OPTIMISTIC",
        "DOWNSIDE",
    ]:

        scenario_data = forecast_df[
            forecast_df["Scenario"] == scenario
        ].copy()

        if scenario_data.empty:
            continue

        first = scenario_data.iloc[0]
        last = scenario_data.iloc[-1]

        rows.append(
            {
                "Scenario": scenario,
                "First_Forecast_Year": int(
                    first["Year"]
                ),
                "Final_Forecast_Year": int(
                    last["Year"]
                ),
                "Final_GDP_Expenditure": (
                    last["GDP_Expenditure"]
                ),
                "Final_GDP_Growth_Pct": (
                    last["GDP_Growth_Pct"]
                ),
                "Final_Consumption": (
                    last["C"]
                ),
                "Final_Investment": (
                    last["I"]
                ),
                "Final_Government": (
                    last["G"]
                ),
                "Final_Exports": (
                    last["X"]
                ),
                "Final_Imports": (
                    last["M"]
                ),
                "Final_Net_Exports": (
                    last["NX"]
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# ECONOMIC INTERPRETATION
# =============================================================================

def create_forecast_interpretation(
    forecast_df,
    latest_historical_gdp,
):
    """
    Generate text-based economic interpretation.
    """

    lines = []

    lines.append(
        "UK EXPENDITURE-SIDE GDP FORECAST"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "IMPORTANT METHODOLOGICAL WARNING"
    )

    lines.append(
        "The historical dataset contains only "
        f"{latest_historical_gdp['observations']} observations."
    )

    lines.append(
        "Forecasts should therefore be interpreted "
        "as illustrative scenarios rather than "
        "high-confidence statistical predictions."
    )

    lines.append("")

    for scenario in [
        "BASELINE",
        "OPTIMISTIC",
        "DOWNSIDE",
    ]:

        data = forecast_df[
            forecast_df["Scenario"] == scenario
        ]

        if data.empty:
            continue

        first = data.iloc[0]
        last = data.iloc[-1]

        lines.append(
            f"{scenario} SCENARIO"
        )

        lines.append(
            "-" * 80
        )

        lines.append(
            f"{int(first['Year'])}: "
            f"GDP expenditure = "
            f"{first['GDP_Expenditure']:.2f}, "
            f"growth = "
            f"{first['GDP_Growth_Pct']:.2f}%"
        )

        lines.append(
            f"{int(last['Year'])}: "
            f"GDP expenditure = "
            f"{last['GDP_Expenditure']:.2f}, "
            f"growth = "
            f"{last['GDP_Growth_Pct']:.2f}%"
        )

        if last["GDP_Growth_Pct"] < 0:

            lines.append(
                "Interpretation: the scenario "
                "indicates economic contraction."
            )

        elif last["GDP_Growth_Pct"] < 1:

            lines.append(
                "Interpretation: the scenario "
                "indicates weak economic growth."
            )

        elif last["GDP_Growth_Pct"] < 2:

            lines.append(
                "Interpretation: the scenario "
                "indicates moderate economic growth."
            )

        else:

            lines.append(
                "Interpretation: the scenario "
                "indicates relatively strong growth."
            )

        lines.append("")

    lines.append(
        "ECONOMIC LIMITATION"
    )

    lines.append(
        "The scenarios are based on recent expenditure "
        "component growth and are not causal forecasts."
    )

    lines.append(
        "They should be updated when official UK "
        "national accounts data become available."
    )

    return "\n".join(lines)


# =============================================================================
# MASTER FORECASTING PIPELINE
# =============================================================================

def run_forecasting_analysis(
    df,
    horizon=FORECAST_HORIZON,
):
    """
    Run the complete Phase 06 forecasting pipeline.
    """

    sample_assessment = assess_forecast_sample(
        df
    )

    scenario_results = generate_all_scenarios(
        df,
        horizon=horizon,
    )

    forecast_df = scenario_results[
        "forecast"
    ]

    summary_df = create_forecast_summary(
        forecast_df
    )

    interpretation = create_forecast_interpretation(
        forecast_df,
        sample_assessment,
    )

    return {
        "forecast": forecast_df,
        "summary": summary_df,
        "recent_growth": scenario_results[
            "recent_growth"
        ],
        "assumptions": scenario_results[
            "assumptions"
        ],
        "sample_assessment": sample_assessment,
        "interpretation": interpretation,
    }


# =============================================================================
# PRINT REPORT
# =============================================================================

def print_forecast_report(
    results,
):
    """
    Print professional Phase 06 report.
    """

    forecast_df = results["forecast"]
    summary_df = results["summary"]
    sample = results["sample_assessment"]

    print()
    print("=" * 80)
    print("PHASE 06 — FORECASTING & ECONOMIC SCENARIOS")
    print("=" * 80)

    print()
    print("SAMPLE SIZE")
    print("-" * 80)

    print(
        f"Observations: {sample['observations']}"
    )

    print(
        f"Assessment: {sample['assessment']}"
    )

    print()
    print(
        "WARNING: Forecasts are exploratory because "
        "the historical sample is very small."
    )

    print()
    print("RECENT COMPONENT GROWTH")
    print("-" * 80)

    for component, value in results[
        "recent_growth"
    ].items():

        print(
            f"{component}: {value:.2f}%"
        )

    print()
    print("SCENARIO ASSUMPTIONS")
    print("-" * 80)

    assumptions = results[
        "assumptions"
    ]

    for scenario, values in assumptions.items():

        print()
        print(scenario)

        for component in COMPONENTS:

            print(
                f"  {component}: "
                f"{values[component]:.2f}%"
            )

    print()
    print("FORECAST RESULTS")
    print("-" * 80)

    display_columns = [
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

    print(
        forecast_df[
            display_columns
        ].round(2).to_string(
            index=False
        )
    )

    print()
    print("SCENARIO SUMMARY")
    print("-" * 80)

    print(
        summary_df.round(2).to_string(
            index=False
        )
    )

    print()
    print("ECONOMIC INTERPRETATION")
    print("-" * 80)

    print(
        results["interpretation"]
    )


# =============================================================================
# QUALITY CONTROL
# =============================================================================

def run_forecast_quality_checks(
    results,
):
    """
    Run Phase 06 quality-control checks.
    """

    forecast_df = results["forecast"]

    checks = {}

    checks["Forecast_Exists"] = (
        not forecast_df.empty
    )

    checks["Scenario_Count"] = (
        forecast_df["Scenario"].nunique() == 3
    )

    checks["Years_Valid"] = (
        forecast_df["Year"].notna().all()
    )

    checks["GDP_Valid"] = (
        forecast_df[
            "GDP_Expenditure"
        ].notna().all()
    )

    checks["Growth_Valid"] = (
        forecast_df[
            "GDP_Growth_Pct"
        ].notna().all()
    )

    checks["Components_Valid"] = (
        forecast_df[
            COMPONENTS
        ].notna().all().all()
    )

    return checks


def print_forecast_quality_report(
    checks,
):
    """
    Print Phase 06 quality-control report.
    """

    print()
    print("=" * 80)
    print("PHASE 06 QUALITY CONTROL")
    print("=" * 80)

    for name, status in checks.items():

        if status:

            print(
                f"✓ {name}: PASS"
            )

        else:

            print(
                f"✗ {name}: FAIL"
            )


# =============================================================================
# SAVE FORECAST OUTPUTS
# =============================================================================

def save_forecast_outputs(
    results,
    forecast_directory,
    clean_data_directory,
):
    """
    Save Phase 06 forecast files.
    """

    forecast_directory = Path(
        forecast_directory
    )

    clean_data_directory = Path(
        clean_data_directory
    )

    forecast_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_data_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Main forecast dataset
    forecast_file = (
        forecast_directory
        / "forecast_results.csv"
    )

    results["forecast"].to_csv(
        forecast_file,
        index=False,
    )

    # Scenario summary
    summary_file = (
        forecast_directory
        / "forecast_summary.csv"
    )

    results["summary"].to_csv(
        summary_file,
        index=False,
    )

    # Clean data copy
    clean_forecast_file = (
        clean_data_directory
        / "forecast_dataset.csv"
    )

    results["forecast"].to_csv(
        clean_forecast_file,
        index=False,
    )

    # Text report
    report_file = (
        forecast_directory
        / "forecast_report.txt"
    )

    report_file.write_text(
        results["interpretation"],
        encoding="utf-8",
    )

    return {
        "forecast_file": forecast_file,
        "summary_file": summary_file,
        "clean_forecast_file": clean_forecast_file,
        "report_file": report_file,
    }