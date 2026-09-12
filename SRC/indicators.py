import pandas as pd
import numpy as np

# ============================================================
# PHASE 04 — PROFESSIONAL MACROECONOMIC INDICATORS
# EXPENDITURE-SIDE GDP MODEL
# ============================================================


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_growth(series):
    """
    Calculate year-on-year percentage growth.
    """
    return series.pct_change() * 100


def calculate_change(series):
    """
    Calculate absolute period-to-period change.
    """
    return series.diff()


# ============================================================
# GDP INDICATORS
# ============================================================

def calculate_gdp_indicators(df):
    data = df.copy()

    # GDP growth
    data["GDP_Growth_Pct"] = calculate_growth(
        data["Y"]
    )

    # GDP acceleration / deceleration
    data["GDP_Growth_Change_Ppt"] = (
        data["GDP_Growth_Pct"].diff()
    )

    # GDP momentum
    data["GDP_Momentum"] = np.select(
        [
            data["GDP_Growth_Change_Ppt"] > 0.5,
            data["GDP_Growth_Change_Ppt"] < -0.5,
        ],
        [
            "ACCELERATING",
            "DECELERATING",
        ],
        default="STABLE",
    )

    # GDP regime
    data["GDP_Regime"] = np.select(
        [
            data["GDP_Growth_Pct"].isna(),
            data["GDP_Growth_Pct"] < 0,
            data["GDP_Growth_Pct"] == 0,
            data["GDP_Growth_Pct"] > 0,
        ],
        [
            "BASE YEAR",
            "CONTRACTION",
            "STAGNATION",
            "EXPANSION",
        ],
        default="UNKNOWN",
    )

    # Strong / weak growth classification
    data["GDP_Growth_Strength"] = np.select(
        [
            data["GDP_Growth_Pct"].isna(),
            data["GDP_Growth_Pct"] < 0,
            data["GDP_Growth_Pct"] < 1.0,
            data["GDP_Growth_Pct"] < 2.0,
            data["GDP_Growth_Pct"] >= 2.0,
        ],
        [
            "BASE YEAR",
            "NEGATIVE GROWTH",
            "WEAK GROWTH",
            "MODERATE GROWTH",
            "STRONG GROWTH",
        ],
        default="UNKNOWN",
    )

    return data


# ============================================================
# EXPENDITURE COMPONENT INDICATORS
# ============================================================

def calculate_expenditure_indicators(df):
    data = df.copy()

    components = {
        "C": "Consumption",
        "I": "Investment",
        "G": "Government",
        "X": "Exports",
        "M": "Imports",
        "NX": "Net_Exports",
        "Domestic_Demand": "Domestic_Demand",
    }

    for column, label in components.items():

        growth_column = f"{label}_Growth_Pct"

        data[growth_column] = calculate_growth(
            data[column]
        )

        data[f"{label}_Growth_Change_Ppt"] = (
            data[growth_column].diff()
        )

    return data


# ============================================================
# GDP SHARES
# ============================================================

def calculate_gdp_shares(df):
    data = df.copy()

    data["Consumption_Share_Pct"] = (
        data["C"] / data["Y"] * 100
    )

    data["Investment_Share_Pct"] = (
        data["I"] / data["Y"] * 100
    )

    data["Government_Share_Pct"] = (
        data["G"] / data["Y"] * 100
    )

    data["Exports_Share_Pct"] = (
        data["X"] / data["Y"] * 100
    )

    data["Imports_Share_Pct"] = (
        data["M"] / data["Y"] * 100
    )

    data["Net_Exports_Share_Pct"] = (
        data["NX"] / data["Y"] * 100
    )

    data["Domestic_Demand_Share_Pct"] = (
        data["Domestic_Demand"] / data["Y"] * 100
    )

    return data


# ============================================================
# INTEREST RATE INDICATORS
# ============================================================

def calculate_interest_rate_indicators(df):
    data = df.copy()

    data["Interest_Rate_Change_Ppt"] = (
        data["Rate"].diff()
    )

    data["Interest_Rate_Direction"] = np.select(
        [
            data["Interest_Rate_Change_Ppt"] > 0,
            data["Interest_Rate_Change_Ppt"] < 0,
        ],
        [
            "TIGHTENING",
            "EASING",
        ],
        default="UNCHANGED",
    )

    data["Financial_Conditions_Regime"] = np.select(
        [
            data["Interest_Rate_Change_Ppt"] > 0.50,
            data["Interest_Rate_Change_Ppt"] > 0,
            data["Interest_Rate_Change_Ppt"] < -0.50,
            data["Interest_Rate_Change_Ppt"] < 0,
        ],
        [
            "STRONG TIGHTENING",
            "MILD TIGHTENING",
            "STRONG EASING",
            "MILD EASING",
        ],
        default="NEUTRAL",
    )

    return data


# ============================================================
# FX INDICATORS
# ============================================================

def calculate_fx_indicators(df):
    data = df.copy()

    data["FX_Change"] = data["FX"].diff()

    data["FX_Change_Pct"] = (
        data["FX"].pct_change() * 100
    )

    data["FX_Momentum"] = np.select(
        [
            data["FX_Change_Pct"] > 1,
            data["FX_Change_Pct"] < -1,
        ],
        [
            "APPRECIATION",
            "DEPRECIATION",
        ],
        default="STABLE",
    )

    # Annual FX volatility proxy
    data["FX_Volatility"] = (
        data["FX_Change_Pct"].rolling(
            window=3,
            min_periods=2
        ).std()
    )

    return data


# ============================================================
# DOMESTIC DEMAND INDICATORS
# ============================================================

def calculate_domestic_demand_indicators(df):
    data = df.copy()

    data["Domestic_Demand_Growth_Pct"] = (
        data["Domestic_Demand"].pct_change()
        * 100
    )

    data["Domestic_Demand_Momentum_Ppt"] = (
        data["Domestic_Demand_Growth_Pct"].diff()
    )

    data["Domestic_Demand_Regime"] = np.select(
        [
            data["Domestic_Demand_Growth_Pct"].isna(),
            data["Domestic_Demand_Growth_Pct"] < 0,
            data["Domestic_Demand_Growth_Pct"] == 0,
            data["Domestic_Demand_Growth_Pct"] > 0,
        ],
        [
            "BASE YEAR",
            "CONTRACTION",
            "STAGNATION",
            "EXPANSION",
        ],
        default="UNKNOWN",
    )

    return data


# ============================================================
# EXTERNAL SECTOR INDICATORS
# ============================================================

def calculate_external_sector_indicators(df):
    data = df.copy()

    data["Trade_Balance"] = (
        data["X"] - data["M"]
    )

    data["Trade_Balance_Share_Pct"] = (
        data["Trade_Balance"]
        / data["Y"]
        * 100
    )

    data["Export_Import_Coverage_Ratio"] = (
        data["X"] / data["M"]
    )

    data["External_Sector_Regime"] = np.select(
        [
            data["NX"] > 0,
            data["NX"] < 0,
            data["NX"] == 0,
        ],
        [
            "NET EXPORT SURPLUS",
            "NET EXPORT DEFICIT",
            "BALANCED",
        ],
        default="UNKNOWN",
    )

    return data


# ============================================================
# MACROECONOMIC VOLATILITY
# ============================================================

def calculate_volatility_indicators(df):
    data = df.copy()

    # Three-period rolling volatility is appropriate
    # for this annual test dataset.
    data["GDP_Growth_Volatility"] = (
        data["GDP_Growth_Pct"]
        .rolling(
            window=3,
            min_periods=2
        )
        .std()
    )

    data["Domestic_Demand_Growth_Volatility"] = (
        data["Domestic_Demand_Growth_Pct"]
        .rolling(
            window=3,
            min_periods=2
        )
        .std()
    )

    data["Investment_Growth_Volatility"] = (
        data["Investment_Growth_Pct"]
        .rolling(
            window=3,
            min_periods=2
        )
        .std()
    )

    return data


# ============================================================
# MACROECONOMIC REGIME CLASSIFICATION
# ============================================================

def calculate_macro_regime(df):
    data = df.copy()

    data["Macro_Regime"] = np.select(
        [
            (
                (data["GDP_Growth_Pct"] < 0)
                & (
                    data["Domestic_Demand_Growth_Pct"]
                    < 0
                )
            ),

            (
                (data["GDP_Growth_Pct"] > 0)
                & (
                    data["Domestic_Demand_Growth_Pct"]
                    > 0
                )
                & (
                    data["Interest_Rate_Change_Ppt"]
                    > 0
                )
            ),

            (
                (data["GDP_Growth_Pct"] > 0)
                & (
                    data["Domestic_Demand_Growth_Pct"]
                    > 0
                )
                & (
                    data["Interest_Rate_Change_Ppt"]
                    < 0
                )
            ),

            (
                (data["GDP_Growth_Pct"] > 0)
                & (
                    data["Domestic_Demand_Growth_Pct"]
                    > 0
                )
            ),
        ],
        [
            "BROAD CONTRACTION",
            "EXPANSION WITH TIGHTENING",
            "EXPANSION WITH EASING",
            "BROAD EXPANSION",
        ],
        default="MIXED / TRANSITION",
    )

    return data


# ============================================================
# MASTER INDICATOR PIPELINE
# ============================================================

def calculate_macro_indicators(df):

    data = df.copy()

    # Ensure required calculated variables exist
    if "NX" not in data.columns:
        data["NX"] = (
            data["X"] - data["M"]
        )

    if "Domestic_Demand" not in data.columns:
        data["Domestic_Demand"] = (
            data["C"]
            + data["I"]
            + data["G"]
        )

    data = calculate_gdp_indicators(
        data
    )

    data = calculate_expenditure_indicators(
        data
    )

    data = calculate_gdp_shares(
        data
    )

    data = calculate_interest_rate_indicators(
        data
    )

    data = calculate_fx_indicators(
        data
    )

    data = calculate_domestic_demand_indicators(
        data
    )

    data = calculate_external_sector_indicators(
        data
    )

    data = calculate_volatility_indicators(
        data
    )

    data = calculate_macro_regime(
        data
    )

    return data


# ============================================================
# PROFESSIONAL INDICATOR REPORT
# ============================================================

def print_macro_indicator_report(df):

    print()
    print("=" * 80)
    print("PHASE 04 — PROFESSIONAL MACROECONOMIC INDICATORS")
    print("=" * 80)

    print()
    print("GDP INDICATORS")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "Y",
                "GDP_Growth_Pct",
                "GDP_Growth_Change_Ppt",
                "GDP_Momentum",
                "GDP_Regime",
                "GDP_Growth_Strength",
            ]
        ].to_string(index=False)
    )

    print()
    print("EXPENDITURE COMPONENT GROWTH")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "Consumption_Growth_Pct",
                "Investment_Growth_Pct",
                "Government_Growth_Pct",
                "Exports_Growth_Pct",
                "Imports_Growth_Pct",
                "Net_Exports_Growth_Pct",
                "Domestic_Demand_Growth_Pct",
            ]
        ].to_string(index=False)
    )

    print()
    print("GDP STRUCTURE")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "Consumption_Share_Pct",
                "Investment_Share_Pct",
                "Government_Share_Pct",
                "Exports_Share_Pct",
                "Imports_Share_Pct",
                "Net_Exports_Share_Pct",
            ]
        ].to_string(index=False)
    )

    print()
    print("INTEREST RATE CONDITIONS")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "Rate",
                "Interest_Rate_Change_Ppt",
                "Interest_Rate_Direction",
                "Financial_Conditions_Regime",
            ]
        ].to_string(index=False)
    )

    print()
    print("FX CONDITIONS")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "FX",
                "FX_Change",
                "FX_Change_Pct",
                "FX_Momentum",
                "FX_Volatility",
            ]
        ].to_string(index=False)
    )

    print()
    print("EXTERNAL SECTOR")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "Trade_Balance",
                "Trade_Balance_Share_Pct",
                "Export_Import_Coverage_Ratio",
                "External_Sector_Regime",
            ]
        ].to_string(index=False)
    )

    print()
    print("VOLATILITY INDICATORS")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "GDP_Growth_Volatility",
                "Domestic_Demand_Growth_Volatility",
                "Investment_Growth_Volatility",
            ]
        ].to_string(index=False)
    )

    print()
    print("MACROECONOMIC REGIME")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "GDP_Regime",
                "Domestic_Demand_Regime",
                "Financial_Conditions_Regime",
                "External_Sector_Regime",
                "Macro_Regime",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)
    print("PHASE 04 REPORT COMPLETE")
    print("=" * 80)


# ============================================================
# PHASE 04 QUALITY CONTROL
# ============================================================

def run_indicator_quality_checks(df):

    required_columns = [
        "Year",
        "GDP_Growth_Pct",
        "Consumption_Growth_Pct",
        "Investment_Growth_Pct",
        "Government_Growth_Pct",
        "Exports_Growth_Pct",
        "Imports_Growth_Pct",
        "Net_Exports_Growth_Pct",
        "Domestic_Demand_Growth_Pct",
        "Consumption_Share_Pct",
        "Investment_Share_Pct",
        "Government_Share_Pct",
        "Net_Exports_Share_Pct",
        "Interest_Rate_Change_Ppt",
        "FX_Change_Pct",
        "GDP_Growth_Volatility",
        "Macro_Regime",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    return {
        "required_columns_present":
            len(missing_columns) == 0,

        "missing_columns":
            missing_columns,

        "row_count":
            len(df),

        "indicator_count":
            len(df.columns),
    }


def print_indicator_quality_report(results):

    print()
    print("=" * 80)
    print("PHASE 04 QUALITY CONTROL")
    print("=" * 80)

    print()

    if results["required_columns_present"]:
        print(
            "✓ Required indicator columns: PASS"
        )
    else:
        print(
            "✗ Required indicator columns: CHECK"
        )

        print()
        print(
            "Missing columns:"
        )

        for column in results[
            "missing_columns"
        ]:
            print(
                f"  - {column}"
            )

    print()
    print(
        f"Observations: "
        f"{results['row_count']}"
    )

    print(
        f"Total variables: "
        f"{results['indicator_count']}"
    )

    print()
    print("=" * 80)