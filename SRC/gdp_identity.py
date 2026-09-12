import pandas as pd
import numpy as np

# ============================================================
# PHASE 03 — GDP IDENTITY & GROWTH DECOMPOSITION
# ============================================================

MINOR_DISCREPANCY_PCT = 0.5
MATERIAL_DISCREPANCY_PCT = 1.0


# ============================================================
# GDP EXPENDITURE IDENTITY
# ============================================================

def calculate_gdp_identity(df):

    data = df.copy()

    data["NX"] = (
        data["X"]
        - data["M"]
    )

    data["Domestic_Demand"] = (
        data["C"]
        + data["I"]
        + data["G"]
    )

    data["GDP_Expenditure"] = (
        data["C"]
        + data["I"]
        + data["G"]
        + data["NX"]
    )

    data["GDP_Identity_Residual"] = (
        data["Y"]
        - data["GDP_Expenditure"]
    )

    data["GDP_Identity_Residual_Abs"] = (
        data["GDP_Identity_Residual"].abs()
    )

    data["GDP_Identity_Error_Pct"] = (
        data["GDP_Identity_Residual"]
        / data["Y"]
        * 100
    )

    absolute_error = (
        data["GDP_Identity_Error_Pct"].abs()
    )

    data["GDP_Identity_Status"] = np.select(
        [
            absolute_error <= MINOR_DISCREPANCY_PCT,
            absolute_error <= MATERIAL_DISCREPANCY_PCT,
            absolute_error > MATERIAL_DISCREPANCY_PCT,
        ],
        [
            "PASS",
            "MINOR DISCREPANCY",
            "MATERIAL DISCREPANCY",
        ],
        default="CHECK",
    )

    return data


# ============================================================
# GDP GROWTH
# ============================================================

def calculate_gdp_growth(df):

    data = df.copy()

    data["GDP_Growth_Pct"] = (
        data["Y"].pct_change()
        * 100
    )

    return data


# ============================================================
# GROWTH CONTRIBUTIONS
# ============================================================

def calculate_growth_contributions(df):

    data = df.copy()

    previous_gdp = (
        data["Y"].shift(1)
    )

    data["C_Change"] = (
        data["C"].diff()
    )

    data["I_Change"] = (
        data["I"].diff()
    )

    data["G_Change"] = (
        data["G"].diff()
    )

    data["NX_Change"] = (
        data["NX"].diff()
    )

    data["C_Contribution_Ppt"] = (
        data["C_Change"]
        / previous_gdp
        * 100
    )

    data["I_Contribution_Ppt"] = (
        data["I_Change"]
        / previous_gdp
        * 100
    )

    data["G_Contribution_Ppt"] = (
        data["G_Change"]
        / previous_gdp
        * 100
    )

    data["NX_Contribution_Ppt"] = (
        data["NX_Change"]
        / previous_gdp
        * 100
    )

    data["Domestic_Demand_Change"] = (
        data["C_Change"]
        + data["I_Change"]
        + data["G_Change"]
    )

    data["Domestic_Demand_Contribution_Ppt"] = (
        data["Domestic_Demand_Change"]
        / previous_gdp
        * 100
    )

    data["Component_Growth_Contribution_Ppt"] = (
        data["C_Contribution_Ppt"]
        + data["I_Contribution_Ppt"]
        + data["G_Contribution_Ppt"]
        + data["NX_Contribution_Ppt"]
    )

    # --------------------------------------------------------
    # GDP identity reconciliation
    # --------------------------------------------------------

    data["GDP_Identity_Reconciliation_Change"] = (
        data["GDP_Identity_Residual"].diff()
    )

    data["GDP_Identity_Reconciliation_Contribution_Ppt"] = (
        data["GDP_Identity_Reconciliation_Change"]
        / previous_gdp
        * 100
    )

    data["Reconciled_GDP_Growth_Ppt"] = (
        data["C_Contribution_Ppt"]
        + data["I_Contribution_Ppt"]
        + data["G_Contribution_Ppt"]
        + data["NX_Contribution_Ppt"]
        + data[
            "GDP_Identity_Reconciliation_Contribution_Ppt"
        ]
    )

    data["Growth_Decomposition_Residual"] = (
        data["GDP_Growth_Pct"]
        - data["Reconciled_GDP_Growth_Ppt"]
    )

    # --------------------------------------------------------
    # Base year handling
    # --------------------------------------------------------

    base_year = data["GDP_Growth_Pct"].isna()

    decomposition_error = (
        data["Growth_Decomposition_Residual"].abs()
    )

    data["Growth_Decomposition_Status"] = np.select(
        [
            base_year,
            decomposition_error <= 0.000001,
            decomposition_error <= 0.01,
            decomposition_error > 0.01,
        ],
        [
            "BASE YEAR",
            "PASS",
            "MINOR ROUNDING DIFFERENCE",
            "CHECK",
        ],
        default="CHECK",
    )

    return data


# ============================================================
# BUSINESS CYCLE FLAGS
# ============================================================

def calculate_business_cycle_flags(df):

    data = df.copy()

    data["GDP_Contraction"] = (
        data["GDP_Growth_Pct"] < 0
    )

    data["GDP_Expansion"] = (
        data["GDP_Growth_Pct"] > 0
    )

    data["GDP_Stagnation"] = (
        data["GDP_Growth_Pct"] == 0
    )

    data["Growth_Regime"] = np.select(
        [
            data["GDP_Growth_Pct"].isna(),
            data["GDP_Growth_Pct"] < 0,
            data["GDP_Growth_Pct"] > 0,
            data["GDP_Growth_Pct"] == 0,
        ],
        [
            "BASE YEAR",
            "CONTRACTION",
            "EXPANSION",
            "STAGNATION",
        ],
        default="UNKNOWN",
    )

    negative_growth = (
        data["GDP_Growth_Pct"] < 0
    )

    data["Two_Period_Contraction"] = (
        negative_growth
        & negative_growth.shift(1).fillna(False)
    )

    return data


# ============================================================
# MASTER ANALYSIS FUNCTION
# ============================================================

def run_gdp_identity_analysis(df):

    data = calculate_gdp_identity(df)

    data = calculate_gdp_growth(data)

    data = calculate_growth_contributions(data)

    data = calculate_business_cycle_flags(data)

    return data


# ============================================================
# PROFESSIONAL REPORT
# ============================================================

def print_gdp_identity_report(df):

    print()
    print("=" * 80)
    print("PHASE 03 — GDP IDENTITY & GROWTH DECOMPOSITION")
    print("=" * 80)

    print()
    print("GDP EXPENDITURE IDENTITY:")
    print()
    print("Y = C + I + G + (X - M)")

    print()
    print("GDP IDENTITY CHECK")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "Y",
                "GDP_Expenditure",
                "GDP_Identity_Residual",
                "GDP_Identity_Error_Pct",
                "GDP_Identity_Status",
            ]
        ].to_string(index=False)
    )

    print()
    print("GDP GROWTH")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "GDP_Growth_Pct",
                "Growth_Regime",
            ]
        ].to_string(index=False)
    )

    print()
    print("GDP GROWTH DECOMPOSITION")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "GDP_Growth_Pct",
                "C_Contribution_Ppt",
                "I_Contribution_Ppt",
                "G_Contribution_Ppt",
                "NX_Contribution_Ppt",
                "GDP_Identity_Reconciliation_Contribution_Ppt",
                "Reconciled_GDP_Growth_Ppt",
                "Growth_Decomposition_Residual",
                "Growth_Decomposition_Status",
            ]
        ].to_string(index=False)
    )

    print()
    print("BUSINESS CYCLE FLAGS")
    print("-" * 80)

    print(
        df[
            [
                "Year",
                "GDP_Growth_Pct",
                "Growth_Regime",
                "GDP_Contraction",
                "GDP_Expansion",
                "Two_Period_Contraction",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)
    print("PHASE 03 REPORT COMPLETE")
    print("=" * 80)