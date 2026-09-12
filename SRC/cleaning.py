import pandas as pd


# ============================================================
# PHASE 02 — PROFESSIONAL DATA CLEANING
# ============================================================

NUMERIC_COLUMNS = [
    "Y",
    "C",
    "I",
    "G",
    "X",
    "M",
    "Rate",
    "FX",
]


# ============================================================
# CLEAN GDP DATA
# ============================================================

def clean_gdp_data(df):

    data = df.copy()

    # --------------------------------------------------------
    # STANDARDISE COLUMN NAMES
    # --------------------------------------------------------

    data.columns = data.columns.str.strip()

    # --------------------------------------------------------
    # CONVERT YEAR
    # --------------------------------------------------------

    if "Year" not in data.columns:
        raise ValueError(
            "Required column 'Year' is missing."
        )

    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="coerce"
    )

    if data["Year"].isna().any():
        raise ValueError(
            "Year contains invalid or missing values."
        )

    data["Year"] = data["Year"].astype(int)

    # --------------------------------------------------------
    # CONVERT NUMERIC VARIABLES
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        if column not in data.columns:
            raise ValueError(
                f"Required column '{column}' is missing."
            )

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # CHECK MISSING VALUES
    # --------------------------------------------------------

    required_columns = [
        "Year",
        *NUMERIC_COLUMNS,
    ]

    missing_values = (
        data[required_columns]
        .isna()
        .sum()
    )

    missing_values = missing_values[
        missing_values > 0
    ]

    if not missing_values.empty:
        raise ValueError(
            "Missing values detected:\n"
            f"{missing_values.to_dict()}"
        )

    # --------------------------------------------------------
    # CHECK DUPLICATE YEARS
    # --------------------------------------------------------

    duplicate_years = data["Year"].duplicated()

    if duplicate_years.any():

        duplicates = (
            data.loc[
                duplicate_years,
                "Year"
            ].tolist()
        )

        raise ValueError(
            f"Duplicate years detected: {duplicates}"
        )

    # --------------------------------------------------------
    # SORT BY YEAR
    # --------------------------------------------------------

    data = (
        data
        .sort_values("Year")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # NET EXPORTS
    # --------------------------------------------------------

    data["NX"] = (
        data["X"] - data["M"]
    )

    # --------------------------------------------------------
    # DOMESTIC DEMAND
    # --------------------------------------------------------

    data["Domestic_Demand"] = (
        data["C"]
        + data["I"]
        + data["G"]
    )

    # --------------------------------------------------------
    # EXPENDITURE GDP
    # --------------------------------------------------------

    data["GDP_Expenditure"] = (
        data["C"]
        + data["I"]
        + data["G"]
        + data["NX"]
    )

    # --------------------------------------------------------
    # GDP IDENTITY RESIDUAL
    # --------------------------------------------------------

    data["GDP_Identity_Residual"] = (
        data["Y"]
        - data["GDP_Expenditure"]
    )

    # --------------------------------------------------------
    # ABSOLUTE RESIDUAL
    # --------------------------------------------------------

    data["GDP_Identity_Residual_Abs"] = (
        data["GDP_Identity_Residual"].abs()
    )

    # --------------------------------------------------------
    # GDP IDENTITY ERROR %
    # --------------------------------------------------------

    data["GDP_Identity_Error_Pct"] = (
        data["GDP_Identity_Residual"]
        / data["Y"]
        * 100
    )

    # --------------------------------------------------------
    # GDP IDENTITY STATUS
    # --------------------------------------------------------

    data["GDP_Identity_Status"] = (
        data["GDP_Identity_Residual_Abs"]
        < 1e-9
    )

    # --------------------------------------------------------
    # DATA QUALITY STATUS
    # --------------------------------------------------------

    data["Data_Quality_Status"] = "PASS"

    expenditure_columns = [
        "Y",
        "C",
        "I",
        "G",
        "X",
        "M",
    ]

    for column in expenditure_columns:

        data.loc[
            data[column] < 0,
            "Data_Quality_Status"
        ] = "WARNING"

    # --------------------------------------------------------
    # GDP GROWTH
    # --------------------------------------------------------

    data["GDP_Growth_Pct"] = (
        data["Y"].pct_change() * 100
    )

    # --------------------------------------------------------
    # COMPONENT GROWTH
    # --------------------------------------------------------

    for column in [
        "C",
        "I",
        "G",
        "X",
        "M",
        "NX",
        "Domestic_Demand",
    ]:

        data[f"{column}_Growth_Pct"] = (
            data[column].pct_change() * 100
        )

    # --------------------------------------------------------
    # GDP COMPONENT SHARES
    # --------------------------------------------------------

    data["Consumption_Share_Pct"] = (
        data["C"] / data["Y"] * 100
    )

    data["Investment_Share_Pct"] = (
        data["I"] / data["Y"] * 100
    )

    data["Government_Share_Pct"] = (
        data["G"] / data["Y"] * 100
    )

    data["Net_Exports_Share_Pct"] = (
        data["NX"] / data["Y"] * 100
    )

    return data


# ============================================================
# PRINT CLEANING REPORT
# ============================================================

def print_cleaning_report(df):

    print()
    print("=" * 80)
    print("PHASE 02 — PROFESSIONAL DATA CLEANING REPORT")
    print("=" * 80)

    print()
    print(f"Observations: {len(df)}")

    print(
        f"Variables: {len(df.columns)}"
    )

    print()

    print(
        f"Years: "
        f"{df['Year'].min()} - "
        f"{df['Year'].max()}"
    )

    print()

    print("GDP identity observations:")

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

    print("Data quality:")

    print(
        df[
            [
                "Year",
                "Data_Quality_Status",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)