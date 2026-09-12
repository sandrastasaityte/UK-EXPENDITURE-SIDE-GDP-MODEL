import pandas as pd


# ============================================================
# REQUIRED VARIABLES
# ============================================================

REQUIRED_COLUMNS = [
    "Year",
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
# CHECK REQUIRED COLUMNS
# ============================================================

def check_required_columns(df):

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        return {
            "status": "FAIL",
            "missing_columns": missing,
        }

    return {
        "status": "PASS",
        "missing_columns": [],
    }


# ============================================================
# CHECK MISSING VALUES
# ============================================================

def check_missing_values(df):

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if missing.empty:

        return {
            "status": "PASS",
            "details": {},
        }

    return {
        "status": "FAIL",
        "details": missing.to_dict(),
    }


# ============================================================
# CHECK DUPLICATE ROWS
# ============================================================

def check_duplicates(df):

    duplicates = int(
        df.duplicated().sum()
    )

    if duplicates == 0:

        status = "PASS"

    else:

        status = "FAIL"

    return {
        "status": status,
        "duplicates": duplicates,
    }


# ============================================================
# CHECK TIME SERIES
# ============================================================

def check_years(df):

    if "Year" not in df.columns:

        return {
            "status": "FAIL",
            "duplicate_years": 0,
            "missing_years": [],
        }

    years = (
        df["Year"]
        .dropna()
        .astype(int)
        .sort_values()
    )

    duplicate_years = int(
        years.duplicated().sum()
    )

    missing_years = []

    if len(years) > 1:

        expected = list(
            range(
                years.min(),
                years.max() + 1
            )
        )

        actual = years.tolist()

        missing_years = [
            year
            for year in expected
            if year not in actual
        ]

    passed = (
        duplicate_years == 0
        and len(missing_years) == 0
    )

    return {
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "duplicate_years": duplicate_years,
        "missing_years": missing_years,
    }


# ============================================================
# CHECK NUMERIC VARIABLES
# ============================================================

def check_numeric_variables(df):

    numeric_columns = [
        "Y",
        "C",
        "I",
        "G",
        "X",
        "M",
        "Rate",
        "FX",
    ]

    invalid = {}

    for column in numeric_columns:

        if column not in df.columns:

            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_count = int(
            (
                converted.isna()
                & df[column].notna()
            ).sum()
        )

        if invalid_count > 0:

            invalid[column] = invalid_count

    return {
        "status": (
            "PASS"
            if not invalid
            else "FAIL"
        ),
        "invalid_values": invalid,
    }


# ============================================================
# CHECK NEGATIVE VALUES
# ============================================================

def check_negative_values(df):

    columns = [
        "Y",
        "C",
        "I",
        "G",
        "X",
        "M",
    ]

    negative = {}

    for column in columns:

        if column not in df.columns:

            continue

        count = int(
            (df[column] < 0).sum()
        )

        if count > 0:

            negative[column] = count

    return {
        "status": (
            "PASS"
            if not negative
            else "WARNING"
        ),
        "negative_values": negative,
    }


# ============================================================
# RUN ALL QUALITY CHECKS
# ============================================================

def run_quality_checks(df):

    return {

        "required_columns":
            check_required_columns(df),

        "missing_values":
            check_missing_values(df),

        "duplicates":
            check_duplicates(df),

        "years":
            check_years(df),

        "numeric_variables":
            check_numeric_variables(df),

        "negative_values":
            check_negative_values(df),

    }


# ============================================================
# PRINT QUALITY REPORT
# ============================================================

def print_quality_report(results):

    print()
    print("=" * 80)
    print("PHASE 01 — DATA QUALITY REPORT")
    print("=" * 80)

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    result = results["required_columns"]

    print()
    print(
        f"Required columns: {result['status']}"
    )

    if result["missing_columns"]:

        print(
            f"Missing columns: "
            f"{result['missing_columns']}"
        )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    result = results["missing_values"]

    print(
        f"Missing values: {result['status']}"
    )

    if result["details"]:

        print(
            f"Missing observations: "
            f"{result['details']}"
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    result = results["duplicates"]

    print(
        f"Duplicate rows: {result['status']}"
    )

    print(
        f"Number of duplicates: "
        f"{result['duplicates']}"
    )

    # --------------------------------------------------------
    # YEARS
    # --------------------------------------------------------

    result = results["years"]

    print(
        f"Time-series structure: "
        f"{result['status']}"
    )

    print(
        f"Duplicate years: "
        f"{result['duplicate_years']}"
    )

    if result["missing_years"]:

        print(
            f"Missing years: "
            f"{result['missing_years']}"
        )

    # --------------------------------------------------------
    # NUMERIC VARIABLES
    # --------------------------------------------------------

    result = results[
        "numeric_variables"
    ]

    print(
        f"Numeric variables: "
        f"{result['status']}"
    )

    if result["invalid_values"]:

        print(
            f"Invalid numeric values: "
            f"{result['invalid_values']}"
        )

    # --------------------------------------------------------
    # NEGATIVE VALUES
    # --------------------------------------------------------

    result = results[
        "negative_values"
    ]

    print(
        f"Negative values: "
        f"{result['status']}"
    )

    if result["negative_values"]:

        print(
            f"Negative observations: "
            f"{result['negative_values']}"
        )

    print()
    print("=" * 80)