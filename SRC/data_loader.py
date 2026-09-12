from pathlib import Path

import pandas as pd


# =============================================================================
# DATA LOADER
# UK EXPENDITURE-SIDE GDP MODEL
# =============================================================================


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


NUMERIC_COLUMNS = [
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


# =============================================================================
# LOAD GDP DATA
# =============================================================================

def load_gdp_data(file_path):
    """
    Load the UK expenditure-side GDP dataset.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Location of the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded GDP dataset.
    """

    file_path = Path(file_path)

    # -------------------------------------------------------------------------
    # Check file exists
    # -------------------------------------------------------------------------

    if not file_path.exists():

        raise FileNotFoundError(
            f"GDP data file not found:\n{file_path}"
        )

    # -------------------------------------------------------------------------
    # Check file extension
    # -------------------------------------------------------------------------

    if file_path.suffix.lower() != ".csv":

        raise ValueError(
            "Expected a CSV file.\n"
            f"Received: {file_path}"
        )

    # -------------------------------------------------------------------------
    # Read CSV
    # -------------------------------------------------------------------------

    try:

        df = pd.read_csv(file_path)

    except Exception as error:

        raise RuntimeError(
            "Unable to read GDP CSV file.\n"
            f"File: {file_path}\n"
            f"Error: {error}"
        ) from error

    # -------------------------------------------------------------------------
    # Clean column names
    # -------------------------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # -------------------------------------------------------------------------
    # Check required columns
    # -------------------------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "GDP dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    # -------------------------------------------------------------------------
    # Keep required columns in professional order
    # -------------------------------------------------------------------------

    extra_columns = [
        column
        for column in df.columns
        if column not in REQUIRED_COLUMNS
    ]

    ordered_columns = (
        REQUIRED_COLUMNS
        + extra_columns
    )

    df = df[ordered_columns]

    # -------------------------------------------------------------------------
    # Convert numeric variables
    # -------------------------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Check conversion errors
    # -------------------------------------------------------------------------

    conversion_errors = (
        df[NUMERIC_COLUMNS]
        .isna()
        .sum()
    )

    conversion_errors = (
        conversion_errors[
            conversion_errors > 0
        ]
    )

    if len(conversion_errors) > 0:

        error_message = (
            "Numeric conversion produced missing values:\n"
        )

        for column, count in conversion_errors.items():

            error_message += (
                f"  - {column}: "
                f"{count} missing value(s)\n"
            )

        raise ValueError(error_message)

    # -------------------------------------------------------------------------
    # Convert Year to integer
    # -------------------------------------------------------------------------

    df["Year"] = df["Year"].astype(int)

    # -------------------------------------------------------------------------
    # Sort by year
    # -------------------------------------------------------------------------

    df = df.sort_values(
        "Year"
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Check duplicate years
    # -------------------------------------------------------------------------

    duplicate_years = (
        df.loc[
            df["Year"].duplicated(),
            "Year",
        ]
        .tolist()
    )

    if duplicate_years:

        raise ValueError(
            "Duplicate years detected:\n"
            f"{duplicate_years}"
        )

    # -------------------------------------------------------------------------
    # Check minimum dataset size
    # -------------------------------------------------------------------------

    if len(df) == 0:

        raise ValueError(
            "GDP dataset contains no observations."
        )

    # -------------------------------------------------------------------------
    # Check GDP variables for negative values
    # -------------------------------------------------------------------------

    economic_columns = [
        "Y",
        "C",
        "I",
        "G",
        "X",
        "M",
    ]

    negative_values = {}

    for column in economic_columns:

        count = (
            df[column] < 0
        ).sum()

        if count > 0:

            negative_values[column] = int(
                count
            )

    if negative_values:

        message = (
            "Negative economic values detected:\n"
        )

        for column, count in negative_values.items():

            message += (
                f"  - {column}: "
                f"{count} observation(s)\n"
            )

        raise ValueError(message)

    # -------------------------------------------------------------------------
    # Return clean loaded dataframe
    # -------------------------------------------------------------------------

    return df


# =============================================================================
# OPTIONAL ALIAS
# =============================================================================

def load_data(file_path):
    """
    Compatibility alias for load_gdp_data().
    """

    return load_gdp_data(file_path)


# =============================================================================
# DATA LOADER REPORT
# =============================================================================

def print_data_loader_report(df):
    """
    Print a professional summary of the loaded dataset.
    """

    print()
    print("=" * 80)
    print("DATA LOADED SUCCESSFULLY")
    print("=" * 80)

    print()
    print(f"Observations: {len(df)}")

    print()
    print(f"Variables: {len(df.columns)}")

    print()
    print("Columns:")

    for column in df.columns:

        print(f"  - {column}")

    print()
    print("Year range:")

    print(
        f"  {int(df['Year'].min())} - "
        f"{int(df['Year'].max())}"
    )