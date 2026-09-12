import pandas as pd
import numpy as np

# Optional econometrics package
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ============================================================
# PHASE 05 — ECONOMETRICS
# PROFESSIONAL ECONOMIST VERSION
# ============================================================

MIN_OBSERVATIONS_FOR_REGRESSION = 8
VERY_SMALL_SAMPLE_THRESHOLD = 15


# ============================================================
# SAMPLE SIZE ASSESSMENT
# ============================================================

def assess_sample_size(df):

    observations = len(df)

    if observations < 8:
        status = "INSUFFICIENT"
        warning = (
            "Too few observations for meaningful regression analysis."
        )

    elif observations < VERY_SMALL_SAMPLE_THRESHOLD:
        status = "VERY SMALL SAMPLE"
        warning = (
            "Econometric results are indicative only and "
            "should not be interpreted as robust causal estimates."
        )

    else:
        status = "ADEQUATE FOR EXPLORATORY ANALYSIS"
        warning = (
            "Results remain observational and should not automatically "
            "be interpreted as causal."
        )

    return {
        "observations": observations,
        "status": status,
        "warning": warning,
    }


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def calculate_correlation_matrix(df):

    variables = [
        "GDP_Growth_Pct",
        "Consumption_Growth_Pct",
        "Investment_Growth_Pct",
        "Government_Growth_Pct",
        "Exports_Growth_Pct",
        "Imports_Growth_Pct",
        "Domestic_Demand_Growth_Pct",
        "Rate",
        "FX",
        "FX_Change_Pct",
    ]

    available = [
        column
        for column in variables
        if column in df.columns
    ]

    correlation_matrix = (
        df[available]
        .corr()
    )

    return correlation_matrix


# ============================================================
# GDP GROWTH CORRELATIONS
# ============================================================

def calculate_gdp_correlations(df):

    target = "GDP_Growth_Pct"

    explanatory_variables = [
        "Consumption_Growth_Pct",
        "Investment_Growth_Pct",
        "Government_Growth_Pct",
        "Exports_Growth_Pct",
        "Imports_Growth_Pct",
        "Domestic_Demand_Growth_Pct",
        "Rate",
        "FX_Change_Pct",
    ]

    records = []

    for variable in explanatory_variables:

        if variable not in df.columns:
            continue

        valid = df[
            [target, variable]
        ].dropna()

        if len(valid) >= 3:
            correlation = valid[
                target
            ].corr(
                valid[variable]
            )

        else:
            correlation = np.nan

        records.append(
            {
                "Variable": variable,
                "Correlation_With_GDP_Growth": correlation,
                "Absolute_Correlation": (
                    abs(correlation)
                    if pd.notna(correlation)
                    else np.nan
                ),
                "Observations": len(valid),
            }
        )

    result = pd.DataFrame(records)

    if not result.empty:
        result = result.sort_values(
            "Absolute_Correlation",
            ascending=False
        ).reset_index(drop=True)

    return result


# ============================================================
# CORRELATION STRENGTH
# ============================================================

def classify_correlation(value):

    if pd.isna(value):
        return "INSUFFICIENT DATA"

    absolute_value = abs(value)

    if absolute_value >= 0.80:
        strength = "VERY STRONG"
    elif absolute_value >= 0.60:
        strength = "STRONG"
    elif absolute_value >= 0.40:
        strength = "MODERATE"
    elif absolute_value >= 0.20:
        strength = "WEAK"
    else:
        strength = "VERY WEAK"

    if value > 0:
        direction = "POSITIVE"
    elif value < 0:
        direction = "NEGATIVE"
    else:
        direction = "ZERO"

    return f"{strength} {direction}"


def add_correlation_classification(df):

    data = df.copy()

    data["Correlation_Classification"] = (
        data[
            "Correlation_With_GDP_Growth"
        ].apply(
            classify_correlation
        )
    )

    return data


# ============================================================
# OLS REGRESSION
# ============================================================

def run_ols_regression(
    df,
    dependent_variable,
    explanatory_variables,
):

    if not STATSMODELS_AVAILABLE:
        return {
            "status": "UNAVAILABLE",
            "message": (
                "statsmodels is not installed."
            ),
            "model": None,
            "data": None,
        }

    required = [
        dependent_variable,
        *explanatory_variables,
    ]

    available = [
        column
        for column in required
        if column in df.columns
    ]

    if len(available) != len(required):
        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        return {
            "status": "MISSING VARIABLES",
            "message": (
                f"Missing variables: {missing}"
            ),
            "model": None,
            "data": None,
        }

    model_data = (
        df[required]
        .dropna()
        .copy()
    )

    observations = len(model_data)

    # Avoid over-parameterisation.
    if observations <= len(explanatory_variables) + 2:

        return {
            "status": "INSUFFICIENT OBSERVATIONS",
            "message": (
                f"{observations} observations available for "
                f"{len(explanatory_variables)} explanatory variables."
            ),
            "model": None,
            "data": model_data,
        }

    X = model_data[
        explanatory_variables
    ]

    X = sm.add_constant(
        X,
        has_constant="add"
    )

    y = model_data[
        dependent_variable
    ]

    model = sm.OLS(
        y,
        X
    ).fit()

    return {
        "status": "SUCCESS",
        "message": "OLS regression completed.",
        "model": model,
        "data": model_data,
    }


# ============================================================
# SIMPLE GDP GROWTH MODEL
# ============================================================

def run_simple_gdp_regression(df):

    explanatory_variables = [
        "Domestic_Demand_Growth_Pct"
    ]

    return run_ols_regression(
        df,
        "GDP_Growth_Pct",
        explanatory_variables,
    )


# ============================================================
# GDP COMPONENT REGRESSION
# ============================================================

def run_component_regression(df):

    explanatory_variables = [
        "Consumption_Growth_Pct",
        "Investment_Growth_Pct",
        "Government_Growth_Pct",
        "Net_Exports_Growth_Pct",
    ]

    return run_ols_regression(
        df,
        "GDP_Growth_Pct",
        explanatory_variables,
    )


# ============================================================
# FINANCIAL CONDITIONS REGRESSION
# ============================================================

def run_financial_conditions_regression(df):

    explanatory_variables = [
        "Interest_Rate_Change_Ppt",
        "FX_Change_Pct",
    ]

    return run_ols_regression(
        df,
        "GDP_Growth_Pct",
        explanatory_variables,
    )


# ============================================================
# REGRESSION SUMMARY
# ============================================================

def extract_regression_summary(
    regression_result,
    model_name,
):

    if regression_result[
        "status"
    ] != "SUCCESS":

        return {
            "Model": model_name,
            "Status": regression_result[
                "status"
            ],
            "Observations": (
                len(
                    regression_result["data"]
                )
                if regression_result["data"]
                is not None
                else 0
            ),
            "R_Squared": np.nan,
            "Adjusted_R_Squared": np.nan,
            "F_Statistic": np.nan,
            "P_Value_F": np.nan,
        }

    model = regression_result[
        "model"
    ]

    return {
        "Model": model_name,
        "Status": "SUCCESS",
        "Observations": int(
            model.nobs
        ),
        "R_Squared": model.rsquared,
        "Adjusted_R_Squared": (
            model.rsquared_adj
        ),
        "F_Statistic": model.fvalue,
        "P_Value_F": model.f_pvalue,
    }


# ============================================================
# REGRESSION COEFFICIENTS
# ============================================================

def extract_regression_coefficients(
    regression_result,
    model_name,
):

    if regression_result[
        "status"
    ] != "SUCCESS":

        return pd.DataFrame(
            columns=[
                "Model",
                "Variable",
                "Coefficient",
                "Std_Error",
                "T_Statistic",
                "P_Value",
            ]
        )

    model = regression_result[
        "model"
    ]

    records = []

    for variable in model.params.index:

        records.append(
            {
                "Model": model_name,
                "Variable": variable,
                "Coefficient": model.params[
                    variable
                ],
                "Std_Error": model.bse[
                    variable
                ],
                "T_Statistic": model.tvalues[
                    variable
                ],
                "P_Value": model.pvalues[
                    variable
                ],
            }
        )

    return pd.DataFrame(records)


# ============================================================
# LAGGED GDP RELATIONSHIPS
# ============================================================

def calculate_lag_relationships(df):

    data = df.copy()

    data[
        "GDP_Growth_Lag1"
    ] = data[
        "GDP_Growth_Pct"
    ].shift(1)

    data[
        "Domestic_Demand_Growth_Lag1"
    ] = data[
        "Domestic_Demand_Growth_Pct"
    ].shift(1)

    data[
        "Investment_Growth_Lag1"
    ] = data[
        "Investment_Growth_Pct"
    ].shift(1)

    data[
        "Interest_Rate_Change_Lag1"
    ] = data[
        "Interest_Rate_Change_Ppt"
    ].shift(1)

    lag_correlations = []

    lag_variables = [
        "GDP_Growth_Lag1",
        "Domestic_Demand_Growth_Lag1",
        "Investment_Growth_Lag1",
        "Interest_Rate_Change_Lag1",
    ]

    for variable in lag_variables:

        valid = data[
            [
                "GDP_Growth_Pct",
                variable,
            ]
        ].dropna()

        if len(valid) >= 3:

            correlation = valid[
                "GDP_Growth_Pct"
            ].corr(
                valid[variable]
            )

        else:
            correlation = np.nan

        lag_correlations.append(
            {
                "Lagged_Variable": variable,
                "Correlation_With_Current_GDP_Growth":
                    correlation,
                "Observations": len(valid),
            }
        )

    return (
        data,
        pd.DataFrame(
            lag_correlations
        )
    )


# ============================================================
# BUSINESS CYCLE ANALYSIS
# ============================================================

def calculate_business_cycle_econometrics(
    df
):

    data = df.copy()

    data[
        "GDP_Growth_Acceleration"
    ] = data[
        "GDP_Growth_Pct"
    ].diff()

    data[
        "Domestic_Demand_Acceleration"
    ] = data[
        "Domestic_Demand_Growth_Pct"
    ].diff()

    data[
        "Investment_Growth_Acceleration"
    ] = data[
        "Investment_Growth_Pct"
    ].diff()

    data[
        "Consumption_Growth_Acceleration"
    ] = data[
        "Consumption_Growth_Pct"
    ].diff()

    return data


# ============================================================
# ECONOMETRIC MASTER PIPELINE
# ============================================================

def run_econometric_analysis(df):

    data = df.copy()

    sample_assessment = (
        assess_sample_size(data)
    )

    correlation_matrix = (
        calculate_correlation_matrix(
            data
        )
    )

    gdp_correlations = (
        calculate_gdp_correlations(
            data
        )
    )

    gdp_correlations = (
        add_correlation_classification(
            gdp_correlations
        )
    )

    simple_regression = (
        run_simple_gdp_regression(
            data
        )
    )

    component_regression = (
        run_component_regression(
            data
        )
    )

    financial_regression = (
        run_financial_conditions_regression(
            data
        )
    )

    lagged_data, lag_relationships = (
        calculate_lag_relationships(
            data
        )
    )

    business_cycle_data = (
        calculate_business_cycle_econometrics(
            lagged_data
        )
    )

    regression_summary = pd.DataFrame(
        [
            extract_regression_summary(
                simple_regression,
                "GDP vs Domestic Demand",
            ),
            extract_regression_summary(
                component_regression,
                "GDP vs Expenditure Components",
            ),
            extract_regression_summary(
                financial_regression,
                "GDP vs Financial Conditions",
            ),
        ]
    )

    regression_coefficients = pd.concat(
        [
            extract_regression_coefficients(
                simple_regression,
                "GDP vs Domestic Demand",
            ),
            extract_regression_coefficients(
                component_regression,
                "GDP vs Expenditure Components",
            ),
            extract_regression_coefficients(
                financial_regression,
                "GDP vs Financial Conditions",
            ),
        ],
        ignore_index=True,
    )

    return {
        "data": business_cycle_data,
        "sample_assessment": sample_assessment,
        "correlation_matrix": correlation_matrix,
        "gdp_correlations": gdp_correlations,
        "lag_relationships": lag_relationships,
        "regression_summary": regression_summary,
        "regression_coefficients":
            regression_coefficients,
        "simple_regression":
            simple_regression,
        "component_regression":
            component_regression,
        "financial_regression":
            financial_regression,
    }


# ============================================================
# PROFESSIONAL ECONOMETRIC REPORT
# ============================================================

def print_econometric_report(
    results
):

    sample = results[
        "sample_assessment"
    ]

    print()
    print("=" * 80)
    print("PHASE 05 — ECONOMETRIC ANALYSIS")
    print("=" * 80)

    print()
    print("SAMPLE SIZE ASSESSMENT")
    print("-" * 80)

    print(
        f"Observations: "
        f"{sample['observations']}"
    )

    print(
        f"Assessment: "
        f"{sample['status']}"
    )

    print(
        f"Warning: "
        f"{sample['warning']}"
    )

    print()
    print("GDP GROWTH CORRELATIONS")
    print("-" * 80)

    print(
        results[
            "gdp_correlations"
        ].to_string(
            index=False
        )
    )

    print()
    print("LAGGED RELATIONSHIPS")
    print("-" * 80)

    print(
        results[
            "lag_relationships"
        ].to_string(
            index=False
        )
    )

    print()
    print("REGRESSION SUMMARY")
    print("-" * 80)

    print(
        results[
            "regression_summary"
        ].to_string(
            index=False
        )
    )

    print()
    print("REGRESSION COEFFICIENTS")
    print("-" * 80)

    if results[
        "regression_coefficients"
    ].empty:

        print(
            "No regression coefficients available."
        )

    else:

        print(
            results[
                "regression_coefficients"
            ].to_string(
                index=False
            )
        )

    print()
    print("INTERPRETATION WARNING")
    print("-" * 80)

    print(
        "Correlation measures association, "
        "not causation."
    )

    print(
        "The current dataset contains only "
        f"{sample['observations']} annual observations."
    )

    print(
        "Regression estimates are therefore "
        "exploratory and should not be interpreted "
        "as robust causal estimates."
    )

    print()
    print("=" * 80)
    print("PHASE 05 ECONOMETRIC REPORT COMPLETE")
    print("=" * 80)


# ============================================================
# ECONOMETRIC QUALITY CONTROL
# ============================================================

def run_econometric_quality_checks(
    results
):

    sample = results[
        "sample_assessment"
    ]

    checks = {
        "Sample_Size_Assessment":
            sample["status"],

        "Correlation_Analysis":
            (
                "PASS"
                if not results[
                    "gdp_correlations"
                ].empty
                else "CHECK"
            ),

        "Lag_Analysis":
            (
                "PASS"
                if not results[
                    "lag_relationships"
                ].empty
                else "CHECK"
            ),

        "Regression_Analysis":
            (
                "AVAILABLE"
                if STATSMODELS_AVAILABLE
                else "UNAVAILABLE"
            ),
    }

    return checks


def print_econometric_quality_report(
    checks
):

    print()
    print("=" * 80)
    print("PHASE 05 QUALITY CONTROL")
    print("=" * 80)

    print()

    for name, status in checks.items():

        print(
            f"{name}: {status}"
        )

    print()
    print("=" * 80)