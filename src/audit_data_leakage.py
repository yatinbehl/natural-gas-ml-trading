import pandas as pd

from build_features import build_features
from merge_all_data import merge_all_data


def audit_dates(data):
    print("\n" + "=" * 70)
    print("1. DATE ORDER CHECK")
    print("=" * 70)

    is_sorted = data.index.is_monotonic_increasing

    print("Dates sorted ascending:")
    print(is_sorted)

    print("\nDuplicate market dates:")
    print(data.index.duplicated().sum())


def audit_storage(data):
    print("\n" + "=" * 70)
    print("2. STORAGE POINT-IN-TIME CHECK")
    print("=" * 70)

    storage_available_column = None

    if "Available_Date_x" in data.columns:
        storage_available_column = "Available_Date_x"

    elif "Available_Date" in data.columns:
        storage_available_column = "Available_Date"

    if storage_available_column is None:
        print(
            "Could not identify storage Available_Date column."
        )
        return

    data[
        storage_available_column
    ] = pd.to_datetime(
        data[
            storage_available_column
        ]
    )

    violations = data[
        data[
            storage_available_column
        ] > data.index
    ]

    print(
        "Storage rows using information "
        "from the future:"
    )
    print(
        len(violations)
    )

    if not violations.empty:
        print(
            violations[
                [
                    "Storage_Bcf",
                    "Weekly_Change_Bcf",
                    storage_available_column,
                ]
            ].head(20)
        )

    print(
        "\nSample recent storage timing:"
    )

    sample_columns = [
        "Storage_Bcf",
        "Weekly_Change_Bcf",
        storage_available_column,
    ]

    available_columns = [
        column
        for column in sample_columns
        if column in data.columns
    ]

    print(
        data[
            available_columns
        ].tail(10)
    )


def audit_forecasts(data):
    print("\n" + "=" * 70)
    print("3. FORECAST POINT-IN-TIME CHECK")
    print("=" * 70)

    forecast_available_column = None

    if "Available_Date_y" in data.columns:
        forecast_available_column = (
            "Available_Date_y"
        )

    elif (
        "Available_Date"
        in data.columns
    ):
        forecast_available_column = (
            "Available_Date"
        )

    if forecast_available_column is None:
        print(
            "Could not identify forecast Available_Date column."
        )
        return

    data[
        forecast_available_column
    ] = pd.to_datetime(
        data[
            forecast_available_column
        ]
    )

    data[
        "Forecast_Date"
    ] = pd.to_datetime(
        data[
            "Forecast_Date"
        ]
    )

    future_available = data[
        data[
            forecast_available_column
        ] > data.index
    ]

    future_forecast_date = data[
        data[
            "Forecast_Date"
        ] >= data.index
    ]

    print(
        "Rows using forecast before "
        "its Available_Date:"
    )
    print(
        len(future_available)
    )

    print(
        "\nRows where Forecast_Date is "
        "same day or later than market date:"
    )
    print(
        len(future_forecast_date)
    )

    print(
        "\nSample recent forecast timing:"
    )

    print(
        data[
            [
                "Forecast_Date",
                forecast_available_column,
                "Forecast_HDD_7D",
                "Forecast_CDD_7D",
            ]
        ].tail(10)
    )


def audit_weather(data):
    print("\n" + "=" * 70)
    print("4. REALIZED WEATHER CHECK")
    print("=" * 70)

    weather = pd.read_csv(
        "data/ng_weather_daily.csv",
        parse_dates=["Date"]
    )

    weather = weather.sort_values(
        "Date"
    ).set_index(
        "Date"
    )

    # Expected values using ONLY prior-day information
    expected_hdd_lag = (
        weather["HDD"].shift(1)
    )

    expected_cdd_lag = (
        weather["CDD"].shift(1)
    )

    expected_hdd_7d = (
        weather["HDD"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    expected_cdd_7d = (
        weather["CDD"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    expected = pd.DataFrame({
        "Expected_HDD_1D_Lag": expected_hdd_lag,
        "Expected_CDD_1D_Lag": expected_cdd_lag,
        "Expected_HDD_7D_Avg": expected_hdd_7d,
        "Expected_CDD_7D_Avg": expected_cdd_7d,
    })

    audit_data = data.join(
        expected,
        how="left"
    )

    hdd_lag_diff = (
        audit_data["HDD_1D_Lag"]
        - audit_data["Expected_HDD_1D_Lag"]
    ).abs()

    cdd_lag_diff = (
        audit_data["CDD_1D_Lag"]
        - audit_data["Expected_CDD_1D_Lag"]
    ).abs()

    hdd_7d_diff = (
        audit_data["HDD_7D_Avg"]
        - audit_data["Expected_HDD_7D_Avg"]
    ).abs()

    cdd_7d_diff = (
        audit_data["CDD_7D_Avg"]
        - audit_data["Expected_CDD_7D_Avg"]
    ).abs()

    print(
        "Maximum HDD 1-day lag difference:"
    )
    print(
        hdd_lag_diff.max()
    )

    print(
        "\nMaximum CDD 1-day lag difference:"
    )
    print(
        cdd_lag_diff.max()
    )

    print(
        "\nMaximum HDD 7-day average difference:"
    )
    print(
        hdd_7d_diff.max()
    )

    print(
        "\nMaximum CDD 7-day average difference:"
    )
    print(
        cdd_7d_diff.max()
    )

    print(
        "\nRecent realized weather audit sample:"
    )

    print(
        audit_data[
            [
                "HDD_1D_Lag",
                "Expected_HDD_1D_Lag",
                "CDD_1D_Lag",
                "Expected_CDD_1D_Lag",
                "HDD_7D_Avg",
                "Expected_HDD_7D_Avg",
                "CDD_7D_Avg",
                "Expected_CDD_7D_Avg",
            ]
        ].tail(10)
    )
def audit_target(data):
    print("\n" + "=" * 70)
    print("5. TARGET CONSTRUCTION CHECK")
    print("=" * 70)

    target_data = build_features(
        data.copy()
    )

    # ----------------------------------------
    # Check Future_Return_3D calculation
    # ----------------------------------------

    manual_future_return = (
        target_data["Close"].shift(-3)
        / target_data["Close"]
        - 1
    )

    difference = (
        target_data["Future_Return_3D"]
        - manual_future_return
    ).abs()

    print(
        "Maximum difference between "
        "stored Future_Return_3D and "
        "manual calculation:"
    )

    print(
        difference.max()
    )

    # ----------------------------------------
    # Check Target_3D direction
    # ----------------------------------------

    valid_rows = target_data[
        target_data["Target_3D"].notna()
        & target_data["Future_Return_3D"].notna()
    ].copy()

    expected_target = (
        valid_rows["Future_Return_3D"] > 0
    ).astype(int)

    actual_target = (
        valid_rows["Target_3D"]
        .astype(int)
    )

    target_mismatches = (
        expected_target
        != actual_target
    )

    print(
        "\nTarget direction mismatches:"
    )

    print(
        target_mismatches.sum()
    )

    print(
        "\nRows checked:"
    )

    print(
        len(valid_rows)
    )

    print(
        "\nLast rows of target data:"
    )

    print(
        target_data[
            [
                "Close",
                "Future_Return_3D",
                "Target_3D",
            ]
        ].tail(10)
    )

def audit_feature_missingness(data):
    print("\n" + "=" * 70)
    print("6. FEATURE MISSINGNESS CHECK")
    print("=" * 70)

    columns = [
        "Weekly_Change_Bcf",
        "Storage_vs_5Y_Avg_Pct",
        "Storage_Surplus_Change",
        "HDD_1D_Lag",
        "CDD_1D_Lag",
        "HDD_7D_Avg",
        "CDD_7D_Avg",
        "Forecast_HDD_7D",
        "Forecast_CDD_7D",
        "HDD_7D_Outlook_Change",
        "CDD_7D_Outlook_Change",
    ]

    available_columns = [
        column
        for column in columns
        if column in data.columns
    ]

    print(
        data[
            available_columns
        ].isna().sum()
    )


if __name__ == "__main__":
    data = merge_all_data()

    data[
        "Date"
    ] = pd.to_datetime(
        data[
            "Date"
        ]
    )

    data = (
        data
        .sort_values(
            "Date"
        )
        .set_index(
            "Date"
        )
    )

    audit_dates(
        data
    )

    audit_storage(
        data.copy()
    )

    audit_forecasts(
        data.copy()
    )

    audit_weather(
        data.copy()
    )

    audit_target(
        data.copy()
    )

    audit_feature_missingness(
        data.copy()
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "LEAKAGE AUDIT COMPLETE"
    )
    print(
        "=" * 70
    )
