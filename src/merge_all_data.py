import pandas as pd

from merge_market_storage import merge_market_and_storage
from build_weather_features import build_weather_features


def merge_all_data():
    # --------------------------------------------------
    # 1. Start with market + storage data
    # --------------------------------------------------
    data = merge_market_and_storage()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    # --------------------------------------------------
    # 2. Add realized weather features
    # --------------------------------------------------
    weather = pd.read_csv(
        "data/ng_weather_daily.csv"
    )

    weather = build_weather_features(
        weather
    )

    weather_features = weather[
        [
            "Date",
            "HDD_1D_Lag",
            "CDD_1D_Lag",
            "HDD_7D_Avg",
            "CDD_7D_Avg",
        ]
    ].copy()

    weather_features["Date"] = pd.to_datetime(
        weather_features["Date"]
    )

    data = pd.merge(
        data,
        weather_features,
        on="Date",
        how="left"
    )

    # --------------------------------------------------
    # 3. Add NOAA forecast-weather features
    # --------------------------------------------------
    forecasts = pd.read_csv(
        "data/ng_weather_forecasts.csv",
        parse_dates=[
            "Forecast_Date",
            "Available_Date",
        ]
    )

    forecast_features = forecasts[
        [
            "Forecast_Date",
            "Available_Date",
            "Forecast_HDD_7D",
            "Forecast_CDD_7D",
            "HDD_7D_Outlook_Change",
            "CDD_7D_Outlook_Change",
        ]
    ].copy()

    # merge_asof requires both sides to be sorted
    data = data.sort_values(
        "Date"
    )

    forecast_features = forecast_features.sort_values(
        "Available_Date"
    )

    # Use the most recent forecast that was
    # already available on each market date.
    data = pd.merge_asof(
        data,
        forecast_features,
        left_on="Date",
        right_on="Available_Date",
        direction="backward"
    )

    return data


if __name__ == "__main__":
    data = merge_all_data()

    columns_to_show = [
        "Date",
        "Close",
        "Storage_Bcf",
        "Storage_vs_5Y_Avg_Pct",
        "HDD_1D_Lag",
        "CDD_1D_Lag",
        "Forecast_Date",
        "Forecast_HDD_7D",
        "Forecast_CDD_7D",
        "HDD_7D_Outlook_Change",
        "CDD_7D_Outlook_Change",
    ]

    # Available_Date may be renamed because
    # storage already contains an Available_Date column.
    if "Available_Date_y" in data.columns:
        columns_to_show.insert(
            7,
            "Available_Date_y"
        )

    elif "Available_Date" in data.columns:
        columns_to_show.insert(
            7,
            "Available_Date"
        )

    print(
        data[
            columns_to_show
        ].tail(15)
    )

    print("\nShape:")
    print(
        data.shape
    )

    print("\nMissing forecast values:")
    print(
        data[
            [
                "Forecast_HDD_7D",
                "Forecast_CDD_7D",
                "HDD_7D_Outlook_Change",
                "CDD_7D_Outlook_Change",
            ]
        ].isna().sum()
    )
