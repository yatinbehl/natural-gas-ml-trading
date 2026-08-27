import pandas as pd

from merge_market_storage import merge_market_and_storage
from build_weather_features import build_weather_features


def merge_all_data():
    # Price + point-in-time storage data
    data = merge_market_and_storage()

    # Weather data
    weather = pd.read_csv(
        "data/ng_weather_daily.csv"
    )

    weather = build_weather_features(weather)

    weather_features = weather[
        [
            "Date",
            "HDD_1D_Lag",
            "CDD_1D_Lag",
            "HDD_7D_Avg",
            "CDD_7D_Avg",
        ]
    ]

    data["Date"] = pd.to_datetime(data["Date"])
    weather_features = weather_features.copy()
    weather_features["Date"] = pd.to_datetime(
        weather_features["Date"]
    )

    data = pd.merge(
        data,
        weather_features,
        on="Date",
        how="left"
    )

    return data


if __name__ == "__main__":
    data = merge_all_data()

    print(
        data[
            [
                "Date",
                "Close",
                "Storage_Bcf",
                "Storage_vs_5Y_Avg_Pct",
                "HDD_1D_Lag",
                "CDD_1D_Lag",
                "HDD_7D_Avg",
                "CDD_7D_Avg",
            ]
        ].tail(15)
    )

    print("\nShape:")
    print(data.shape)

    print("\nMissing weather values:")
    print(
        data[
            [
                "HDD_1D_Lag",
                "CDD_1D_Lag",
                "HDD_7D_Avg",
                "CDD_7D_Avg",
            ]
        ].isna().sum()
    )
