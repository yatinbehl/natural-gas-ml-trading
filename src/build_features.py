import numpy as np
import pandas as pd


def build_features(data):
    data = data.copy()

    # Historical features
    data["Daily_Return"] = data["Close"].pct_change()

    data["Volatility_20D"] = (
        data["Daily_Return"]
        .rolling(window=20)
        .std()
    )

    data["Momentum_5D"] = data["Close"].pct_change(periods=5)
    data["Momentum_20D"] = data["Close"].pct_change(periods=20)

    # Calendar / seasonality features
    data["Day_of_Year"] = data.index.dayofyear

    data["Season_Sin"] = np.sin(
        2 * np.pi * data["Day_of_Year"] / 365.25
    )

    data["Season_Cos"] = np.cos(
        2 * np.pi * data["Day_of_Year"] / 365.25
    )

    # Future 3-day return
    data["Future_Return_3D"] = (
        data["Close"].shift(-3) / data["Close"] - 1
    )

    # Target: 1 = price rises, 0 = price falls
    data["Target_3D"] = pd.NA

    valid_target = data["Future_Return_3D"].notna()

    data.loc[valid_target, "Target_3D"] = (
        data.loc[valid_target, "Future_Return_3D"] > 0
    ).astype(int)

    return data


if __name__ == "__main__":
    ng_data = pd.read_csv(
        "data/ng_futures.csv",
        index_col="Date",
        parse_dates=True
    )

    featured_data = build_features(ng_data)

    print(featured_data.tail())
    print("\nTarget distribution:")
    print(featured_data["Target_3D"].value_counts())

    print("\nTarget percentages:")
    print(
        featured_data["Target_3D"]
        .value_counts(normalize=True) * 100
    )
