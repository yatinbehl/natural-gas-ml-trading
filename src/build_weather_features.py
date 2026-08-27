import pandas as pd


def build_weather_features(weather):
    weather = weather.copy()

    weather["Date"] = pd.to_datetime(weather["Date"])
    weather = weather.sort_values("Date")

    # Yesterday's realized weather
    weather["HDD_1D_Lag"] = weather["HDD"].shift(1)
    weather["CDD_1D_Lag"] = weather["CDD"].shift(1)

    # Average weather over the previous 7 days
    # shift(1) ensures today's realized weather is excluded
    weather["HDD_7D_Avg"] = (
        weather["HDD"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    weather["CDD_7D_Avg"] = (
        weather["CDD"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    return weather


if __name__ == "__main__":
    weather = pd.read_csv(
        "data/ng_weather_daily.csv"
    )

    weather = build_weather_features(weather)

    print(weather.tail(15))
