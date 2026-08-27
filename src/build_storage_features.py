import pandas as pd


def build_storage_features(storage):
    storage = storage.copy()

    storage["Week_Ending_Date"] = pd.to_datetime(
        storage["Week_Ending_Date"]
    )

    storage = storage.sort_values("Week_Ending_Date")

    # Week number lets us compare similar times of year
    storage["Week"] = storage["Week_Ending_Date"].dt.isocalendar().week.astype(int)

    storage["Storage_5Y_Avg"] = (
        storage.groupby("Week")["Storage_Bcf"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=5,
                min_periods=3
            ).mean()
        )
    )

    storage["Storage_vs_5Y_Avg"] = (
        storage["Storage_Bcf"]
        - storage["Storage_5Y_Avg"]
    )

    storage["Storage_vs_5Y_Avg_Pct"] = (
        storage["Storage_vs_5Y_Avg"]
        / storage["Storage_5Y_Avg"]
    )

    storage["Storage_Surplus_Change"] = (
        storage["Storage_vs_5Y_Avg"].diff()
    )

    return storage


if __name__ == "__main__":
    storage = pd.read_csv(
        "data/ng_storage_weekly.csv"
    )

    storage = build_storage_features(storage)

    print(
        storage[
            [
                "Week_Ending_Date",
                "Storage_Bcf",
                "Storage_5Y_Avg",
                "Storage_vs_5Y_Avg",
                "Storage_vs_5Y_Avg_Pct",
                "Storage_Surplus_Change",
            ]
        ].tail(20)
    )
