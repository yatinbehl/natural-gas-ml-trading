import pandas as pd


def merge_market_and_storage():
    market = pd.read_csv(
        "data/ng_futures.csv",
        index_col="Date",
        parse_dates=True
    )


    storage = pd.read_csv(
        "data/ng_storage_weekly.csv",
        parse_dates=[
            "Week_Ending_Date",
            "Release_Date",
            "Available_Date"
        ]
    )

    market = market.reset_index()

    market = market.sort_values("Date")
    storage = storage.sort_values("Available_Date")

    merged = pd.merge_asof(
        market,
        storage,
        left_on="Date",
        right_on="Available_Date",
        direction="backward"
    )
    return merged


if __name__ == "__main__":
    merged_data = merge_market_and_storage()

    print(merged_data.tail(15))

    print("\nShape:")
    print(merged_data.shape)
