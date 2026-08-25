import yfinance as yf


def fetch_natural_gas_data():
    data = yf.download(
        "NG=F",
        start="2010-01-01",
        auto_adjust=False
    )
    data.columns = data.columns.get_level_values(0)

    return data


if __name__ == "__main__":
    ng_data = fetch_natural_gas_data()
    ng_data.to_csv("data/ng_futures.csv")

    print(ng_data.head())
    print(ng_data.tail())

    print("\nDataset shape:")
    print(ng_data.shape)

    print("\nMissing values:")
    print(ng_data.isna().sum())

    print("\nColumn names:")
    print(ng_data.columns)
