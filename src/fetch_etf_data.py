import yfinance as yf
import pandas as pd
from pathlib import Path


START_DATE = "2014-01-01"

TICKERS = {
    "HNU": "HNU.TO",
    "HND": "HND.TO",
}

DATA_DIR = Path("data")


def download_etf(name, ticker):

    print(
        f"\nDownloading {name} ({ticker})..."
    )

    data = yf.download(
        ticker,
        start=START_DATE,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No data downloaded for {ticker}"
        )

    # yfinance can return MultiIndex columns
    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):
        data.columns = (
            data.columns
            .get_level_values(0)
        )

    data = data.reset_index()

    print(
        f"Rows downloaded: {len(data)}"
    )

    print(
        f"First date: {data['Date'].min()}"
    )

    print(
        f"Last date: {data['Date'].max()}"
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        data.head().to_string(
            index=False
        )
    )

    print(
        "\nLast 5 rows:"
    )

    print(
        data.tail().to_string(
            index=False
        )
    )

    output_path = (
        DATA_DIR
        / f"{name.lower()}_prices.csv"
    )

    data.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved to {output_path}"
    )

    return data


if __name__ == "__main__":

    DATA_DIR.mkdir(
        exist_ok=True
    )

    datasets = {}

    for name, ticker in TICKERS.items():

        datasets[name] = download_etf(
            name,
            ticker,
        )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "DOWNLOAD COMPLETE"
    )

    print(
        "=" * 60
    )

    for name, data in datasets.items():

        print(
            f"{name}: "
            f"{len(data)} rows, "
            f"{data['Date'].min()} "
            f"to {data['Date'].max()}"
        )
