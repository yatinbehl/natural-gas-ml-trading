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

    print("\n" + "=" * 70)
    print(f"Downloading repaired {name} ({ticker})")
    print("=" * 70)

    stock = yf.Ticker(ticker)

    data = stock.history(
        start=START_DATE,
        auto_adjust=True,
        actions=True,
        repair=True,
    )

    if data.empty:
        raise ValueError(
            f"No data downloaded for {ticker}"
        )

    data = data.reset_index()

    # Remove timezone if present
    data["Date"] = pd.to_datetime(
        data["Date"]
    ).dt.tz_localize(None)

    data["Daily_Return"] = (
        data["Close"].pct_change()
    )

    print(
        f"\nRows: {len(data)}"
    )

    print(
        f"Dates: "
        f"{data['Date'].min()} "
        f"to "
        f"{data['Date'].max()}"
    )

    print(
        "\nLargest positive returns:"
    )

    print(
        data.nlargest(
            10,
            "Daily_Return",
        )[
            [
                "Date",
                "Close",
                "Daily_Return",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nLargest negative returns:"
    )

    print(
        data.nsmallest(
            10,
            "Daily_Return",
        )[
            [
                "Date",
                "Close",
                "Daily_Return",
            ]
        ].to_string(
            index=False
        )
    )

    # Show corporate actions
    print(
        "\nRecorded stock splits / consolidations:"
    )

    if "Stock Splits" in data.columns:

        actions = data[
            data["Stock Splits"] != 0
        ]

        if len(actions) > 0:

            print(
                actions[
                    [
                        "Date",
                        "Close",
                        "Stock Splits",
                    ]
                ].to_string(
                    index=False
                )
            )

        else:
            print("None reported.")

    # Known HND consolidation check
    if name == "HND":

        print(
            "\nMay 2026 HND check:"
        )

        check = data[
            (
                data["Date"]
                >= "2026-05-20"
            )
            &
            (
                data["Date"]
                <= "2026-05-27"
            )
        ]

        columns = [
            "Date",
            "Open",
            "Close",
            "Daily_Return",
        ]

        if "Stock Splits" in check.columns:
            columns.append(
                "Stock Splits"
            )

        if "Repaired?" in check.columns:
            columns.append(
                "Repaired?"
            )

        print(
            check[
                columns
            ].to_string(
                index=False
            )
        )

    output_path = (
        DATA_DIR
        / f"{name.lower()}_adjusted.csv"
    )

    data.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved to {output_path}"
    )


if __name__ == "__main__":

    DATA_DIR.mkdir(
        exist_ok=True
    )

    for name, ticker in TICKERS.items():

        download_etf(
            name,
            ticker,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "REPAIRED ETF DOWNLOAD COMPLETE"
    )

    print(
        "=" * 70
    )
