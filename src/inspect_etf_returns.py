import pandas as pd


FILES = {
    "HNU": "data/hnu_prices.csv",
    "HND": "data/hnd_prices.csv",
}


for name, filepath in FILES.items():

    df = pd.read_csv(
        filepath,
        parse_dates=["Date"],
    )

    df = df.sort_values("Date")

    df["Daily_Return"] = (
        df["Close"].pct_change()
    )

    print(
        "\n"
        + "=" * 70
    )

    print(name)

    print(
        "=" * 70
    )

    print(
        "\nLargest positive daily returns:"
    )

    print(
        df.nlargest(
            15,
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
        "\nLargest negative daily returns:"
    )

    print(
        df.nsmallest(
            15,
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
