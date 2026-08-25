import pandas as pd
import matplotlib.pyplot as plt


def plot_natural_gas_prices():
    data = pd.read_csv(
        "data/ng_futures.csv",
        index_col="Date",
        parse_dates=True
    )

    plt.figure(figsize=(12, 6))

    plt.plot(data.index, data["Close"])

    plt.title("Natural Gas Futures Price (2010–Present)")
    plt.xlabel("Date")
    plt.ylabel("Price ($/MMBtu)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_natural_gas_prices()
