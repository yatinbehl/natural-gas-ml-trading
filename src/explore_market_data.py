import pandas as pd
import matplotlib.pyplot as plt


def plot_natural_gas_prices():
    data = pd.read_csv(
        "data/ng_futures.csv",
        index_col="Date",
        parse_dates=True
    )

    data["Daily_Return"] = data["Close"].pct_change()

    data["Volatility_20D"] = data["Daily_Return"].rolling(window=20).std()

    plt.figure(figsize=(12, 6))

    plt.plot(data.index, data["Close"])

    plt.title("Natural Gas Futures Price (2010–Present)")
    plt.xlabel("Date")
    plt.ylabel("Price ($/MMBtu)")
    plt.grid(True)

    plt.tight_layout()
    
    print("\nReturn statistics:")

    print(data["Daily_Return"].describe())

    print("\n20-day volatility statistics:")
    print(data["Volatility_20D"].describe())


    plt.show()


    plt.figure(figsize=(12, 6))

    plt.plot(data.index, data["Volatility_20D"])

    plt.title("Natural Gas 20-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Daily Volatility")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_natural_gas_prices()
