import numpy as np
import pandas as pd

from backtest_portfolio import (
    load_backtest_data,
    generate_walk_forward_signals,
)


THRESHOLDS = [0.50, 0.55, 0.60, 0.625, 0.65]

TRANSACTION_COSTS = [
    0.000,
    0.001,
    0.0025,
    0.005,
]

POSITION_SIZE = 0.25
STARTING_CAPITAL = 10000.0
HOLDING_PERIOD = 3


def select_trades(
    signals,
    full_data,
    threshold,
    transaction_cost,
):
    trades = []

    trading_dates = list(full_data.index)
    signal_dates = set(signals.index)

    i = 0

    while i < len(trading_dates) - HOLDING_PERIOD:

        entry_date = trading_dates[i]

        if entry_date not in signal_dates:
            i += 1
            continue

        row = signals.loc[entry_date]

        if row["Confidence"] < threshold:
            i += 1
            continue

        prediction = int(row["Prediction"])

        exit_date = trading_dates[
            i + HOLDING_PERIOD
        ]

        entry_price = full_data.loc[
            entry_date,
            "Close"
        ]

        exit_price = full_data.loc[
            exit_date,
            "Close"
        ]

        raw_return = (
            exit_price / entry_price - 1
        )

        if prediction == 1:
            trade_return = raw_return
        else:
            trade_return = -raw_return

        net_return = (
            trade_return
            - transaction_cost
        )

        trades.append({
            "Entry_Date": entry_date,
            "Exit_Date": exit_date,
            "Prediction": prediction,
            "Confidence": row["Confidence"],
            "Net_Return": net_return,
        })

        # Prevent overlapping 3-day positions
        i += HOLDING_PERIOD + 1

    return pd.DataFrame(trades)


def calculate_results(
    trades,
    threshold,
    transaction_cost,
):
    if trades.empty:
        return {
            "Threshold": threshold,
            "Transaction_Cost": transaction_cost,
            "Trades": 0,
            "Win_Rate": np.nan,
            "Final_Equity": STARTING_CAPITAL,
            "Total_Return": 0,
            "CAGR": np.nan,
            "Max_Drawdown": np.nan,
        }

    equity = STARTING_CAPITAL
    equity_curve = []

    for trade_return in trades["Net_Return"]:

        portfolio_return = (
            POSITION_SIZE
            * trade_return
        )

        equity *= (
            1 + portfolio_return
        )

        equity_curve.append(equity)

    equity_series = pd.Series(
        equity_curve
    )

    running_max = equity_series.cummax()

    drawdown = (
        equity_series
        / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    total_return = (
        equity / STARTING_CAPITAL - 1
    )

    start_date = trades[
        "Entry_Date"
    ].min()

    end_date = trades[
        "Exit_Date"
    ].max()

    years = (
        end_date - start_date
    ).days / 365.25

    if years > 0:
        cagr = (
            equity / STARTING_CAPITAL
        ) ** (1 / years) - 1
    else:
        cagr = np.nan

    win_rate = (
        trades["Net_Return"] > 0
    ).mean()

    return {
        "Threshold": threshold,
        "Transaction_Cost": transaction_cost,
        "Trades": len(trades),
        "Win_Rate": win_rate,
        "Final_Equity": equity,
        "Total_Return": total_return,
        "CAGR": cagr,
        "Max_Drawdown": max_drawdown,
    }


if __name__ == "__main__":

    print(
        "\nLoading data and generating "
        "walk-forward predictions..."
    )

    full_data = load_backtest_data()

    signals = generate_walk_forward_signals(
        full_data
    )

    results = []

    for threshold in THRESHOLDS:

        for cost in TRANSACTION_COSTS:

            trades = select_trades(
                signals=signals,
                full_data=full_data,
                threshold=threshold,
                transaction_cost=cost,
            )

            result = calculate_results(
                trades=trades,
                threshold=threshold,
                transaction_cost=cost,
            )

            results.append(result)

    results_df = pd.DataFrame(
        results
    )

    print(
        "\nStress-test results:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nCAGR matrix:"
    )

    print(
        results_df.pivot(
            index="Threshold",
            columns="Transaction_Cost",
            values="CAGR",
        )
    )

    print(
        "\nWin-rate matrix:"
    )

    print(
        results_df.pivot(
            index="Threshold",
            columns="Transaction_Cost",
            values="Win_Rate",
        )
    )

    print(
        "\nTrade-count matrix:"
    )

    print(
        results_df.pivot(
            index="Threshold",
            columns="Transaction_Cost",
            values="Trades",
        )
    )
