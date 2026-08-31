import pandas as pd
import numpy as np

from backtest_portfolio import load_backtest_data
from test_prediction_horizons import generate_predictions


HORIZON = 2
CONFIDENCE_THRESHOLD = 0.60
TRANSACTION_COST = 0.001
POSITION_SIZE = 0.25
STARTING_CAPITAL = 10000.0


if __name__ == "__main__":

    print("\nLoading data...")

    data = load_backtest_data()

    print(
        "Generating frozen 2-day "
        "walk-forward predictions..."
    )

    predictions = generate_predictions(
        data=data,
        horizon=HORIZON,
    )

    # -----------------------------------------------
    # 2025 ONLY
    # -----------------------------------------------

    holdout = predictions[
        predictions["Year"] == 2025
    ].copy()

    high_conf = holdout[
        holdout["Confidence"]
        >= CONFIDENCE_THRESHOLD
    ].copy()

    print("\n" + "=" * 70)
    print("2025 PSEUDO-HOLDOUT")
    print("=" * 70)

    print(
        f"\nAll 2025 predictions: "
        f"{len(holdout)}"
    )

    all_accuracy = (
        holdout["Prediction"]
        == holdout["Actual"]
    ).mean()

    print(
        f"All prediction accuracy: "
        f"{all_accuracy:.4f}"
    )

    print(
        f"\nHigh-confidence signals: "
        f"{len(high_conf)}"
    )

    high_accuracy = (
        high_conf["Prediction"]
        == high_conf["Actual"]
    ).mean()

    print(
        f"High-confidence accuracy: "
        f"{high_accuracy:.4f}"
    )

    # -----------------------------------------------
    # SHOW EVERY HIGH-CONFIDENCE SIGNAL
    # -----------------------------------------------

    high_conf["Direction"] = np.where(
        high_conf["Prediction"] == 1,
        "UP",
        "DOWN",
    )

    high_conf["Correct"] = (
        high_conf["Prediction"]
        == high_conf["Actual"]
    )

    high_conf["Strategy_Return"] = np.where(
        high_conf["Prediction"] == 1,
        high_conf["Future_Return"],
        -high_conf["Future_Return"],
    )

    high_conf["Net_Return"] = (
        high_conf["Strategy_Return"]
        - TRANSACTION_COST
    )

    columns = [
        "Entry_Date",
        "Exit_Date",
        "Direction",
        "Confidence",
        "Future_Return",
        "Correct",
        "Net_Return",
    ]

    print(
        "\n\nHIGH-CONFIDENCE SIGNALS:"
    )

    print(
        high_conf[columns].to_string()
    )

    # -----------------------------------------------
    # UP / DOWN SUMMARY
    # -----------------------------------------------

    print(
        "\n\nDIRECTION SUMMARY:"
    )

    for direction in ["UP", "DOWN"]:

        subset = high_conf[
            high_conf["Direction"]
            == direction
        ]

        if subset.empty:
            continue

        accuracy = subset[
            "Correct"
        ].mean()

        avg_return = subset[
            "Net_Return"
        ].mean()

        print(
            f"\n{direction}"
        )

        print(
            f"Signals: {len(subset)}"
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Average net signal return: "
            f"{avg_return:.4f}"
        )

    # -----------------------------------------------
    # NON-OVERLAPPING PORTFOLIO
    # -----------------------------------------------

    trades = []

    last_exit_date = None

    for signal_date, row in high_conf.iterrows():

        entry_date = pd.Timestamp(
            row["Entry_Date"]
        )

        exit_date = pd.Timestamp(
            row["Exit_Date"]
        )

        if (
            last_exit_date is not None
            and entry_date <= last_exit_date
        ):
            continue

        trades.append(row)

        last_exit_date = exit_date

    trades = pd.DataFrame(trades)

    equity = STARTING_CAPITAL
    equity_curve = []

    if not trades.empty:

        for trade_return in trades[
            "Net_Return"
        ]:

            equity *= (
                1
                + POSITION_SIZE
                * trade_return
            )

            equity_curve.append(equity)

        equity_series = pd.Series(
            equity_curve
        )

        running_max = (
            equity_series.cummax()
        )

        drawdown = (
            equity_series
            / running_max
            - 1
        )

        max_drawdown = drawdown.min()

    else:

        max_drawdown = np.nan

    total_return = (
        equity / STARTING_CAPITAL - 1
    )

    print(
        "\n\n2025 NON-OVERLAPPING PORTFOLIO:"
    )

    print(
        f"Trades: {len(trades)}"
    )

    print(
        f"Starting capital: "
        f"${STARTING_CAPITAL:,.2f}"
    )

    print(
        f"Final equity: "
        f"${equity:,.2f}"
    )

    print(
        f"Return: "
        f"{total_return:.2%}"
    )

    print(
        f"Max drawdown: "
        f"{max_drawdown:.2%}"
    )
