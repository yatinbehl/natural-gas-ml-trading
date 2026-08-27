import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from build_features import build_features
from merge_all_data import merge_all_data
from train_baseline_model import (
    TECHNICAL_FEATURES,
    STORAGE_FEATURES,
    FORECAST_WEATHER_FEATURES,
)


FEATURES = (
    TECHNICAL_FEATURES
    + STORAGE_FEATURES
    + FORECAST_WEATHER_FEATURES
)

CONFIDENCE_THRESHOLD = 0.60

STARTING_CAPITAL = 10000.0

TRANSACTION_COST = 0.001

POSITION_SIZES = [
    1.00,
    0.50,
    0.25,
    0.10,
]


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )


def load_backtest_data():
    data = merge_all_data()

    data = data.set_index("Date")

    data = build_features(data)

    return data


def generate_walk_forward_signals(data):
    model_data = data[
        FEATURES
        + [
            "Target_3D",
            "Future_Return_3D",
        ]
    ].dropna()

    signal_frames = []

    for year in range(2019, 2026):

        train_data = model_data[
            model_data.index.year < year
        ]

        test_data = model_data[
            model_data.index.year == year
        ]

        X_train = train_data[
            FEATURES
        ]

        y_train = train_data[
            "Target_3D"
        ].astype(int)

        X_test = test_data[
            FEATURES
        ]

        model = make_random_forest()

        model.fit(
            X_train,
            y_train,
        )

        prob_up = model.predict_proba(
            X_test
        )[:, 1]

        prediction = (
            prob_up >= 0.50
        ).astype(int)

        confidence = np.maximum(
            prob_up,
            1 - prob_up,
        )

        signals = test_data.copy()

        signals["Prob_Up"] = prob_up
        signals["Prediction"] = prediction
        signals["Confidence"] = confidence

        signal_frames.append(
            signals
        )

    return pd.concat(
        signal_frames
    ).sort_index()


def select_non_overlapping_trades(
    signals,
    full_data,
    direction="all",
):
    trades = []

    trading_dates = list(
        full_data.index
    )

    signal_dates = set(
        signals.index
    )

    i = 0

    while i < len(trading_dates) - 3:

        entry_date = trading_dates[i]

        if entry_date not in signal_dates:
            i += 1
            continue

        row = signals.loc[
            entry_date
        ]

        if (
            row["Confidence"]
            < CONFIDENCE_THRESHOLD
        ):
            i += 1
            continue

        prediction = int(
            row["Prediction"]
        )

        if (
            direction == "up"
            and prediction != 1
        ):
            i += 1
            continue

        if (
            direction == "down"
            and prediction != 0
        ):
            i += 1
            continue

        exit_date = trading_dates[
            i + 3
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
            exit_price
            / entry_price
            - 1
        )

        if prediction == 1:
            gross_return = raw_return
            trade_direction = "UP"

        else:
            gross_return = -raw_return
            trade_direction = "DOWN"

        net_return = (
            gross_return
            - TRANSACTION_COST
        )

        trades.append({
            "Entry_Date": entry_date,
            "Exit_Date": exit_date,
            "Direction": trade_direction,
            "Confidence": row[
                "Confidence"
            ],
            "Entry_Price": entry_price,
            "Exit_Price": exit_price,
            "Gross_Return": gross_return,
            "Net_Return": net_return,
        })

        # Move to the first trading day
        # AFTER the current position closes.
        i += 4

    return pd.DataFrame(
        trades
    )


def calculate_max_drawdown(
    equity_series
):
    running_max = equity_series.cummax()

    drawdown = (
        equity_series
        / running_max
        - 1
    )

    return drawdown.min()


def run_portfolio(
    trades,
    position_size,
    name,
):
    trades = trades.copy()

    equity = STARTING_CAPITAL

    starting_equity = []
    ending_equity = []

    portfolio_returns = []

    for trade_return in trades[
        "Net_Return"
    ]:

        starting_equity.append(
            equity
        )

        portfolio_return = (
            position_size
            * trade_return
        )

        equity = equity * (
            1 + portfolio_return
        )

        portfolio_returns.append(
            portfolio_return
        )

        ending_equity.append(
            equity
        )

    trades[
        "Position_Size"
    ] = position_size

    trades[
        "Portfolio_Return"
    ] = portfolio_returns

    trades[
        "Starting_Equity"
    ] = starting_equity

    trades[
        "Ending_Equity"
    ] = ending_equity

    if trades.empty:
        return None

    total_return = (
        equity
        / STARTING_CAPITAL
        - 1
    )

    start_date = trades[
        "Entry_Date"
    ].min()

    end_date = trades[
        "Exit_Date"
    ].max()

    years = (
        end_date
        - start_date
    ).days / 365.25

    if years > 0:
        cagr = (
            equity
            / STARTING_CAPITAL
        ) ** (
            1 / years
        ) - 1
    else:
        cagr = np.nan

    max_drawdown = (
        calculate_max_drawdown(
            trades[
                "Ending_Equity"
            ]
        )
    )

    avg_return = trades[
        "Portfolio_Return"
    ].mean()

    std_return = trades[
        "Portfolio_Return"
    ].std()

    if (
        pd.isna(std_return)
        or std_return == 0
    ):
        sharpe_like = np.nan

    else:
        sharpe_like = (
            avg_return
            / std_return
            * np.sqrt(
                len(trades)
            )
        )

    win_rate = (
        trades[
            "Portfolio_Return"
        ] > 0
    ).mean()

    summary = {
        "Strategy": name,
        "Position_Size": position_size,
        "Trades": len(trades),
        "Win_Rate": win_rate,
        "Final_Equity": equity,
        "Total_Return": total_return,
        "CAGR": cagr,
        "Max_Drawdown": max_drawdown,
        "Sharpe_Like": sharpe_like,
    }

    return (
        trades,
        summary,
    )


if __name__ == "__main__":

    full_data = load_backtest_data()

    signals = generate_walk_forward_signals(
        full_data
    )

    strategy_configs = [
        (
            "all",
            "All Signals",
        ),
        (
            "up",
            "UP Only",
        ),
        (
            "down",
            "DOWN Only",
        ),
    ]

    summaries = []

    saved_up_trades = None

    for direction, strategy_name in strategy_configs:

        trades = select_non_overlapping_trades(
            signals,
            full_data,
            direction=direction,
        )

        for position_size in POSITION_SIZES:

            result = run_portfolio(
                trades,
                position_size,
                strategy_name,
            )

            if result is None:
                continue

            portfolio_trades, summary = result

            summaries.append(
                summary
            )

            if (
                strategy_name == "UP Only"
                and position_size == 0.25
            ):
                saved_up_trades = (
                    portfolio_trades.copy()
                )

    summary_df = pd.DataFrame(
        summaries
    )

    print(
        "\nPortfolio backtest summary:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    if saved_up_trades is not None:

        print(
            "\nExample UP-only trades "
            "at 25% position size:"
        )

        print(
            saved_up_trades[
                [
                    "Entry_Date",
                    "Exit_Date",
                    "Direction",
                    "Confidence",
                    "Entry_Price",
                    "Exit_Price",
                    "Net_Return",
                    "Starting_Equity",
                    "Ending_Equity",
                ]
            ].tail(10).to_string(
                index=False
            )
        )
