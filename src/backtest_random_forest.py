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


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )


def load_backtest_data():
    # Load full merged dataset
    data = merge_all_data()

    # Use Date as index
    data = data.set_index("Date")

    # Build technical features,
    # target AND Future_Return_3D
    data = build_features(data)

    # Keep the actual forward return because
    # the backtest needs the size of the move,
    # not only whether it was up or down.
    columns_needed = (
        FEATURES
        + [
            "Target_3D",
            "Future_Return_3D",
        ]
    )

    model_data = data[
        columns_needed
    ].dropna()

    return model_data


def generate_walk_forward_signals(
    model_data
):
    all_signals = []

    for year in range(
        2019,
        2026,
    ):
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

        signals[
            "Prob_Up"
        ] = prob_up

        signals[
            "Prediction"
        ] = prediction

        signals[
            "Confidence"
        ] = confidence

        signals = signals[
            signals[
                "Confidence"
            ] >= CONFIDENCE_THRESHOLD
        ].copy()

        all_signals.append(
            signals
        )

    return pd.concat(
        all_signals
    ).sort_index()


def calculate_trade_returns(
    signals
):
    signals = signals.copy()

    # Future_Return_3D was calculated
    # BEFORE we filtered to trading signals.
    #
    # Therefore this is genuinely the
    # next 3 trading-day return.
    signals[
        "Strategy_Return"
    ] = np.where(
        signals[
            "Prediction"
        ] == 1,

        # Long trade
        signals[
            "Future_Return_3D"
        ],

        # Short trade
        -signals[
            "Future_Return_3D"
        ],
    )

    return signals


def summarize_trades(
    trades,
    name,
):
    trades = trades.dropna(
        subset=[
            "Strategy_Return"
        ]
    )

    number_of_trades = len(
        trades
    )

    if number_of_trades == 0:
        return {
            "Strategy": name,
            "Trades": 0,
            "Win_Rate": np.nan,
            "Avg_Trade_Return": np.nan,
            "Median_Trade_Return": np.nan,
            "Simple_Return_Sum": np.nan,
        }

    win_rate = (
        trades[
            "Strategy_Return"
        ] > 0
    ).mean()

    average_return = trades[
        "Strategy_Return"
    ].mean()

    median_return = trades[
        "Strategy_Return"
    ].median()

    simple_return_sum = trades[
        "Strategy_Return"
    ].sum()

    return {
        "Strategy": name,
        "Trades": number_of_trades,
        "Win_Rate": win_rate,
        "Avg_Trade_Return": average_return,
        "Median_Trade_Return": median_return,
        "Simple_Return_Sum": simple_return_sum,
    }


if __name__ == "__main__":
    model_data = (
        load_backtest_data()
    )

    signals = (
        generate_walk_forward_signals(
            model_data
        )
    )

    signals = (
        calculate_trade_returns(
            signals
        )
    )

    up_trades = signals[
        signals[
            "Prediction"
        ] == 1
    ]

    down_trades = signals[
        signals[
            "Prediction"
        ] == 0
    ]

    summary = pd.DataFrame([
        summarize_trades(
            signals,
            "All Signals",
        ),

        summarize_trades(
            up_trades,
            "UP Only",
        ),

        summarize_trades(
            down_trades,
            "DOWN Only",
        ),
    ])

    print(
        "\nBacktest summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ----------------------------------------
    # Year-by-year results
    # ----------------------------------------

    yearly_results = []

    for year in range(
        2019,
        2026,
    ):
        yearly_trades = signals[
            signals.index.year
            == year
        ]

        yearly_results.append(
            summarize_trades(
                yearly_trades,
                str(year),
            )
        )

    print(
        "\nYear-by-year:"
    )

    print(
        pd.DataFrame(
            yearly_results
        ).to_string(
            index=False
        )
    )
