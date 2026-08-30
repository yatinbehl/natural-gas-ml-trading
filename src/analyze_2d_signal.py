import pandas as pd
import numpy as np

from test_prediction_horizons import (
    generate_predictions,
)

from backtest_portfolio import (
    load_backtest_data,
)


HORIZON = 2

THRESHOLDS = [
    0.50,
    0.525,
    0.55,
    0.575,
    0.60,
    0.625,
    0.65,
]


def calculate_stats(data):

    if len(data) == 0:

        return {
            "Signals": 0,
            "Accuracy": np.nan,
            "Avg_Confidence": np.nan,
            "Avg_Strategy_Return": np.nan,
        }

    correct = (
        data["Prediction"]
        == data["Actual"]
    )

    accuracy = correct.mean()

    # Directional strategy return:
    #
    # Prediction 1 = long
    # Prediction 0 = short
    #
    strategy_return = np.where(
        data["Prediction"] == 1,
        data["Future_Return"],
        -data["Future_Return"],
    )

    return {
        "Signals": len(data),
        "Accuracy": accuracy,
        "Avg_Confidence": data[
            "Confidence"
        ].mean(),
        "Avg_Strategy_Return": (
            strategy_return.mean()
        ),
    }


if __name__ == "__main__":

    print("\nLoading data...")

    data = load_backtest_data()

    print(
        "Generating purged "
        "walk-forward 2-day predictions..."
    )

    predictions = generate_predictions(
        data=data,
        horizon=HORIZON,
    )

    print(
        f"\nTotal predictions: "
        f"{len(predictions)}"
    )

    # -------------------------------------------------
    # CONFIDENCE CURVE
    # -------------------------------------------------

    confidence_results = []

    for threshold in THRESHOLDS:

        selected = predictions[
            predictions["Confidence"]
            >= threshold
        ]

        stats = calculate_stats(
            selected
        )

        confidence_results.append(
            {
                "Threshold": threshold,
                **stats,
            }
        )

    confidence_df = pd.DataFrame(
        confidence_results
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "2-DAY CONFIDENCE CURVE"
    )

    print(
        "=" * 70
    )

    print(
        confidence_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # UP VS DOWN
    # -------------------------------------------------

    direction_results = []

    for threshold in THRESHOLDS:

        selected = predictions[
            predictions["Confidence"]
            >= threshold
        ]

        for prediction_value, direction in [
            (1, "UP"),
            (0, "DOWN"),
        ]:

            subset = selected[
                selected["Prediction"]
                == prediction_value
            ]

            stats = calculate_stats(
                subset
            )

            direction_results.append(
                {
                    "Threshold": threshold,
                    "Direction": direction,
                    **stats,
                }
            )

    direction_df = pd.DataFrame(
        direction_results
    )

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "UP VS DOWN"
    )

    print(
        "=" * 70
    )

    print(
        direction_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # YEAR-BY-YEAR AT 0.60
    # -------------------------------------------------

    threshold = 0.60

    selected = predictions[
        predictions["Confidence"]
        >= threshold
    ]

    yearly_results = []

    for year in sorted(
        selected["Year"].unique()
    ):

        year_data = selected[
            selected["Year"] == year
        ]

        for prediction_value, direction in [
            (1, "UP"),
            (0, "DOWN"),
        ]:

            subset = year_data[
                year_data["Prediction"]
                == prediction_value
            ]

            stats = calculate_stats(
                subset
            )

            yearly_results.append(
                {
                    "Year": year,
                    "Direction": direction,
                    **stats,
                }
            )

    yearly_df = pd.DataFrame(
        yearly_results
    )

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "YEAR-BY-YEAR AT 0.60 CONFIDENCE"
    )

    print(
        "=" * 70
    )

    print(
        yearly_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # SIMPLE MATRICES
    # -------------------------------------------------

    print(
        "\n\nACCURACY BY DIRECTION:"
    )

    accuracy_matrix = (
        direction_df.pivot(
            index="Threshold",
            columns="Direction",
            values="Accuracy",
        )
    )

    print(
        accuracy_matrix.to_string()
    )

    print(
        "\n\nSIGNAL COUNT BY DIRECTION:"
    )

    signal_matrix = (
        direction_df.pivot(
            index="Threshold",
            columns="Direction",
            values="Signals",
        )
    )

    print(
        signal_matrix.to_string()
    )
