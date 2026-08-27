import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from train_baseline_model import (
    TECHNICAL_FEATURES,
    STORAGE_FEATURES,
    FORECAST_WEATHER_FEATURES,
    load_model_data,
)


FEATURES = (
    TECHNICAL_FEATURES
    + STORAGE_FEATURES
    + FORECAST_WEATHER_FEATURES
)


THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )


def evaluate_year(
    year,
    model_data,
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

    y_test = test_data[
        "Target_3D"
    ].astype(int)

    model = make_random_forest()

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )

    prob_up = probabilities[:, 1]

    predictions = (
        prob_up >= 0.50
    ).astype(int)

    confidence = pd.Series(
        [
            p if prediction == 1
            else 1 - p
            for p, prediction
            in zip(
                prob_up,
                predictions,
            )
        ],
        index=test_data.index,
    )

    results = []

    signal_rows = []

    for threshold in THRESHOLDS:
        mask = (
            confidence >= threshold
        )

        trades = int(
            mask.sum()
        )

        if trades == 0:
            accuracy = None
        else:
            accuracy = accuracy_score(
                y_test[mask],
                predictions[mask],
            )

        coverage = (
            trades
            / len(test_data)
        )

        results.append({
            "Year": year,
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Trades": trades,
            "Coverage": coverage,
        })

        selected_dates = test_data.index[
            mask
        ]

        for date in selected_dates:
            signal_rows.append({
                "Date": date,
                "Year": year,
                "Threshold": threshold,
                "Actual": int(
                    y_test.loc[date]
                ),
                "Prediction": int(
                    predictions[
                        test_data.index.get_loc(
                            date
                        )
                    ]
                ),
                "Confidence": float(
                    confidence.loc[date]
                ),
            })

    return (
        results,
        signal_rows,
    )


if __name__ == "__main__":
    model_data = load_model_data(
        FEATURES
    )

    all_results = []
    all_signals = []

    for year in range(
        2019,
        2026,
    ):
        yearly_results, yearly_signals = (
            evaluate_year(
                year,
                model_data,
            )
        )

        all_results.extend(
            yearly_results
        )

        all_signals.extend(
            yearly_signals
        )

    results = pd.DataFrame(
        all_results
    )

    signals = pd.DataFrame(
        all_signals
    )

    print(
        "\nYear-by-year confidence results:"
    )

    print(
        results
    )

    summary_rows = []

    for threshold in THRESHOLDS:
        threshold_signals = signals[
            signals[
                "Threshold"
            ] == threshold
        ]

        trades = len(
            threshold_signals
        )

        if trades == 0:
            pooled_accuracy = None

        else:
            pooled_accuracy = (
                threshold_signals[
                    "Actual"
                ]
                == threshold_signals[
                    "Prediction"
                ]
            ).mean()

        average_coverage = (
            results[
                results[
                    "Threshold"
                ] == threshold
            ][
                "Coverage"
            ].mean()
        )

        summary_rows.append({
            "Threshold": threshold,
            "Pooled_Accuracy": pooled_accuracy,
            "Total_Trades": trades,
            "Average_Coverage": (
                average_coverage * 100
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        "\nConfidence summary:"
    )

    print(
        summary
    )

    # ----------------------------------------
    # UP vs DOWN performance
    # ----------------------------------------

    direction_rows = []

    for threshold in THRESHOLDS:
        threshold_signals = signals[
            signals[
                "Threshold"
            ] == threshold
        ]

        for prediction_value, label in [
            (1, "UP"),
            (0, "DOWN"),
        ]:
            direction_signals = (
                threshold_signals[
                    threshold_signals[
                        "Prediction"
                    ] == prediction_value
                ]
            )

            trades = len(
                direction_signals
            )

            if trades == 0:
                direction_accuracy = None

            else:
                direction_accuracy = (
                    direction_signals[
                        "Actual"
                    ]
                    == direction_signals[
                        "Prediction"
                    ]
                ).mean()

            direction_rows.append({
                "Threshold": threshold,
                "Direction": label,
                "Accuracy": direction_accuracy,
                "Trades": trades,
            })

    direction_summary = pd.DataFrame(
        direction_rows
    )

    print(
        "\nUP vs DOWN performance:"
    )

    print(
        direction_summary
    )
