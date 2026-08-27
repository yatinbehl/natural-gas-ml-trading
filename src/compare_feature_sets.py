import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_baseline_model import (
    TECHNICAL_FEATURES,
    STORAGE_FEATURES,
    WEATHER_FEATURES,
    FORECAST_WEATHER_FEATURES,
    load_model_data,
)


def make_model():
    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])


def walk_forward_accuracy(features):
    model_data = load_model_data(features)

    results = []

    for year in range(2019, 2026):
        train_data = model_data[
            model_data.index.year < year
        ]

        test_data = model_data[
            model_data.index.year == year
        ]

        X_train = train_data[features]
        y_train = train_data[
            "Target_3D"
        ].astype(int)

        X_test = test_data[features]
        y_test = test_data[
            "Target_3D"
        ].astype(int)

        model = make_model()

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results.append({
            "Year": year,
            "Accuracy": accuracy
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    # ----------------------------------------
    # 1. Technical only
    # ----------------------------------------

    technical_results = walk_forward_accuracy(
        TECHNICAL_FEATURES
    )

    # ----------------------------------------
    # 2. Technical + storage
    # ----------------------------------------

    technical_storage_features = (
        TECHNICAL_FEATURES
        + STORAGE_FEATURES
    )

    storage_results = walk_forward_accuracy(
        technical_storage_features
    )

    # ----------------------------------------
    # 3. Technical + storage + realized weather
    # ----------------------------------------

    realized_weather_features = (
        TECHNICAL_FEATURES
        + STORAGE_FEATURES
        + WEATHER_FEATURES
    )

    realized_weather_results = (
        walk_forward_accuracy(
            realized_weather_features
        )
    )

    # ----------------------------------------
    # 4. Technical + storage + forecast weather
    # ----------------------------------------

    forecast_weather_features = (
        TECHNICAL_FEATURES
        + STORAGE_FEATURES
        + FORECAST_WEATHER_FEATURES
    )

    forecast_weather_results = (
        walk_forward_accuracy(
            forecast_weather_features
        )
    )

    # ----------------------------------------
    # Build comparison table
    # ----------------------------------------

    comparison = technical_results.rename(
        columns={
            "Accuracy": "Technical"
        }
    )

    comparison[
        "Technical_Plus_Storage"
    ] = storage_results[
        "Accuracy"
    ]

    comparison[
        "Technical_Storage_Realized_Weather"
    ] = realized_weather_results[
        "Accuracy"
    ]

    comparison[
        "Technical_Storage_Forecast_Weather"
    ] = forecast_weather_results[
        "Accuracy"
    ]

    # ----------------------------------------
    # Print results
    # ----------------------------------------

    print(comparison)

    print("\nTechnical average:")
    print(
        comparison[
            "Technical"
        ].mean()
    )

    print(
        "\nTechnical + Storage average:"
    )
    print(
        comparison[
            "Technical_Plus_Storage"
        ].mean()
    )

    print(
        "\nTechnical + Storage + "
        "Realized Weather average:"
    )
    print(
        comparison[
            "Technical_Storage_Realized_Weather"
        ].mean()
    )

    print(
        "\nTechnical + Storage + "
        "Forecast Weather average:"
    )
    print(
        comparison[
            "Technical_Storage_Forecast_Weather"
        ].mean()
    )
