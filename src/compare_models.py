import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def make_logistic_model():
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


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )


def walk_forward_model(model_name):
    model_data = load_model_data(
        FEATURES
    )

    results = []

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

        y_test = test_data[
            "Target_3D"
        ].astype(int)

        if model_name == "logistic":
            model = make_logistic_model()

        elif model_name == "random_forest":
            model = make_random_forest()

        else:
            raise ValueError(
                "Unknown model"
            )

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

    return pd.DataFrame(
        results
    )


if __name__ == "__main__":

    logistic_results = walk_forward_model(
        "logistic"
    )

    random_forest_results = walk_forward_model(
        "random_forest"
    )

    comparison = logistic_results.rename(
        columns={
            "Accuracy":
            "Logistic_Regression"
        }
    )

    comparison[
        "Random_Forest"
    ] = random_forest_results[
        "Accuracy"
    ]

    print(comparison)

    print(
        "\nLogistic Regression average:"
    )
    print(
        comparison[
            "Logistic_Regression"
        ].mean()
    )

    print(
        "\nRandom Forest average:"
    )
    print(
        comparison[
            "Random_Forest"
        ].mean()
    )
