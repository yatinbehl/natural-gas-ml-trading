import pandas as pd

from sklearn.ensemble import RandomForestClassifier

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


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )


if __name__ == "__main__":
    model_data = load_model_data(
        FEATURES
    )

    yearly_importances = []

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

        model = make_random_forest()

        model.fit(
            X_train,
            y_train
        )

        importance_table = pd.DataFrame({
            "Year": year,
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        })

        yearly_importances.append(
            importance_table
        )

    all_importances = pd.concat(
        yearly_importances,
        ignore_index=True
    )

    average_importance = (
        all_importances
        .groupby("Feature")["Importance"]
        .mean()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    print(
        "\nAverage feature importance:"
    )

    print(
        average_importance
    )

    print(
        "\nYear-by-year feature importance:"
    )

    pivot_table = (
        all_importances
        .pivot(
            index="Feature",
            columns="Year",
            values="Importance"
        )
    )

    print(
        pivot_table
    )
