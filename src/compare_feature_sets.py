import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_baseline_model import (
    TECHNICAL_FEATURES,
    STORAGE_FEATURES,
    load_model_data,
)


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])


def walk_forward_accuracy(features):
    model_data = load_model_data(features)

    results = []

    for year in range(2019, 2026):
        train_data = model_data[model_data.index.year < year]
        test_data = model_data[model_data.index.year == year]

        X_train = train_data[features]
        y_train = train_data["Target_3D"].astype(int)

        X_test = test_data[features]
        y_test = test_data["Target_3D"].astype(int)

        model = make_model()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results.append({
            "Year": year,
            "Accuracy": accuracy_score(y_test, predictions)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    technical_results = walk_forward_accuracy(
        TECHNICAL_FEATURES
    )

    combined_features = (
        TECHNICAL_FEATURES + STORAGE_FEATURES
    )

    combined_results = walk_forward_accuracy(
        combined_features
    )

    comparison = technical_results.rename(
        columns={"Accuracy": "Technical"}
    )

    comparison["Technical_Plus_Storage"] = (
        combined_results["Accuracy"]
    )

    print(comparison)

    print("\nTechnical average:")
    print(comparison["Technical"].mean())

    print("\nTechnical + Storage average:")
    print(
        comparison["Technical_Plus_Storage"].mean()
    )

