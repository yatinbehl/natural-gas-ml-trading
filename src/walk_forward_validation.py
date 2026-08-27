import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_baseline_model import FEATURES, load_model_data


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])


if __name__ == "__main__":
    model_data = load_model_data()

    test_years = range(2019, 2026)

    results = []

    for year in test_years:
        train_data = model_data[model_data.index.year < year]
        test_data = model_data[model_data.index.year == year]

        X_train = train_data[FEATURES]
        y_train = train_data["Target_3D"].astype(int)

        X_test = test_data[FEATURES]
        y_test = test_data["Target_3D"].astype(int)

        model = make_model()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        naive_accuracy = y_test.value_counts(normalize=True).max()

        results.append({
            "Year": year,
            "Accuracy": accuracy,
            "Naive_Accuracy": naive_accuracy
        })

    results_df = pd.DataFrame(results)

    print(results_df)

    print("\nAverage model accuracy:")
    print(results_df["Accuracy"].mean())

    print("\nAverage naive accuracy:")
    print(results_df["Naive_Accuracy"].mean())
