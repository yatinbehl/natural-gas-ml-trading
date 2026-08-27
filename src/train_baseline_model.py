import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from build_features import build_features


FEATURES = [
    "Daily_Return",
    "Volatility_20D",
    "Momentum_5D",
    "Momentum_20D",
]


def load_model_data():
    data = pd.read_csv(
        "data/ng_futures.csv",
        index_col="Date",
        parse_dates=True
    )

    data = build_features(data)

    model_data = data[
        FEATURES + ["Target_3D"]
    ].dropna()

    return model_data


if __name__ == "__main__":
    model_data = load_model_data()

    split_index = int(len(model_data) * 0.80)

    train_data = model_data.iloc[:split_index]
    test_data = model_data.iloc[split_index:]

    X_train = train_data[FEATURES]
    y_train = train_data["Target_3D"].astype(int)

    X_test = test_data[FEATURES]
    y_test = test_data["Target_3D"].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])



    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Train period:")
    print(train_data.index.min(), "to", train_data.index.max())

    print("\nTest period:")
    print(test_data.index.min(), "to", test_data.index.max())

    print("\nTest target distribution:")
    print(y_test.value_counts())

    print("\nTest target percentages:")
    print(y_test.value_counts(normalize=True) * 100)

    print("\nBaseline accuracy:")
    print(accuracy)

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    coefficient_table = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.named_steps["classifier"].coef_[0]
    })

    print("\nModel coefficients:")
    print(coefficient_table)
