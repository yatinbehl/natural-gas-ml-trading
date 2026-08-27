import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from build_features import build_features
from merge_all_data import merge_all_data

TECHNICAL_FEATURES = [
    "Daily_Return",
    "Volatility_20D",
    "Momentum_5D",
    "Momentum_20D",
]

STORAGE_FEATURES = [
    "Weekly_Change_Bcf",
    "Storage_vs_5Y_Avg_Pct",
    "Storage_Surplus_Change",
]

WEATHER_FEATURES = [
    "HDD_1D_Lag",
    "CDD_1D_Lag",
    "HDD_7D_Avg",
    "CDD_7D_Avg",
]

FORECAST_WEATHER_FEATURES = [
    "Forecast_HDD_7D",
    "Forecast_CDD_7D",
    "HDD_7D_Outlook_Change",
    "CDD_7D_Outlook_Change",
]


FEATURES = TECHNICAL_FEATURES

def load_model_data(features=FEATURES):

    data = merge_all_data()

    data = data.set_index("Date")


    data = build_features(data)

    model_data = data[
        features + ["Target_3D"]
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
