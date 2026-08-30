import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from backtest_portfolio import load_backtest_data
from train_baseline_model import (
    TECHNICAL_FEATURES,
    STORAGE_FEATURES,
    FORECAST_WEATHER_FEATURES,
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

HORIZONS = [1, 2, 3, 5, 10]

TEST_YEARS = range(2019, 2026)

CONFIDENCE_THRESHOLD = 0.60

STARTING_CAPITAL = 10000.0
POSITION_SIZE = 0.25
TRANSACTION_COST = 0.001


FEATURES = (
    TECHNICAL_FEATURES
    + STORAGE_FEATURES
    + FORECAST_WEATHER_FEATURES
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def build_model():

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


# ---------------------------------------------------------
# CREATE A TRADABLE TARGET
# ---------------------------------------------------------

def create_horizon_target(data, horizon):

    df = data.copy()

    dates = pd.Series(
        df.index,
        index=df.index,
    )

    # Signal is generated after today's close.
    #
    # We therefore assume:
    #
    # Entry = next trading day's close
    # Exit  = N trading days after entry
    #
    # This avoids using today's close as an
    # unrealistically executable entry.

    df["Entry_Date"] = dates.shift(-1)

    df["Exit_Date"] = dates.shift(
        -(horizon + 1)
    )

    df["Entry_Price"] = df["Close"].shift(-1)

    df["Exit_Price"] = df["Close"].shift(
        -(horizon + 1)
    )

    df["Future_Return"] = (
        df["Exit_Price"]
        / df["Entry_Price"]
        - 1
    )

    df["Target"] = (
        df["Future_Return"] > 0
    ).astype("Int64")

    # Rows without a complete future horizon
    # must not receive a target.
    invalid = (
        df["Entry_Price"].isna()
        | df["Exit_Price"].isna()
    )

    df.loc[
        invalid,
        "Target"
    ] = pd.NA

    return df


# ---------------------------------------------------------
# WALK-FORWARD PREDICTIONS
# ---------------------------------------------------------

def generate_predictions(data, horizon):

    df = create_horizon_target(
        data,
        horizon,
    )

    prediction_frames = []

    for year in TEST_YEARS:

        test_start = pd.Timestamp(
            f"{year}-01-01"
        )

        test_end = pd.Timestamp(
            f"{year}-12-31"
        )

        # -------------------------------------------------
        # PURGED TRAINING SET
        #
        # A training row is allowed only if the COMPLETE
        # future target occurred before the test period.
        #
        # Example:
        #
        # Signal Dec 30
        # Target uses Jan 3 price
        #
        # That row must NOT train the 2019 model if
        # Jan 3 belongs to the 2019 test period.
        # -------------------------------------------------

        train_mask = (
            df["Exit_Date"] < test_start
        )

        test_mask = (
            (df.index >= test_start)
            & (df.index <= test_end)
        )

        train = df.loc[
            train_mask
        ].copy()

        test = df.loc[
            test_mask
        ].copy()

        required_train = (
            FEATURES
            + ["Target"]
        )

        required_test = (
            FEATURES
            + [
                "Target",
                "Entry_Date",
                "Exit_Date",
                "Future_Return",
            ]
        )

        train = train.dropna(
            subset=required_train
        )

        test = test.dropna(
            subset=required_test
        )

        if train.empty or test.empty:
            continue

        X_train = train[FEATURES]
        y_train = train["Target"].astype(int)

        X_test = test[FEATURES]
        y_test = test["Target"].astype(int)

        model = build_model()

        model.fit(
            X_train,
            y_train,
        )

        prediction = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )

        confidence = probabilities.max(
            axis=1
        )

        result = test[
            [
                "Entry_Date",
                "Exit_Date",
                "Future_Return",
            ]
        ].copy()

        result["Actual"] = y_test.values
        result["Prediction"] = prediction
        result["Confidence"] = confidence
        result["Year"] = year

        prediction_frames.append(
            result
        )

    if not prediction_frames:

        return pd.DataFrame()

    return pd.concat(
        prediction_frames
    ).sort_index()


# ---------------------------------------------------------
# NON-OVERLAPPING PORTFOLIO BACKTEST
# ---------------------------------------------------------

def backtest_predictions(predictions):

    high_conf = predictions[
        predictions["Confidence"]
        >= CONFIDENCE_THRESHOLD
    ].copy()

    if high_conf.empty:

        return {
            "Trades": 0,
            "Win_Rate": np.nan,
            "Final_Equity": STARTING_CAPITAL,
            "Total_Return": 0.0,
            "CAGR": np.nan,
            "Max_Drawdown": np.nan,
        }

    trades = []

    last_exit_date = None

    for signal_date, row in high_conf.iterrows():

        entry_date = pd.Timestamp(
            row["Entry_Date"]
        )

        exit_date = pd.Timestamp(
            row["Exit_Date"]
        )

        # Only one open position at a time.
        if (
            last_exit_date is not None
            and entry_date <= last_exit_date
        ):
            continue

        raw_return = row[
            "Future_Return"
        ]

        if row["Prediction"] == 1:

            strategy_return = raw_return

        else:

            strategy_return = -raw_return

        net_return = (
            strategy_return
            - TRANSACTION_COST
        )

        trades.append(
            {
                "Signal_Date": signal_date,
                "Entry_Date": entry_date,
                "Exit_Date": exit_date,
                "Prediction": row["Prediction"],
                "Confidence": row["Confidence"],
                "Return": net_return,
            }
        )

        last_exit_date = exit_date

    trades = pd.DataFrame(
        trades
    )

    if trades.empty:

        return {
            "Trades": 0,
            "Win_Rate": np.nan,
            "Final_Equity": STARTING_CAPITAL,
            "Total_Return": 0.0,
            "CAGR": np.nan,
            "Max_Drawdown": np.nan,
        }

    equity = STARTING_CAPITAL

    equity_curve = []

    for trade_return in trades["Return"]:

        portfolio_return = (
            POSITION_SIZE
            * trade_return
        )

        equity *= (
            1 + portfolio_return
        )

        equity_curve.append(
            equity
        )

    equity_series = pd.Series(
        equity_curve
    )

    running_max = equity_series.cummax()

    drawdown = (
        equity_series
        / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    total_return = (
        equity
        / STARTING_CAPITAL
        - 1
    )

    start_date = trades[
        "Entry_Date"
    ].min()

    end_date = trades[
        "Exit_Date"
    ].max()

    years = (
        end_date - start_date
    ).days / 365.25

    if years > 0:

        cagr = (
            equity
            / STARTING_CAPITAL
        ) ** (1 / years) - 1

    else:

        cagr = np.nan

    win_rate = (
        trades["Return"] > 0
    ).mean()

    return {
        "Trades": len(trades),
        "Win_Rate": win_rate,
        "Final_Equity": equity,
        "Total_Return": total_return,
        "CAGR": cagr,
        "Max_Drawdown": max_drawdown,
    }


# ---------------------------------------------------------
# RUN EXPERIMENT
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nLoading model data...")

    data = load_backtest_data()

    print(
        f"Rows loaded: {len(data)}"
    )

    print(
        "\nFeatures used:"
    )

    for feature in FEATURES:
        print(
            f"  {feature}"
        )

    results = []

    yearly_results = []

    for horizon in HORIZONS:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Testing {horizon}-day horizon"
        )

        print(
            "=" * 60
        )

        predictions = generate_predictions(
            data=data,
            horizon=horizon,
        )

        if predictions.empty:

            print(
                "No predictions generated."
            )

            continue

        # ---------------------------------------------
        # ALL MODEL PREDICTIONS
        # ---------------------------------------------

        overall_accuracy = accuracy_score(
            predictions["Actual"],
            predictions["Prediction"],
        )

        # ---------------------------------------------
        # HIGH-CONFIDENCE PREDICTIONS
        # ---------------------------------------------

        high_conf = predictions[
            predictions["Confidence"]
            >= CONFIDENCE_THRESHOLD
        ]

        if len(high_conf) > 0:

            high_conf_accuracy = accuracy_score(
                high_conf["Actual"],
                high_conf["Prediction"],
            )

        else:

            high_conf_accuracy = np.nan

        # ---------------------------------------------
        # PORTFOLIO BACKTEST
        # ---------------------------------------------

        portfolio = backtest_predictions(
            predictions
        )

        results.append(
            {
                "Horizon": horizon,
                "All_Predictions": len(
                    predictions
                ),
                "Overall_Accuracy": overall_accuracy,
                "High_Conf_Signals": len(
                    high_conf
                ),
                "High_Conf_Accuracy": high_conf_accuracy,
                "Trades": portfolio[
                    "Trades"
                ],
                "Win_Rate": portfolio[
                    "Win_Rate"
                ],
                "CAGR": portfolio[
                    "CAGR"
                ],
                "Max_Drawdown": portfolio[
                    "Max_Drawdown"
                ],
                "Final_Equity": portfolio[
                    "Final_Equity"
                ],
            }
        )

        # ---------------------------------------------
        # YEAR-BY-YEAR DIAGNOSTICS
        # ---------------------------------------------

        for year in TEST_YEARS:

            year_data = predictions[
                predictions["Year"]
                == year
            ]

            if year_data.empty:
                continue

            year_high = year_data[
                year_data["Confidence"]
                >= CONFIDENCE_THRESHOLD
            ]

            year_accuracy = accuracy_score(
                year_data["Actual"],
                year_data["Prediction"],
            )

            if len(year_high) > 0:

                year_high_accuracy = accuracy_score(
                    year_high["Actual"],
                    year_high["Prediction"],
                )

            else:

                year_high_accuracy = np.nan

            yearly_results.append(
                {
                    "Horizon": horizon,
                    "Year": year,
                    "Accuracy": year_accuracy,
                    "High_Conf_Signals": len(
                        year_high
                    ),
                    "High_Conf_Accuracy": year_high_accuracy,
                }
            )

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    yearly_df = pd.DataFrame(
        yearly_results
    )

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "MULTI-HORIZON SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\n\nHIGH-CONFIDENCE ACCURACY BY YEAR"
    )

    if not yearly_df.empty:

        matrix = yearly_df.pivot(
            index="Year",
            columns="Horizon",
            values="High_Conf_Accuracy",
        )

        print(
            matrix.to_string()
        )

    print(
        "\n\nHIGH-CONFIDENCE SIGNAL COUNT BY YEAR"
    )

    if not yearly_df.empty:

        count_matrix = yearly_df.pivot(
            index="Year",
            columns="Horizon",
            values="High_Conf_Signals",
        )

        print(
            count_matrix.to_string()
        )
