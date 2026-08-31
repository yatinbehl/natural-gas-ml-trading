import pandas as pd
import numpy as np

from backtest_portfolio import load_backtest_data
from test_prediction_horizons import generate_predictions


HORIZON = 2
CONFIDENCE_THRESHOLD = 0.60

STARTING_CAPITAL = 10000.0

# Keep consistent with our earlier
# conservative portfolio backtests.
POSITION_SIZE = 0.25

# Approximate total round-trip trading friction.
TRANSACTION_COST = 0.001

HNU_FILE = "data/hnu_adjusted.csv"
HND_FILE = "data/hnd_adjusted.csv"


def load_etf_data(filepath):

    df = pd.read_csv(
        filepath,
        parse_dates=["Date"],
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    return df


def get_frozen_predictions():

    data = load_backtest_data()

    preds = generate_predictions(
        data,
        horizon=HORIZON,
    )

    preds["Entry_Date"] = pd.to_datetime(
        preds["Entry_Date"]
    )

    preds["Exit_Date"] = pd.to_datetime(
        preds["Exit_Date"]
    )

    preds = preds[
        preds["Confidence"]
        >= CONFIDENCE_THRESHOLD
    ].copy()

    preds = (
        preds.sort_values("Entry_Date")
        .reset_index(drop=True)
    )

    return preds


def first_etf_row_on_or_after(
    etf,
    target_date,
):

    rows = etf[
        etf["Date"] >= target_date
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def build_trade_candidates(
    predictions,
    hnu,
    hnd,
):

    trades = []

    for _, row in predictions.iterrows():

        prediction = int(
            row["Prediction"]
        )

        if prediction == 1:
            etf_name = "HNU"
            etf = hnu
        else:
            etf_name = "HND"
            etf = hnd

        entry_row = (
            first_etf_row_on_or_after(
                etf,
                row["Entry_Date"],
            )
        )

        exit_row = (
            first_etf_row_on_or_after(
                etf,
                row["Exit_Date"],
            )
        )

        if (
            entry_row is None
            or exit_row is None
        ):
            continue

        entry_date = entry_row["Date"]
        exit_date = exit_row["Date"]

        if exit_date <= entry_date:
            continue

        entry_price = float(
            entry_row["Open"]
        )

        exit_price = float(
            exit_row["Close"]
        )

        if (
            entry_price <= 0
            or exit_price <= 0
        ):
            continue

        gross_return = (
            exit_price
            / entry_price
            - 1
        )

        net_return = (
            gross_return
            - TRANSACTION_COST
        )

        trades.append(
            {
                "Model_Entry_Date":
                    row["Entry_Date"],

                "Model_Exit_Date":
                    row["Exit_Date"],

                "ETF_Entry_Date":
                    entry_date,

                "ETF_Exit_Date":
                    exit_date,

                "Prediction":
                    prediction,

                "Direction":
                    (
                        "UP"
                        if prediction == 1
                        else "DOWN"
                    ),

                "ETF":
                    etf_name,

                "Confidence":
                    row["Confidence"],

                "Entry_Price":
                    entry_price,

                "Exit_Price":
                    exit_price,

                "ETF_Gross_Return":
                    gross_return,

                "ETF_Net_Return":
                    net_return,
            }
        )

    return pd.DataFrame(trades)


def remove_overlapping_trades(
    trades,
):

    selected = []

    last_exit_date = None

    for _, trade in trades.iterrows():

        entry_date = (
            trade["ETF_Entry_Date"]
        )

        exit_date = (
            trade["ETF_Exit_Date"]
        )

        if (
            last_exit_date is not None
            and entry_date <= last_exit_date
        ):
            continue

        selected.append(
            trade.to_dict()
        )

        last_exit_date = exit_date

    return pd.DataFrame(selected)


def portfolio_backtest(
    trades,
    label,
):

    equity = STARTING_CAPITAL

    peak = equity

    max_drawdown = 0.0

    wins = 0

    equity_curve = []

    for _, trade in trades.iterrows():

        trade_return = (
            trade["ETF_Net_Return"]
        )

        portfolio_return = (
            POSITION_SIZE
            * trade_return
        )

        equity *= (
            1 + portfolio_return
        )

        if trade_return > 0:
            wins += 1

        peak = max(
            peak,
            equity,
        )

        drawdown = (
            equity / peak - 1
        )

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

        equity_curve.append(equity)

    number_trades = len(trades)

    if number_trades > 0:

        win_rate = (
            wins / number_trades
        )

        avg_trade_return = (
            trades[
                "ETF_Net_Return"
            ].mean()
        )

    else:

        win_rate = np.nan
        avg_trade_return = np.nan

    total_return = (
        equity
        / STARTING_CAPITAL
        - 1
    )

    print(
        "\n"
        + "=" * 70
    )

    print(label)

    print(
        "=" * 70
    )

    print(
        f"Trades: {number_trades}"
    )

    print(
        f"Win rate: "
        f"{win_rate:.2%}"
    )

    print(
        f"Average ETF trade return: "
        f"{avg_trade_return:.2%}"
    )

    print(
        f"Starting capital: "
        f"${STARTING_CAPITAL:,.2f}"
    )

    print(
        f"Final equity: "
        f"${equity:,.2f}"
    )

    print(
        f"Portfolio return: "
        f"{total_return:.2%}"
    )

    print(
        f"Max drawdown: "
        f"{max_drawdown:.2%}"
    )

    return {
        "Strategy": label,
        "Trades": number_trades,
        "Win_Rate": win_rate,
        "Avg_Trade_Return":
            avg_trade_return,
        "Final_Equity": equity,
        "Total_Return": total_return,
        "Max_Drawdown":
            max_drawdown,
    }


def benchmark_return(
    etf,
    start_date,
    end_date,
    name,
):

    start_row = (
        first_etf_row_on_or_after(
            etf,
            start_date,
        )
    )

    end_row = (
        first_etf_row_on_or_after(
            etf,
            end_date,
        )
    )

    if (
        start_row is None
        or end_row is None
    ):
        return None

    start_price = float(
        start_row["Open"]
    )

    end_price = float(
        end_row["Close"]
    )

    total_return = (
        end_price
        / start_price
        - 1
    )

    print(
        f"{name}: "
        f"{total_return:.2%}"
    )

    return total_return


def basket_tilt_analysis(
    trades,
    hnu,
    hnd,
):

    results = []

    for _, trade in trades.iterrows():

        entry_date = (
            trade["ETF_Entry_Date"]
        )

        exit_date = (
            trade["ETF_Exit_Date"]
        )

        hnu_entry = (
            first_etf_row_on_or_after(
                hnu,
                entry_date,
            )
        )

        hnu_exit = (
            first_etf_row_on_or_after(
                hnu,
                exit_date,
            )
        )

        hnd_entry = (
            first_etf_row_on_or_after(
                hnd,
                entry_date,
            )
        )

        hnd_exit = (
            first_etf_row_on_or_after(
                hnd,
                exit_date,
            )
        )

        if any(
            x is None
            for x in [
                hnu_entry,
                hnu_exit,
                hnd_entry,
                hnd_exit,
            ]
        ):
            continue

        hnu_return = (
            float(hnu_exit["Close"])
            / float(hnu_entry["Open"])
            - 1
        )

        hnd_return = (
            float(hnd_exit["Close"])
            / float(hnd_entry["Open"])
            - 1
        )

        # Neutral comparison:
        # 50% HNU + 50% HND
        base_return = (
            0.50 * hnu_return
            + 0.50 * hnd_return
        )

        # User's $150/$100 idea
        # normalized to constant capital:
        #
        # 150 / 250 = 60%
        # 100 / 250 = 40%
        #
        # This avoids giving the tilted
        # strategy extra capital/leverage.

        if trade["Prediction"] == 1:

            tilted_return = (
                0.60 * hnu_return
                + 0.40 * hnd_return
            )

        else:

            tilted_return = (
                0.40 * hnu_return
                + 0.60 * hnd_return
            )

        results.append(
            {
                "Entry_Date":
                    entry_date,

                "Exit_Date":
                    exit_date,

                "Direction":
                    trade["Direction"],

                "Confidence":
                    trade["Confidence"],

                "HNU_Return":
                    hnu_return,

                "HND_Return":
                    hnd_return,

                "Base_50_50_Return":
                    base_return,

                "Tilted_60_40_Return":
                    tilted_return,

                "Tilt_Improvement":
                    (
                        tilted_return
                        - base_return
                    ),
            }
        )

    result_df = pd.DataFrame(
        results
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "50/50 HNU-HND VS ML 60/40 TILT"
    )

    print(
        "=" * 70
    )

    print(
        f"Periods tested: "
        f"{len(result_df)}"
    )

    print(
        f"Average 50/50 return: "
        f"{result_df['Base_50_50_Return'].mean():.2%}"
    )

    print(
        f"Average ML tilt return: "
        f"{result_df['Tilted_60_40_Return'].mean():.2%}"
    )

    print(
        f"Average improvement from tilt: "
        f"{result_df['Tilt_Improvement'].mean():.2%}"
    )

    tilt_wins = (
        result_df[
            "Tilt_Improvement"
        ] > 0
    ).mean()

    print(
        f"Tilt beats 50/50: "
        f"{tilt_wins:.2%} of periods"
    )

    return result_df


if __name__ == "__main__":

    print(
        "\nLoading repaired ETF data..."
    )

    hnu = load_etf_data(
        HNU_FILE
    )

    hnd = load_etf_data(
        HND_FILE
    )

    print(
        "Generating frozen "
        "2-day predictions..."
    )

    predictions = (
        get_frozen_predictions()
    )

    print(
        f"\nHigh-confidence signals: "
        f"{len(predictions)}"
    )

    trade_candidates = (
        build_trade_candidates(
            predictions,
            hnu,
            hnd,
        )
    )

    print(
        f"ETF-mapped trade candidates: "
        f"{len(trade_candidates)}"
    )

    trades = (
        remove_overlapping_trades(
            trade_candidates
        )
    )

    print(
        f"Non-overlapping ETF trades: "
        f"{len(trades)}"
    )

    print(
        "\nFirst 10 trades:"
    )

    print(
        trades.head(
            10
        ).to_string(
            index=False
        )
    )

    summaries = []

    summaries.append(
        portfolio_backtest(
            trades,
            "ALL ML SIGNALS",
        )
    )

    up_trades = trades[
        trades["Prediction"] == 1
    ]

    summaries.append(
        portfolio_backtest(
            up_trades,
            "UP SIGNALS -> HNU",
        )
    )

    down_trades = trades[
        trades["Prediction"] == 0
    ]

    summaries.append(
        portfolio_backtest(
            down_trades,
            "DOWN SIGNALS -> HND",
        )
    )

    tilt_results = (
        basket_tilt_analysis(
            trades,
            hnu,
            hnd,
        )
    )

    if len(trades) > 0:

        start_date = (
            trades[
                "ETF_Entry_Date"
            ].min()
        )

        end_date = (
            trades[
                "ETF_Exit_Date"
            ].max()
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "PASSIVE ETF BENCHMARKS"
        )

        print(
            "=" * 70
        )

        print(
            f"Period: "
            f"{start_date.date()} "
            f"to "
            f"{end_date.date()}"
        )

        hnu_bh = benchmark_return(
            hnu,
            start_date,
            end_date,
            "HNU buy & hold",
        )

        hnd_bh = benchmark_return(
            hnd,
            start_date,
            end_date,
            "HND buy & hold",
        )

        if (
            hnu_bh is not None
            and hnd_bh is not None
        ):

            equal_basket = (
                0.5 * hnu_bh
                + 0.5 * hnd_bh
            )

            print(
                "Initial 50/50 "
                "HNU-HND basket: "
                f"{equal_basket:.2%}"
            )

    summary_df = pd.DataFrame(
        summaries
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STRATEGY SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    trades.to_csv(
        "data/etf_ml_trades.csv",
        index=False,
    )

    tilt_results.to_csv(
        "data/etf_tilt_analysis.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        "data/etf_ml_trades.csv"
    )

    print(
        "data/etf_tilt_analysis.csv"
    )
