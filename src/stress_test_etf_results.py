import numpy as np
import pandas as pd

from backtest_etf_strategy import (
    STARTING_CAPITAL,
    POSITION_SIZE,
    TRANSACTION_COST,
    load_etf_data,
    get_frozen_predictions,
    build_trade_candidates,
    remove_overlapping_trades,
)


N_RANDOM_SIMULATIONS = 5000
RANDOM_SEED = 42


def portfolio_stats(trades):

    equity = STARTING_CAPITAL
    peak = equity
    max_drawdown = 0.0
    wins = 0

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

    n = len(trades)

    return {
        "Trades": n,
        "Win_Rate":
            wins / n if n else np.nan,
        "Avg_Trade_Return":
            trades[
                "ETF_Net_Return"
            ].mean()
            if n else np.nan,
        "Final_Equity": equity,
        "Total_Return":
            equity
            / STARTING_CAPITAL
            - 1,
        "Max_Drawdown":
            max_drawdown,
    }


def year_by_year_analysis(
    trades,
):

    rows = []

    trades = trades.copy()

    trades["Year"] = (
        pd.to_datetime(
            trades[
                "ETF_Entry_Date"
            ]
        ).dt.year
    )

    for year, group in (
        trades.groupby("Year")
    ):

        stats = portfolio_stats(
            group
        )

        up = group[
            group["Prediction"] == 1
        ]

        down = group[
            group["Prediction"] == 0
        ]

        rows.append(
            {
                "Year": year,

                "Trades":
                    stats["Trades"],

                "Win_Rate":
                    stats["Win_Rate"],

                "Avg_Return":
                    stats[
                        "Avg_Trade_Return"
                    ],

                "Portfolio_Return":
                    stats[
                        "Total_Return"
                    ],

                "Max_Drawdown":
                    stats[
                        "Max_Drawdown"
                    ],

                "UP_Trades":
                    len(up),

                "UP_Win_Rate":
                    (
                        (
                            up[
                                "ETF_Net_Return"
                            ] > 0
                        ).mean()
                        if len(up)
                        else np.nan
                    ),

                "DOWN_Trades":
                    len(down),

                "DOWN_Win_Rate":
                    (
                        (
                            down[
                                "ETF_Net_Return"
                            ] > 0
                        ).mean()
                        if len(down)
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def pseudo_holdout_2025(
    trades,
):

    test = trades[
        pd.to_datetime(
            trades[
                "ETF_Entry_Date"
            ]
        ).dt.year
        == 2025
    ].copy()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "2025 ETF PSEUDO-HOLDOUT"
    )

    print(
        "=" * 70
    )

    stats = portfolio_stats(
        test
    )

    for key, value in (
        stats.items()
    ):

        if key == "Trades":

            print(
                f"{key}: {value}"
            )

        else:

            print(
                f"{key}: "
                f"{value:.2%}"
                if "Equity"
                not in key
                else
                f"{key}: "
                f"${value:,.2f}"
            )

    if len(test):

        print(
            "\n2025 trades:"
        )

        print(
            test[
                [
                    "ETF_Entry_Date",
                    "ETF_Exit_Date",
                    "Direction",
                    "ETF",
                    "Confidence",
                    "ETF_Net_Return",
                ]
            ].to_string(
                index=False
            )
        )

    return test


def get_return_for_direction(
    row,
    hnu,
    hnd,
    direction,
):

    entry_date = row[
        "ETF_Entry_Date"
    ]

    exit_date = row[
        "ETF_Exit_Date"
    ]

    if direction == 1:
        etf = hnu
    else:
        etf = hnd

    entry_match = etf[
        etf["Date"] == entry_date
    ]

    exit_match = etf[
        etf["Date"] == exit_date
    ]

    if (
        entry_match.empty
        or exit_match.empty
    ):
        return np.nan

    entry_price = float(
        entry_match.iloc[0]["Open"]
    )

    exit_price = float(
        exit_match.iloc[0]["Close"]
    )

    gross_return = (
        exit_price
        / entry_price
        - 1
    )

    return (
        gross_return
        - TRANSACTION_COST
    )


def build_randomization_matrix(
    trades,
    hnu,
    hnd,
):

    matrix = []

    for _, row in trades.iterrows():

        hnu_return = (
            get_return_for_direction(
                row,
                hnu,
                hnd,
                1,
            )
        )

        hnd_return = (
            get_return_for_direction(
                row,
                hnu,
                hnd,
                0,
            )
        )

        matrix.append(
            {
                "ETF_Entry_Date":
                    row[
                        "ETF_Entry_Date"
                    ],

                "ETF_Exit_Date":
                    row[
                        "ETF_Exit_Date"
                    ],

                "Original_Prediction":
                    row["Prediction"],

                "HNU_Return":
                    hnu_return,

                "HND_Return":
                    hnd_return,
            }
        )

    return pd.DataFrame(
        matrix
    ).dropna()


def randomized_direction_test(
    matrix,
    actual_total_return,
):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    original_predictions = (
        matrix[
            "Original_Prediction"
        ].to_numpy()
    )

    hnu_returns = (
        matrix[
            "HNU_Return"
        ].to_numpy()
    )

    hnd_returns = (
        matrix[
            "HND_Return"
        ].to_numpy()
    )

    random_results = []

    for _ in range(
        N_RANDOM_SIMULATIONS
    ):

        shuffled = rng.permutation(
            original_predictions
        )

        chosen_returns = np.where(
            shuffled == 1,
            hnu_returns,
            hnd_returns,
        )

        equity = (
            STARTING_CAPITAL
        )

        for trade_return in (
            chosen_returns
        ):

            equity *= (
                1
                + POSITION_SIZE
                * trade_return
            )

        total_return = (
            equity
            / STARTING_CAPITAL
            - 1
        )

        random_results.append(
            total_return
        )

    random_results = np.array(
        random_results
    )

    percentile = (
        random_results
        < actual_total_return
    ).mean()

    p_value = (
        random_results
        >= actual_total_return
    ).mean()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RANDOMIZED-DIRECTION NULL TEST"
    )

    print(
        "=" * 70
    )

    print(
        f"Simulations: "
        f"{N_RANDOM_SIMULATIONS}"
    )

    print(
        f"Actual model return: "
        f"{actual_total_return:.2%}"
    )

    print(
        f"Random median return: "
        f"{np.median(random_results):.2%}"
    )

    print(
        f"Random mean return: "
        f"{random_results.mean():.2%}"
    )

    print(
        f"Random 95th percentile: "
        f"{np.percentile(random_results, 95):.2%}"
    )

    print(
        f"Random 99th percentile: "
        f"{np.percentile(random_results, 99):.2%}"
    )

    print(
        f"Model percentile: "
        f"{percentile:.2%}"
    )

    print(
        f"Empirical p-value: "
        f"{p_value:.4f}"
    )

    return pd.DataFrame(
        {
            "Random_Total_Return":
                random_results
        }
    )


if __name__ == "__main__":

    print(
        "\nLoading repaired ETF data..."
    )

    hnu = load_etf_data(
        "data/hnu_adjusted.csv"
    )

    hnd = load_etf_data(
        "data/hnd_adjusted.csv"
    )

    predictions = (
        get_frozen_predictions()
    )

    trade_candidates = (
        build_trade_candidates(
            predictions,
            hnu,
            hnd,
        )
    )

    trades = (
        remove_overlapping_trades(
            trade_candidates
        )
    )

    print(
        f"\nNon-overlapping trades: "
        f"{len(trades)}"
    )

    actual_stats = (
        portfolio_stats(
            trades
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ACTUAL MODEL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Trades: "
        f"{actual_stats['Trades']}"
    )

    print(
        f"Win rate: "
        f"{actual_stats['Win_Rate']:.2%}"
    )

    print(
        f"Average trade return: "
        f"{actual_stats['Avg_Trade_Return']:.2%}"
    )

    print(
        f"Final equity: "
        f"${actual_stats['Final_Equity']:,.2f}"
    )

    print(
        f"Total return: "
        f"{actual_stats['Total_Return']:.2%}"
    )

    print(
        f"Max drawdown: "
        f"{actual_stats['Max_Drawdown']:.2%}"
    )

    yearly = (
        year_by_year_analysis(
            trades
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "YEAR-BY-YEAR ETF RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        yearly.to_string(
            index=False
        )
    )

    pseudo_holdout_2025(
        trades
    )

    matrix = (
        build_randomization_matrix(
            trades,
            hnu,
            hnd,
        )
    )

    random_results = (
        randomized_direction_test(
            matrix,
            actual_stats[
                "Total_Return"
            ],
        )
    )

    yearly.to_csv(
        "data/etf_yearly_results.csv",
        index=False,
    )

    random_results.to_csv(
        "data/etf_randomization_results.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        "data/etf_yearly_results.csv"
    )

    print(
        "data/etf_randomization_results.csv"
    )
