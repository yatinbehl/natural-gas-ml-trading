import numpy as np
import pandas as pd

from backtest_etf_strategy import (
    load_etf_data,
    get_frozen_predictions,
    build_trade_candidates,
    remove_overlapping_trades,
)


STARTING_CAPITAL = 10000.0

# Frozen allocation rule:
#
# No active signal:
#   HNU 50%
#   HND 50%
#
# UP signal:
#   HNU 60%
#   HND 40%
#
# DOWN signal:
#   HNU 40%
#   HND 60%

BASE_WEIGHTS = {
    "HNU": 0.50,
    "HND": 0.50,
}

UP_WEIGHTS = {
    "HNU": 0.60,
    "HND": 0.40,
}

DOWN_WEIGHTS = {
    "HNU": 0.40,
    "HND": 0.60,
}

# Cost applied only to dollars actually traded.
# 0.001 = 0.10% of traded notional.
TRADING_COST_RATE = 0.001


def get_price(
    df,
    date,
    column,
):

    match = df[
        df["Date"] == date
    ]

    if match.empty:
        raise ValueError(
            f"No ETF data for {date}"
        )

    return float(
        match.iloc[0][column]
    )


def portfolio_value(
    shares,
    hnu_price,
    hnd_price,
):

    return (
        shares["HNU"] * hnu_price
        +
        shares["HND"] * hnd_price
    )


def rebalance(
    shares,
    hnu_price,
    hnd_price,
    target_weights,
):

    current_hnu = (
        shares["HNU"]
        * hnu_price
    )

    current_hnd = (
        shares["HND"]
        * hnd_price
    )

    equity_before = (
        current_hnu
        + current_hnd
    )

    target_hnu_before_cost = (
        equity_before
        * target_weights["HNU"]
    )

    target_hnd_before_cost = (
        equity_before
        * target_weights["HND"]
    )

    traded_notional = (
        abs(
            target_hnu_before_cost
            - current_hnu
        )
        +
        abs(
            target_hnd_before_cost
            - current_hnd
        )
    )

    cost = (
        traded_notional
        * TRADING_COST_RATE
    )

    equity_after_cost = (
        equity_before
        - cost
    )

    new_shares = {
        "HNU":
            (
                equity_after_cost
                * target_weights["HNU"]
                / hnu_price
            ),

        "HND":
            (
                equity_after_cost
                * target_weights["HND"]
                / hnd_price
            ),
    }

    return (
        new_shares,
        cost,
        traded_notional,
    )


def initialize_portfolio(
    capital,
    hnu_price,
    hnd_price,
):

    cost = (
        capital
        * TRADING_COST_RATE
    )

    investable = (
        capital - cost
    )

    shares = {
        "HNU":
            (
                investable
                * 0.50
                / hnu_price
            ),

        "HND":
            (
                investable
                * 0.50
                / hnd_price
            ),
    }

    return shares, cost


def max_drawdown(
    values,
):

    values = np.array(
        values,
        dtype=float,
    )

    peaks = np.maximum.accumulate(
        values
    )

    drawdowns = (
        values
        / peaks
        - 1
    )

    return drawdowns.min()


def simulate(
    trades,
    hnu,
    hnd,
):

    first_date = (
        trades.iloc[0][
            "ETF_Entry_Date"
        ]
    )

    first_hnu_open = get_price(
        hnu,
        first_date,
        "Open",
    )

    first_hnd_open = get_price(
        hnd,
        first_date,
        "Open",
    )

    # --------------------------
    # ML TILT PORTFOLIO
    # --------------------------

    ml_shares, ml_cost = (
        initialize_portfolio(
            STARTING_CAPITAL,
            first_hnu_open,
            first_hnd_open,
        )
    )

    # --------------------------
    # CONTROL PORTFOLIO
    #
    # Same rebalance schedule,
    # but ALWAYS returns to 50/50.
    # This isolates whether the
    # ML direction adds value.
    # --------------------------

    control_shares, control_cost = (
        initialize_portfolio(
            STARTING_CAPITAL,
            first_hnu_open,
            first_hnd_open,
        )
    )

    total_ml_cost = ml_cost
    total_control_cost = control_cost

    ml_values = [
        STARTING_CAPITAL
        - ml_cost
    ]

    control_values = [
        STARTING_CAPITAL
        - control_cost
    ]

    rows = []

    for _, trade in (
        trades.iterrows()
    ):

        entry_date = (
            trade["ETF_Entry_Date"]
        )

        exit_date = (
            trade["ETF_Exit_Date"]
        )

        prediction = int(
            trade["Prediction"]
        )

        # ==========================
        # ENTRY OPEN
        # ==========================

        hnu_entry = get_price(
            hnu,
            entry_date,
            "Open",
        )

        hnd_entry = get_price(
            hnd,
            entry_date,
            "Open",
        )

        ml_before = portfolio_value(
            ml_shares,
            hnu_entry,
            hnd_entry,
        )

        control_before = (
            portfolio_value(
                control_shares,
                hnu_entry,
                hnd_entry,
            )
        )

        if prediction == 1:

            ml_target = (
                UP_WEIGHTS
            )

            direction = "UP"

        else:

            ml_target = (
                DOWN_WEIGHTS
            )

            direction = "DOWN"

        # ML portfolio changes
        # from its base exposure
        # to 60/40 or 40/60.

        (
            ml_shares,
            entry_ml_cost,
            entry_ml_turnover,
        ) = rebalance(
            ml_shares,
            hnu_entry,
            hnd_entry,
            ml_target,
        )

        total_ml_cost += (
            entry_ml_cost
        )

        # Control experiences the
        # SAME event dates but simply
        # rebalances back to 50/50.

        (
            control_shares,
            entry_control_cost,
            entry_control_turnover,
        ) = rebalance(
            control_shares,
            hnu_entry,
            hnd_entry,
            BASE_WEIGHTS,
        )

        total_control_cost += (
            entry_control_cost
        )

        # ==========================
        # EXIT CLOSE
        # ==========================

        hnu_exit = get_price(
            hnu,
            exit_date,
            "Close",
        )

        hnd_exit = get_price(
            hnd,
            exit_date,
            "Close",
        )

        ml_before_exit_rebalance = (
            portfolio_value(
                ml_shares,
                hnu_exit,
                hnd_exit,
            )
        )

        control_before_exit_rebalance = (
            portfolio_value(
                control_shares,
                hnu_exit,
                hnd_exit,
            )
        )

        # After signal expires,
        # ML returns to 50/50.

        (
            ml_shares,
            exit_ml_cost,
            exit_ml_turnover,
        ) = rebalance(
            ml_shares,
            hnu_exit,
            hnd_exit,
            BASE_WEIGHTS,
        )

        total_ml_cost += (
            exit_ml_cost
        )

        # Control also rebalances
        # at the exact same time.
        # Fair transaction/rebalance
        # comparison.

        (
            control_shares,
            exit_control_cost,
            exit_control_turnover,
        ) = rebalance(
            control_shares,
            hnu_exit,
            hnd_exit,
            BASE_WEIGHTS,
        )

        total_control_cost += (
            exit_control_cost
        )

        ml_after = portfolio_value(
            ml_shares,
            hnu_exit,
            hnd_exit,
        )

        control_after = (
            portfolio_value(
                control_shares,
                hnu_exit,
                hnd_exit,
            )
        )

        ml_values.append(
            ml_after
        )

        control_values.append(
            control_after
        )

        rows.append(
            {
                "Entry_Date":
                    entry_date,

                "Exit_Date":
                    exit_date,

                "Direction":
                    direction,

                "Confidence":
                    trade["Confidence"],

                "ML_Value_Before":
                    ml_before,

                "ML_Value_After":
                    ml_after,

                "Control_Value_Before":
                    control_before,

                "Control_Value_After":
                    control_after,

                "ML_Entry_Cost":
                    entry_ml_cost,

                "ML_Exit_Cost":
                    exit_ml_cost,

                "Control_Entry_Cost":
                    entry_control_cost,

                "Control_Exit_Cost":
                    exit_control_cost,

                "ML_Entry_Turnover":
                    entry_ml_turnover,

                "ML_Exit_Turnover":
                    exit_ml_turnover,
            }
        )

    result = pd.DataFrame(
        rows
    )

    return {
        "results": result,

        "ml_final":
            ml_values[-1],

        "control_final":
            control_values[-1],

        "ml_drawdown":
            max_drawdown(
                ml_values
            ),

        "control_drawdown":
            max_drawdown(
                control_values
            ),

        "ml_cost":
            total_ml_cost,

        "control_cost":
            total_control_cost,

        "ml_values":
            ml_values,

        "control_values":
            control_values,
    }


def static_50_50(
    trades,
    hnu,
    hnd,
):

    start_date = (
        trades.iloc[0][
            "ETF_Entry_Date"
        ]
    )

    end_date = (
        trades.iloc[-1][
            "ETF_Exit_Date"
        ]
    )

    hnu_start = get_price(
        hnu,
        start_date,
        "Open",
    )

    hnd_start = get_price(
        hnd,
        start_date,
        "Open",
    )

    hnu_end = get_price(
        hnu,
        end_date,
        "Close",
    )

    hnd_end = get_price(
        hnd,
        end_date,
        "Close",
    )

    initial_cost = (
        STARTING_CAPITAL
        * TRADING_COST_RATE
    )

    capital = (
        STARTING_CAPITAL
        - initial_cost
    )

    hnu_shares = (
        capital
        * 0.50
        / hnu_start
    )

    hnd_shares = (
        capital
        * 0.50
        / hnd_start
    )

    final_value = (
        hnu_shares
        * hnu_end
        +
        hnd_shares
        * hnd_end
    )

    return (
        final_value,
        initial_cost,
    )


if __name__ == "__main__":

    print(
        "\nLoading repaired ETFs..."
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

    candidates = (
        build_trade_candidates(
            predictions,
            hnu,
            hnd,
        )
    )

    trades = (
        remove_overlapping_trades(
            candidates
        )
    )

    print(
        f"Signals used: "
        f"{len(trades)}"
    )

    simulation = simulate(
        trades,
        hnu,
        hnd,
    )

    static_final, static_cost = (
        static_50_50(
            trades,
            hnu,
            hnd,
        )
    )

    ml_final = (
        simulation["ml_final"]
    )

    control_final = (
        simulation[
            "control_final"
        ]
    )

    ml_return = (
        ml_final
        / STARTING_CAPITAL
        - 1
    )

    control_return = (
        control_final
        / STARTING_CAPITAL
        - 1
    )

    static_return = (
        static_final
        / STARTING_CAPITAL
        - 1
    )

    start_date = (
        trades.iloc[0][
            "ETF_Entry_Date"
        ]
    )

    end_date = (
        trades.iloc[-1][
            "ETF_Exit_Date"
        ]
    )

    years = (
        (
            end_date
            - start_date
        ).days
        / 365.25
    )

    ml_cagr = (
        (
            ml_final
            / STARTING_CAPITAL
        )
        ** (1 / years)
        - 1
    )

    control_cagr = (
        (
            control_final
            / STARTING_CAPITAL
        )
        ** (1 / years)
        - 1
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CONTINUOUS PORTFOLIO TEST"
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

    print(
        f"Signals: "
        f"{len(trades)}"
    )

    print(
        "\nML 60/40 TILT:"
    )

    print(
        f"Final value: "
        f"${ml_final:,.2f}"
    )

    print(
        f"Total return: "
        f"{ml_return:.2%}"
    )

    print(
        f"CAGR: "
        f"{ml_cagr:.2%}"
    )

    print(
        f"Event max drawdown: "
        f"{simulation['ml_drawdown']:.2%}"
    )

    print(
        f"Trading costs: "
        f"${simulation['ml_cost']:,.2f}"
    )

    print(
        "\nEVENT-REBALANCED 50/50 CONTROL:"
    )

    print(
        f"Final value: "
        f"${control_final:,.2f}"
    )

    print(
        f"Total return: "
        f"{control_return:.2%}"
    )

    print(
        f"CAGR: "
        f"{control_cagr:.2%}"
    )

    print(
        f"Event max drawdown: "
        f"{simulation['control_drawdown']:.2%}"
    )

    print(
        f"Trading costs: "
        f"${simulation['control_cost']:,.2f}"
    )

    print(
        "\nSTATIC 50/50 BUY & HOLD:"
    )

    print(
        f"Final value: "
        f"${static_final:,.2f}"
    )

    print(
        f"Total return: "
        f"{static_return:.2%}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALUE ADDED BY ML TILT"
    )

    print(
        "=" * 70
    )

    print(
        "ML minus event-rebalanced "
        "control:"
    )

    print(
        f"${ml_final - control_final:,.2f}"
    )

    print(
        "Return difference:"
    )

    print(
        f"{ml_return - control_return:.2%}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Event drawdown is measured only "
        "at signal exit checkpoints, "
        "not every trading day."
    )

    simulation[
        "results"
    ].to_csv(
        "data/continuous_tilt_results.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        "data/continuous_tilt_results.csv"
    )
