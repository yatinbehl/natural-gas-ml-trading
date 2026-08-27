from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


START_YEAR = 2014
END_YEAR = 2026

OUTPUT_FILE = Path(
    "data/ng_weather_forecasts.csv"
)

BASE_URL = (
    "https://ftp.cpc.ncep.noaa.gov/htdocs/"
    "degree_days/weighted/daily_forecasts_7day"
)


def create_session():
    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "GET"
        ],
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter
    )

    return session


SESSION = create_session()


def download_forecast(
    date,
    filename
):
    date_path = date.strftime(
        "%Y/%m/%d"
    )

    url = (
        f"{BASE_URL}/"
        f"{date_path}/"
        f"{filename}"
    )

    response = SESSION.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    return response.text


def parse_conus_total(text):
    lines = text.splitlines()

    table_start = next(
        (
            i
            for i, line
            in enumerate(lines)
            if line.startswith(
                "Region|"
            )
        ),
        None
    )

    if table_start is None:
        return None

    table_text = "\n".join(
        lines[
            table_start:
        ]
    )

    data = pd.read_csv(
        StringIO(
            table_text
        ),
        sep="|"
    )

    conus = data[
        data["Region"]
        == "CONUS"
    ]

    if conus.empty:
        return None

    return conus[
        "Total"
    ].iloc[0]


def fetch_date(date):
    try:
        hdd_text = (
            download_forecast(
                date,
                "Population.Heating.txt"
            )
        )

        # Small pause between requests
        time.sleep(0.25)

        cdd_text = (
            download_forecast(
                date,
                "Population.Cooling.txt"
            )
        )

        hdd = parse_conus_total(
            hdd_text
        )

        cdd = parse_conus_total(
            cdd_text
        )

        if (
            hdd is None
            or cdd is None
        ):
            print(
                "Invalid forecast format "
                f"for {date.date()}"
            )

            return None

        return {
            "Forecast_Date": date,
            "Forecast_HDD_7D": hdd,
            "Forecast_CDD_7D": cdd,
        }

    except requests.RequestException as error:
        print(
            "Request failed for "
            f"{date.date()}: "
            f"{error}"
        )

        return None


def load_existing_data():
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()

    existing = pd.read_csv(
        OUTPUT_FILE,
        parse_dates=[
            "Forecast_Date"
        ]
    )

    print(
        f"Loaded {len(existing)} "
        "existing forecast rows."
    )

    return existing


def save_progress(rows):
    if not rows:
        return

    forecasts = pd.DataFrame(
        rows
    )

    forecasts = forecasts.sort_values(
        "Forecast_Date"
    )

    forecasts = (
        forecasts
        .drop_duplicates(
            subset="Forecast_Date",
            keep="last"
        )
    )

    forecasts.to_csv(
        OUTPUT_FILE,
        index=False
    )


def fetch_all_forecasts():
    existing = load_existing_data()

    if existing.empty:
        rows = []
        completed_dates = set()

    else:
        rows = existing[
            [
                "Forecast_Date",
                "Forecast_HDD_7D",
                "Forecast_CDD_7D",
            ]
        ].to_dict(
            "records"
        )

        completed_dates = set(
            existing[
                "Forecast_Date"
            ].dt.normalize()
        )

    start_date = pd.Timestamp(
        START_YEAR,
        1,
        1
    )

    end_date = min(
        pd.Timestamp(
            END_YEAR,
            12,
            31
        ),
        pd.Timestamp.today().normalize()
    )

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    downloads_since_save = 0

    for date in dates:
        normalized_date = (
            date.normalize()
        )

        if (
            normalized_date
            in completed_dates
        ):
            continue

        print(
            f"Downloading "
            f"{date.date()}..."
        )

        row = fetch_date(
            date
        )

        if row is not None:
            rows.append(
                row
            )

        downloads_since_save += 1

        # Avoid hammering NOAA
        time.sleep(0.5)

        # Save every 25 attempted dates
        if (
            downloads_since_save
            >= 25
        ):
            save_progress(
                rows
            )

            print(
                "Checkpoint saved: "
                f"{len(rows)} rows"
            )

            downloads_since_save = 0

            time.sleep(2)

    forecasts = pd.DataFrame(
        rows
    )

    forecasts = (
        forecasts
        .sort_values(
            "Forecast_Date"
        )
        .drop_duplicates(
            subset="Forecast_Date",
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    # Conservative point-in-time rule:
    # forecast dated today is used
    # beginning the next calendar day.
    forecasts[
        "Available_Date"
    ] = (
        forecasts[
            "Forecast_Date"
        ]
        + pd.Timedelta(
            days=1
        )
    )

    # Change in the rolling 7-day outlook
    forecasts[
        "HDD_7D_Outlook_Change"
    ] = (
        forecasts[
            "Forecast_HDD_7D"
        ].diff()
    )

    forecasts[
        "CDD_7D_Outlook_Change"
    ] = (
        forecasts[
            "Forecast_CDD_7D"
        ].diff()
    )

    return forecasts


if __name__ == "__main__":
    forecasts = (
        fetch_all_forecasts()
    )

    forecasts.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nFirst rows:"
    )

    print(
        forecasts.head()
    )

    print(
        "\nLast rows:"
    )

    print(
        forecasts.tail()
    )

    print(
        "\nShape:"
    )

    print(
        forecasts.shape
    )

    print(
        "\nMissing values:"
    )

    print(
        forecasts.isna().sum()
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )
