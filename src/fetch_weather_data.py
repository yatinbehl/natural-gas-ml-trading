from io import StringIO

import pandas as pd
import requests


START_YEAR = 2010
END_YEAR = 2026


def build_url(year, degree_type):
    return (
        "https://ftp.cpc.ncep.noaa.gov/htdocs/"
        f"degree_days/weighted/daily_data/{year}/"
        f"Population.{degree_type}.txt"
    )


def download_file(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_degree_days(text, value_name):
    lines = text.splitlines()

    table_start = next(
        i for i, line in enumerate(lines)
        if line.startswith("Region|")
    )

    table_text = "\n".join(lines[table_start:])

    data = pd.read_csv(
        StringIO(table_text),
        sep="|"
    )

    data = data.melt(
        id_vars="Region",
        var_name="Date",
        value_name=value_name
    )

    data["Date"] = pd.to_datetime(
        data["Date"],
        format="%Y%m%d",
        errors="coerce"
    )

    data[value_name] = pd.to_numeric(
        data[value_name],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Date", value_name]
    )

    return data


def fetch_all_years():
    hdd_frames = []
    cdd_frames = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Downloading {year}...")

        hdd_text = download_file(
            build_url(year, "Heating")
        )

        cdd_text = download_file(
            build_url(year, "Cooling")
        )

        hdd = parse_degree_days(
            hdd_text,
            "HDD"
        )

        cdd = parse_degree_days(
            cdd_text,
            "CDD"
        )

        hdd = hdd[hdd["Region"] == "CONUS"]
        cdd = cdd[cdd["Region"] == "CONUS"]

        hdd_frames.append(
            hdd[["Date", "HDD"]]
        )

        cdd_frames.append(
            cdd[["Date", "CDD"]]
        )

    all_hdd = pd.concat(
        hdd_frames,
        ignore_index=True
    )

    all_cdd = pd.concat(
        cdd_frames,
        ignore_index=True
    )

    weather = pd.merge(
        all_hdd,
        all_cdd,
        on="Date",
        how="inner"
    )

    weather = weather.sort_values("Date")

    return weather


if __name__ == "__main__":
    weather_data = fetch_all_years()

    weather_data.to_csv(
        "data/ng_weather_daily.csv",
        index=False
    )

    print("\nFirst rows:")
    print(weather_data.head())

    print("\nLast rows:")
    print(weather_data.tail())

    print("\nShape:")
    print(weather_data.shape)

    print("\nMissing values:")
    print(weather_data.isna().sum())
