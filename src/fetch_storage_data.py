import os

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()


def fetch_storage_data():
    api_key = os.getenv("EIA_API_KEY")

    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"

    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "NW2_EPG0_SWO_R48_BCF",
        "start": "2010-01-01",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    rows = response.json()["response"]["data"]

    data = pd.DataFrame(rows)

    data = data[["period", "value"]].copy()

    data["period"] = pd.to_datetime(data["period"])
    data["value"] = pd.to_numeric(data["value"])

    data = data.rename(columns={
        "period": "Week_Ending_Date",
        "value": "Storage_Bcf"
    })

    data["Weekly_Change_Bcf"] = data["Storage_Bcf"].diff()

    data["Release_Date"] = (
        data["Week_Ending_Date"] + pd.Timedelta(days=6)
    )

    data["Available_Date"] = (
        data["Release_Date"] + pd.Timedelta(days=1)
    )

    return data


if __name__ == "__main__":
    storage_data = fetch_storage_data()

    storage_data.to_csv(
        "data/ng_storage_weekly.csv",
        index=False
    )

    print(storage_data.head())
    print(storage_data.tail())
    print("\nShape:")
    print(storage_data.shape)
    print("\nColumns:")
    print(storage_data.columns)
