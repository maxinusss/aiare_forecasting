FRED_API_KEY = "40570fb17d48c75549e679b52d87522c"

import os
import requests
import pandas as pd
from pathlib import Path
import pandas as pd
import numpy as np

# Set working directory to project root (parent of eda folder)
os.chdir(Path(__file__).parent.parent)


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
# FRED_API_KEY = os.environ.get("FRED_API_KEY", "YOUR_FRED_API_KEY")

SERIES = {
    "consumer_sentiment": "UMCSENT",   # University of Michigan Consumer Sentiment
    "unemployment_rate": "UNRATE",     # Unemployment Rate
    "cpi": "CPIAUCSL",                 # CPI
    "gas_price": "GASREGW",            # Weekly gas price
}

START_DATE = "2017-01-01"
END_DATE = "2026-12-31"


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def fetch_fred_series(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    observations = payload.get("observations", [])
    df = pd.DataFrame(observations)

    if df.empty:
        return pd.DataFrame(columns=["date", "value"])

    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def aggregate_to_month(df: pd.DataFrame, value_col: str, agg: str = "mean") -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month

    if agg == "mean":
        result = (
            out.groupby(["year", "month"], as_index=False)[value_col]
            .mean()
        )
    elif agg == "sum":
        result = (
            out.groupby(["year", "month"], as_index=False)[value_col]
            .sum()
        )
    elif agg == "last":
        result = (
            out.sort_values("date")
            .groupby(["year", "month"], as_index=False)[value_col]
            .last()
        )
    else:
        raise ValueError(f"Unsupported agg: {agg}")

    result["year_month"] = pd.to_datetime(
        dict(year=result["year"], month=result["month"], day=1)
    )
    return result


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    monthly_frames = []

    for column_name, series_id in SERIES.items():
        df = fetch_fred_series(series_id, START_DATE, END_DATE)
        df = df.rename(columns={"value": column_name})

        # Mean is a good default for monthly aggregation
        monthly_df = aggregate_to_month(df, column_name, agg="mean")
        monthly_frames.append(monthly_df)

    features = monthly_frames[0]
    for frame in monthly_frames[1:]:
        features = features.merge(frame, on=["year", "month", "year_month"], how="outer")

    features = features.sort_values(["year", "month"]).reset_index(drop=True)

    # Optional: YoY CPI inflation by month
    features["cpi_yoy_pct"] = features["cpi"].pct_change(12) * 100

    # Optional: simple economic pressure index
    z_cols = ["gas_price", "cpi_yoy_pct", "unemployment_rate", "consumer_sentiment"]
    for col in z_cols:
        if col in features.columns:
            std = features[col].std()
            if pd.notna(std) and std != 0:
                features[f"{col}_z"] = (features[col] - features[col].mean()) / std
            else:
                features[f"{col}_z"] = 0.0

    needed = {
        "gas_price_z",
        "cpi_yoy_pct_z",
        "unemployment_rate_z",
        "consumer_sentiment_z",
    }
    if needed.issubset(features.columns):
        features["economic_pressure_index"] = (
            features["gas_price_z"]
            + features["cpi_yoy_pct_z"]
            + features["unemployment_rate_z"]
            - features["consumer_sentiment_z"]
        )

    print(features.to_string(index=False))
    features.to_csv("data/cleaned_data/monthly_economic_features.csv", index=False)


if __name__ == "__main__":
    main()

