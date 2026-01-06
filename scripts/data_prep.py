import pandas as pd
import numpy as np

def prepare_daily_meter_data(
    file_path: str,
    delim: str = ";",
    meter_prefix: str = "MT_",
    end_date: str | None = "2015-01-01",
):
    df = pd.read_csv(file_path, sep=delim, low_memory=False)
    df = df.rename(columns={"Unnamed: 0": "timestamp"}, errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()]

    meter_cols = [c for c in df.columns if c.startswith(meter_prefix)]

    df = df.melt(
        id_vars="timestamp",
        value_vars=meter_cols,
        var_name="meter_id",
        value_name="consumption",
    )

    df["consumption"] = (
        df["consumption"].astype(str)
        .str.replace(",", "", regex=False)
        .replace("", np.nan)
        .astype(float)
    )

    df["Date"] = df["timestamp"].dt.date.astype(str)
    df["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M").astype(str)

    df = (
        df.groupby(["Month", "Date", "meter_id"], observed=True)
        .agg(
            daily_consumption=("consumption", "sum"),
            n_intervals=("consumption", "count"),
            n_missing=("consumption", lambda x: x.isna().sum()),
        )
        .reset_index()
        .drop_duplicates()
    )

    if end_date:
        df = df[df["Date"] < end_date]

    first_active = (
        df[df["daily_consumption"] > 0]
        .groupby("meter_id")["Date"]
        .min()
        .rename("first_active_date")
        .reset_index()
    )

    df = df.merge(first_active, on="meter_id", how="left")
    df = df[df["first_active_date"].notna()]
    df = df[df["Date"] >= df["first_active_date"]]

    return df
