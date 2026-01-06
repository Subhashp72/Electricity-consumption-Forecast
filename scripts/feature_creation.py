import pandas as pd
import numpy as np
import re
import gc

###############################################################################
# Feature Engineering Utilities
###############################################################################

def add_temporal_features(
    df: pd.DataFrame,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Add standard calendar/time features from a date column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least `date_col`.
    date_col : str, default="Date"
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with additional temporal columns:
        - dow: day of week (0=Mon, 6=Sun)
        - is_weekend: 1 if Sat/Sun else 0
        - day_of_month
        - week_of_year (ISO week)
        - month, quarter, year
        - month_start: first day of the month (timestamp)
        - week_start: Monday of the same week (timestamp)
    """
    out = df.copy()

    # Parse the date column into pandas datetime
    d = pd.to_datetime(out[date_col])
    out[date_col] = d

    # Basic temporal features
    out["dow"] = d.dt.dayofweek  # 0=Mon, 6=Sun
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["day_of_month"] = d.dt.day
    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter
    out["year"] = d.dt.year

    # Useful anchors for joins / grouping
    out["month_start"] = d.dt.to_period("M").dt.to_timestamp()
    out["week_start"] = (d - pd.to_timedelta(out["dow"], unit="D")).dt.normalize()

    return out


###############################################################################
# Month-Anchored Lag Feature Builders
###############################################################################

def add_month_anchored_lag_stats(
    df,
    date_col="Date",
    meter_col="meter_id",
    y_col="daily_consumption",
    windows=(7, 15, 30, 90, 180, 365),
    qs=(0.10, 0.80),
    shift_days=1,
    min_periods=None,
    prefix="lag",
):
    """
    Compute leakage-safe rolling statistics per meter using shifted history,
    but make features constant within each month (month-anchored).

    Logic
    -----
    1) Compute rolling stats per meter on daily level (using y shifted by `shift_days`).
    2) Keep only the computed values on month start rows.
    3) Broadcast those month start feature values to all days in that month.

    Parameters
    ----------
    df : pd.DataFrame
        Daily long-format dataset (Date × meter_id).
    date_col, meter_col, y_col : str
        Column names for date, meter identifier, and target value.
    windows : tuple
        Rolling windows in days.
    qs : tuple
        Quantiles to compute.
    shift_days : int
        Shifts the target by `shift_days` to avoid leakage (history up to date-shift_days).
    min_periods : int or None
        Minimum observations required per rolling window.
        If None, defaults to `window` size.
    prefix : str
        Prefix for generated feature columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with month-anchored lag features merged back.
    """
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out[date_col] = d

    # Month anchor used for broadcasting features
    out["month_start"] = d.dt.to_period("M").dt.to_timestamp()
    out = out.sort_values([meter_col, date_col]).reset_index(drop=True)

    # Shifted series (leakage-safe)
    y_shift = out.groupby(meter_col)[y_col].shift(shift_days)

    feats = []

    # Compute rolling features for each window
    for w in windows:
        mp = w if min_periods is None else min_periods
        r = y_shift.groupby(out[meter_col]).rolling(w, min_periods=mp)

        # Common rolling stats
        for name, func in {
            "mean": r.mean,
            "min": r.min,
            "max": r.max,
            "median": r.median,
            "std": r.std,
        }.items():
            c = f"{prefix}_{name}_{w}d"
            out[c] = func().reset_index(level=0, drop=True)
            feats.append(c)

        # Quantiles
        for q in qs:
            c = f"{prefix}_p{int(q * 100)}_{w}d"
            out[c] = r.quantile(q).reset_index(level=0, drop=True)
            feats.append(c)

    # Snapshot features on month start rows
    ms = out.loc[out[date_col] == out["month_start"], [meter_col, "month_start"] + feats]

    # Drop daily rolling columns and broadcast month-start values back to all rows
    out = out.drop(columns=feats).merge(ms, on=[meter_col, "month_start"], how="left")

    return out


def add_month_anchored_lag_stats_lowmem(
    df,
    date_col="Date",
    meter_col="meter_id",
    y_col="daily_consumption",
    windows=(7, 15, 30, 90, 180, 365),
    qs=(0.10, 0.80),
    shift_days=1,
    min_periods=None,
    prefix="lag",
    use_float32=True,
):
    """
    Low-memory version of month-anchored rolling stats.

    Key optimization
    ----------------
    - Compute rolling stats on the full series.
    - Store only month-start values in a compact table `ms`.
    - Merge monthly features back to all daily rows.

    This reduces memory significantly vs keeping rolling outputs for all rows.

    Parameters
    ----------
    df : pd.DataFrame
        Daily long-format dataset (Date × meter_id).
    date_col, meter_col, y_col : str
        Column names.
    windows : tuple
        Rolling window sizes (days).
    qs : tuple
        Quantiles to compute.
    shift_days : int
        Shift target to avoid leakage.
    min_periods : int or None
        Minimum required observations per rolling window.
    prefix : str
        Prefix for generated feature names.
    use_float32 : bool
        If True, casts y_col and stored features to float32 to reduce memory.

    Returns
    -------
    pd.DataFrame
        Dataframe with month-anchored lag features merged back.
    """
    out = df.copy()

    # Parse dates and create month anchor
    out[date_col] = pd.to_datetime(out[date_col])
    out["month_start"] = out[date_col].dt.to_period("M").dt.to_timestamp()
    out = out.sort_values([meter_col, date_col]).reset_index(drop=True)

    # Reduce memory footprint of target column
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    if use_float32:
        out[y_col] = out[y_col].astype("float32")

    # Month-start mask (we only keep feature values for these rows)
    ms_mask = out[date_col].eq(out["month_start"])

    # Small table holding only month-level features
    ms = out.loc[ms_mask, [meter_col, "month_start"]].copy()

    # Shift target to avoid leakage
    y_shift = out.groupby(meter_col, sort=False)[y_col].shift(shift_days)
    if use_float32:
        y_shift = y_shift.astype("float32")

    def _add_feat(series_full, colname):
        """
        Helper to store only month-start values from a full-length feature series.
        """
        nonlocal ms
        vals = series_full.loc[ms_mask].to_numpy()
        if use_float32:
            vals = vals.astype("float32", copy=False)
        ms[colname] = vals

    for w in windows:
        mp = w if min_periods is None else min_periods
        r = y_shift.groupby(out[meter_col], sort=False).rolling(w, min_periods=mp)

        # Cheaper stats first
        _add_feat(r.mean().reset_index(level=0, drop=True), f"{prefix}_mean_{w}d")
        _add_feat(r.min().reset_index(level=0, drop=True),  f"{prefix}_min_{w}d")
        _add_feat(r.max().reset_index(level=0, drop=True),  f"{prefix}_max_{w}d")
        _add_feat(r.std().reset_index(level=0, drop=True),  f"{prefix}_std_{w}d")

        # More expensive stats last
        _add_feat(r.median().reset_index(level=0, drop=True), f"{prefix}_median_{w}d")
        for q in qs:
            _add_feat(
                r.quantile(q).reset_index(level=0, drop=True),
                f"{prefix}_p{int(q * 100)}_{w}d",
            )

        # Encourage early memory release in long loops
        gc.collect()

    # Broadcast month-level features back to all daily rows
    out = out.merge(ms, on=[meter_col, "month_start"], how="left")

    return out


###############################################################################
# Seasonality Indexes (Train-only)
###############################################################################

def compute_month_seasonality_index(
    df,
    date_col="Date",
    y_col="daily_consumption",
    data_flag_col="data_flag",
    train_flag="train",
):
    """
    Compute month-of-year seasonality index using TRAIN data only.

    Seasonality index = (mean consumption in month) / (overall mean consumption)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing date, target, and data_flag column.
    date_col, y_col : str
        Column names for date and target.
    data_flag_col : str
        Column name indicating train/test split.
    train_flag : str
        Value in data_flag_col that indicates training rows.

    Returns
    -------
    pd.DataFrame
        Columns: ['month', 'month_seasonality_index']
    """
    tmp = df.loc[df[data_flag_col] == train_flag, [date_col, y_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp["month"] = tmp[date_col].dt.month.astype(int)

    overall_mean = tmp[y_col].mean()

    month_index = (
        tmp.groupby("month")[y_col].mean()
        .div(overall_mean)
        .rename("month_seasonality_index")
        .reset_index()
    )
    return month_index


def compute_dow_seasonality_index(
    df,
    date_col="Date",
    y_col="daily_consumption",
    data_flag_col="data_flag",
    train_flag="train",
):
    """
    Compute day-of-week seasonality index using TRAIN data only.

    Seasonality index = (mean consumption on DOW) / (overall mean consumption)

    Returns
    -------
    pd.DataFrame
        Columns: ['dow', 'dow_seasonality_index']
    """
    tmp = df.loc[df[data_flag_col] == train_flag, [date_col, y_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp["dow"] = tmp[date_col].dt.dayofweek.astype(int)

    overall_mean = tmp[y_col].mean()

    dow_index = (
        tmp.groupby("dow")[y_col].mean()
        .div(overall_mean)
        .rename("dow_seasonality_index")
        .reset_index()
    )
    return dow_index


###############################################################################
# One-stop feature builder for modeling
###############################################################################

def get_model_data(df_daily):
    """
    Prepare final model-ready dataset from daily aggregated data.

    Steps
    -----
    1) Keep core columns: Date, meter_id, daily_consumption
    2) Add temporal features (dow, month, etc.)
    3) Compute seasonality indexes from TRAIN data only
    4) Merge seasonality features into full dataset
    5) Add month-anchored lag features (low-memory version)

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily aggregated data in long format.

    Returns
    -------
    pd.DataFrame
        Final model dataset with temporal, seasonality, and lag features.
    """
    # Drop any unwanted columns if present
    df_daily.drop(["Unnamed: 0"], axis=1, inplace=True, errors="ignore")

    # Keep only required columns
    df_daily = df_daily[["Date", "meter_id", "daily_consumption"]].copy()
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])

    # Add temporal features (includes month_start)
    df_daily = add_temporal_features(df_daily)

    # Compute seasonality indexes (expects data_flag; if missing, this will error)
    dow_idx = compute_dow_seasonality_index(df_daily)
    month_idx = compute_month_seasonality_index(df_daily)

    # Merge seasonality features back
    df_feat = (
        df_daily
        .assign(
            dow=pd.to_datetime(df_daily["Date"]).dt.dayofweek,
            month=pd.to_datetime(df_daily["Date"]).dt.month,
        )
        .merge(dow_idx, on="dow", how="left")
        .merge(month_idx, on="month", how="left")
    )

    # Free up memory early
    del df_daily, dow_idx, month_idx
    gc.collect()

    # Add month-anchored lag stats (low memory)
    model_data = add_month_anchored_lag_stats_lowmem(df_feat)

    return model_data
