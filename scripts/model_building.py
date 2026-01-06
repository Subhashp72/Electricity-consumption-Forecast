import pandas as pd
import numpy as np
import config, feature_creation
from tqdm import tqdm
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

import statsmodels.api as sm
import matplotlib.pyplot as plt
# import lightgbm as lgb
tqdm.pandas()
#########################
def split_train_test_val_dataframes(model_data, random_state= 42, valid_frac = 0.15):
    model_data = model_data[(model_data.Date>=config.training_time_period[0]) & (model_data.Date<config.test_time_period[1])]
    model_data['data_flag'] = np.where(model_data.Date<= config.training_time_period[1], "train", "test")
    # ensure reproducibility
    RANDOM_STATE = 42
    VALID_FRAC = 0.15

    # initialize valid flag only within train
    train_mask = model_data["data_flag"] == "train"

    # sample indices from train only
    valid_idx = (
        model_data.loc[train_mask]
        .sample(frac=valid_frac, random_state=random_state)
        .index
    )

    # update flag in-place
    model_data.loc[valid_idx, "data_flag"] = "valid"
    return model_data

def get_feature_var_and_correlations(model_features: pd.DataFrame, target_col: str):
    """
    Returns:
      variance_df: Feature-wise variance (numeric only)
      corr_df: Pairwise correlations among features (excluding target), unique pairs
      corr_with_target: Correlations of each feature with target
    """

    num_df = model_features.select_dtypes(include=np.number)

    # --- Variance (NaN-safe) ---
    # Option A (fastest, no tqdm needed):
    var_s = num_df.var(axis=0, ddof=1, skipna=True)

    # If you really want tqdm progress (slower than vectorized):
    # var_s = num_df.progress_apply(lambda s: s.var(ddof=1, skipna=True), axis=0)

    variance_df = var_s.reset_index()
    variance_df.columns = ["Feature", "Variance"]

    # --- Correlations ---
    corr = num_df.corr()  # Pearson by default; handles NaNs pairwise
    corr_df = (
        corr.stack()
            .reset_index()
            .rename(columns={"level_0": "var1", "level_1": "var2", 0: "correlation_coefficient"})
    )

    # Remove self-correlation + duplicate pairs (A,B) vs (B,A)
    corr_df = corr_df[corr_df["var1"] != corr_df["var2"]].reset_index(drop=True)
    corr_df["pair"] = corr_df[["var1", "var2"]].apply(lambda x: tuple(sorted(x)), axis=1)
    corr_df = corr_df.drop_duplicates("pair").drop(columns="pair").reset_index(drop=True)

    # --- Ensure must_include_features appear as var1 (swap where needed) ---
    mask = corr_df["var2"].isin(getattr(config, "must_include_features", [])) & ~corr_df["var1"].isin(getattr(config, "must_include_features", []))
    corr_df.loc[mask, ["var1", "var2"]] = corr_df.loc[mask, ["var2", "var1"]].to_numpy()

    # --- Correlation with target ---
    corr_with_target = corr_df[(corr_df["var1"] == target_col) | (corr_df["var2"] == target_col)].copy()

    # Ensure target is always in var1
    swap_mask = corr_with_target["var2"] == target_col
    corr_with_target.loc[swap_mask, ["var1", "var2"]] = corr_with_target.loc[swap_mask, ["var2", "var1"]].to_numpy()
    corr_with_target = corr_with_target.drop_duplicates().reset_index(drop=True)

    # Remove target pairs from corr_df
    corr_df = corr_df[(corr_df["var1"] != target_col) & (corr_df["var2"] != target_col)].reset_index(drop=True)

    return variance_df, corr_df, corr_with_target
###########
def impute_lag_features_using_train_only(
    df: pd.DataFrame,
    lag_prefix: str = "lag_",
    meter_col: str = "meter_id",
    data_flag_col: str = "data_flag",
    train_value: str = "train",
    strategy: str = "median",   # "median" or "mean"
    add_missing_flag: bool = True,
) -> pd.DataFrame:
    """
    Impute missing lag / rolling features using TRAIN DATA ONLY.

    Steps:
    1) Compute meter-level stats on train subset
    2) Compute global fallback stats on train subset
    3) Apply imputation to full dataset (train/valid/test)
    4) Optionally add missing indicators

    Leakage-safe by construction.
    """
    out = df.copy()

    lag_cols = [c for c in out.columns if c.startswith(lag_prefix)]
    if not lag_cols:
        return out

    train_df = out[out[data_flag_col] == train_value]

    for col in lag_cols:
        # Missing indicator (computed on full data, OK)
        if add_missing_flag:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)

        # --- Meter-level stats from TRAIN only ---
        if strategy == "median":
            meter_stat_map = (
                train_df
                .groupby(meter_col)[col]
                .median()
            )
            global_stat = train_df[col].median()
        else:
            meter_stat_map = (
                train_df
                .groupby(meter_col)[col]
                .mean()
            )
            global_stat = train_df[col].mean()

        # Map meter-level stats to full df
        meter_stat_full = out[meter_col].map(meter_stat_map)

        # Fill order: meter-level → global
        out[col] = out[col].fillna(meter_stat_full)
        out[col] = out[col].fillna(global_stat)

    return out
####################################

def feature_funnel_selector(
    model_data: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    data_flag_col: str = "data_flag",
    train_value: str = "train",
    sample_n: int | None = 200_000,
    random_state: int = 42,
):
    """
    Train-only feature funnel using thresholds from config:
      1) Drop high missing%
      2) Drop low variance
      3) Drop low |corr(target)|
      4) Drop highly correlated feature pairs (keep higher |corr(target)|)

    Returns:
      final_features, filtered_log_df, stats_df
    """
    if 'data_flag' not in model_data.columns:
        model_data = split_train_test_val_dataframes(model_data)
    model_data.drop(columns=['dow_seasonality_index','month_seasonality_index' ], inplace=True, errors= 'ignore')
    dow_idx = feature_creation.compute_dow_seasonality_index(model_data)
    month_idx = feature_creation.compute_month_seasonality_index(model_data)

    model_data = (
        model_data
        .assign(
            dow=pd.to_datetime(model_data["Date"]).dt.dayofweek,
            month=pd.to_datetime(model_data["Date"]).dt.month
        )
        .merge(dow_idx, on="dow", how="left")
        .merge(month_idx, on="month", how="left")
        )
    # thresholds from config
    missing_perc_threshold = config.missing_perc_threshold
    variance_threshold = config.variance_threshold
    corr_with_target_threshold = config.correlation_with_target_threshold
    corr_between_features_threshold = getattr(config, "corr_between_features_threshold", 0.95)
    must_include_features = getattr(config, "must_include_features", [])

    filtered_log = []

    # ---- Train-only slice ----
    df_train = model_data[model_data[data_flag_col] == train_value].copy()

    if feature_cols is None:
        feature_cols = [
            c for c in df_train.select_dtypes(include=np.number).columns
            if c != target_col
        ]

    if sample_n is not None and len(df_train) > sample_n:
        df_train = df_train.sample(n=sample_n, random_state=random_state)

    df_train = df_train[[target_col] + feature_cols]

    # ---- 1) Missing% ----
    missing_pct = df_train[feature_cols].isna().mean()
    high_missing = missing_pct[missing_pct > missing_perc_threshold].index.tolist()

    for f in high_missing:
        filtered_log.append({"Feature": f, "Reason": "High Missing", "Value": float(missing_pct[f])})

    candidates = [f for f in feature_cols if (f not in high_missing) or (f in must_include_features)]

    # ---- 2) Variance ----
    var_s = df_train[candidates].var(ddof=1, skipna=True)
    low_var = var_s[var_s < variance_threshold].index.tolist()

    for f in low_var:
        if f not in must_include_features:
            filtered_log.append({"Feature": f, "Reason": "Low Variance", "Value": float(var_s[f])})

    candidates = [f for f in candidates if (f not in low_var) or (f in must_include_features)]

    # ---- 3) Corr with target ----
    corr_with_target = df_train[candidates].corrwith(df_train[target_col]).fillna(0.0)
    low_corr = corr_with_target[corr_with_target.abs() < corr_with_target_threshold].index.tolist()

    for f in low_corr:
        if f not in must_include_features:
            filtered_log.append({"Feature": f, "Reason": "Low corr with target", "Value": float(corr_with_target[f])})

    candidates = [f for f in candidates if (f not in low_corr) or (f in must_include_features)]

    # ---- 4) Inter-feature correlation pruning ----
    X = df_train[candidates]
    corr_mat = X.corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        if col in to_drop or col in must_include_features:
            continue

        partners = upper.index[upper[col] > corr_between_features_threshold].tolist()
        for other in partners:
            if other in to_drop or other in must_include_features:
                continue

            c1 = abs(corr_with_target.get(col, 0.0))
            c2 = abs(corr_with_target.get(other, 0.0))

            drop_feat = other if c1 >= c2 else col
            keep_feat = col if drop_feat == other else other

            if drop_feat in must_include_features:
                drop_feat = keep_feat
                keep_feat = [x for x in [col, other] if x != drop_feat][0]

            if drop_feat not in must_include_features:
                to_drop.add(drop_feat)
                filtered_log.append({
                    "Feature": drop_feat,
                    "Reason": f"High corr with {keep_feat}",
                    "Value": float(corr_mat.loc[drop_feat, keep_feat])
                })

            if col in to_drop:
                break

    final_features = [f for f in candidates if f not in to_drop]
    # ensure must-includes are present if they exist in original feature list
    for f in must_include_features:
        if f in feature_cols and f not in final_features:
            final_features.append(f)

    # ---- Stats (train-only) ----
    stats_df = pd.DataFrame({
        "Feature": feature_cols,
        "Missing%_train": missing_pct.reindex(feature_cols).values,
        "Variance_train": var_s.reindex(feature_cols).values,
        "CorrWithTarget_train": corr_with_target.reindex(feature_cols).values,
    })

    filtered_log_df = (
        pd.DataFrame(filtered_log)
        .drop_duplicates(subset=["Feature", "Reason"], keep="first")
        .reset_index(drop=True)
    )

    return final_features, filtered_log_df, stats_df, model_data
#####################################################
def wape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


def train_models_log_target_with_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "daily_consumption",
    data_flag_col: str = "data_flag",
    train_flag: str = "train",
    test_flag: str = "test",
    pred_prefix: str = "pred",
    random_state: int = 42,
    # permutation importance controls (for HGBR; optional for others)
    compute_perm_importance_gbm: bool = True,
    perm_sample_n: int = 50_000,
    perm_n_repeats: int = 3,
):
    """
    Trains 4 models with log1p(target), saves predictions (original scale),
    returns metrics (R2 on log scale, MAE/WAPE on original scale),
    and returns feature importance for each model.

    Returns:
      df_out, metrics_df, importance_df, models_dict
    """
    df_out = df.copy()

    # ---- checks ----
    needed = [target_col, data_flag_col] + feature_cols
    missing = [c for c in needed if c not in df_out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # numeric safety
    df_out[target_col] = pd.to_numeric(df_out[target_col], errors="coerce")
    for c in feature_cols:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    train_mask = df_out[data_flag_col].eq(train_flag)
    test_mask = df_out[data_flag_col].eq(test_flag)

    X_train = df_out.loc[train_mask, feature_cols]
    y_train_log = np.log1p(df_out.loc[train_mask, target_col].values)

    X_test = df_out.loc[test_mask, feature_cols]
    y_test_log = np.log1p(df_out.loc[test_mask, target_col].values)

    y_train = df_out.loc[train_mask, target_col].values
    y_test = df_out.loc[test_mask, target_col].values

    metrics_rows = []
    imp_rows = []
    models = {}

    # ---- OLS (statsmodels) ----
    imp = SimpleImputer(strategy="median")
    Xtr_lr = pd.DataFrame(imp.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    Xte_lr = pd.DataFrame(imp.transform(X_test), columns=feature_cols, index=X_test.index)

    Xtr_lr_c = sm.add_constant(Xtr_lr, has_constant="add")
    Xte_lr_c = sm.add_constant(Xte_lr, has_constant="add")

    ols = sm.OLS(y_train_log, Xtr_lr_c.values).fit()
    models["linear_regression_ols"] = ols

    pred_lr_tr = np.expm1(ols.predict(Xtr_lr_c.values))
    pred_lr_te = np.expm1(ols.predict(Xte_lr_c.values))

    df_out[f"{pred_prefix}_lr"] = np.nan
    df_out.loc[train_mask, f"{pred_prefix}_lr"] = pred_lr_tr
    df_out.loc[test_mask, f"{pred_prefix}_lr"] = pred_lr_te

    # OLS importance: |t-stat| (exclude intercept)
    params = pd.Series(ols.params, index=["const"] + feature_cols)
    tvals = pd.Series(ols.tvalues, index=["const"] + feature_cols)
    ols_imp = tvals.drop("const").abs().rename("importance").reset_index()
    ols_imp.columns = ["feature", "importance"]
    ols_imp["model"] = "linear_regression_ols"
    ols_imp["method"] = "abs_tstat"
    imp_rows.append(ols_imp)

    metrics_rows += [
        {
            "model": "linear_regression_ols",
            "split": "train",
            "r2": float(ols.rsquared),  # on log scale
            "mae": float(mean_absolute_error(y_train, pred_lr_tr)),
            "wape": float(wape(y_train, pred_lr_tr)),
        },
        {
            "model": "linear_regression_ols",
            "split": "test",
            "r2": float(r2_score(y_test_log, np.log1p(pred_lr_te))),
            "mae": float(mean_absolute_error(y_test, pred_lr_te)),
            "wape": float(wape(y_test, pred_lr_te)),
        },
    ]

    # ---- helper for pipelines ----
    def fit_pipeline(model, name, pred_col_suffix, want_builtin_importance=True, want_perm_importance=False):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])
        pipe.fit(X_train, y_train_log)
        models[name] = pipe

        pred_tr = np.expm1(pipe.predict(X_train))
        pred_te = np.expm1(pipe.predict(X_test))

        col = f"{pred_prefix}_{pred_col_suffix}"
        df_out[col] = np.nan
        df_out.loc[train_mask, col] = pred_tr
        df_out.loc[test_mask, col] = pred_te

        metrics_rows.extend([
            {
                "model": name,
                "split": "train",
                "r2": float(r2_score(y_train_log, np.log1p(pred_tr))),
                "mae": float(mean_absolute_error(y_train, pred_tr)),
                "wape": float(wape(y_train, pred_tr)),
            },
            {
                "model": name,
                "split": "test",
                "r2": float(r2_score(y_test_log, np.log1p(pred_te))),
                "mae": float(mean_absolute_error(y_test, pred_te)),
                "wape": float(wape(y_test, pred_te)),
            },
        ])

        # --- importance ---
        m = pipe.named_steps["model"]

        if want_builtin_importance and hasattr(m, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": feature_cols,
                "importance": m.feature_importances_,
                "model": name,
                "method": "built_in",
            })
            imp_rows.append(fi)

        if want_perm_importance:
            # compute on a sampled TEST set (more realistic)
            Xp = X_test.copy()
            yp = y_test_log.copy()

            if perm_sample_n is not None and len(Xp) > perm_sample_n:
                Xp = Xp.sample(n=perm_sample_n, random_state=random_state)
                yp = pd.Series(yp, index=X_test.index).loc[Xp.index].values

            perm = permutation_importance(
                pipe, Xp, yp,
                n_repeats=perm_n_repeats,
                random_state=random_state,
                n_jobs=-1,
            )
            fi = pd.DataFrame({
                "feature": feature_cols,
                "importance": perm.importances_mean,
                "importance_std": perm.importances_std,
                "model": name,
                "method": "permutation",
            }).sort_values("importance", ascending=False)
            imp_rows.append(fi)

    # ---- Random Forest ----
    fit_pipeline(
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=50,
            n_jobs=-1,
            random_state=random_state,
        ),
        name="random_forest",
        pred_col_suffix="rf",
        want_builtin_importance=True,
        want_perm_importance=False,
    )

    # ---- sklearn GBM (HistGradientBoosting) ----
    fit_pipeline(
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=64,
            min_samples_leaf=200,
            early_stopping=True,
            random_state=random_state,
        ),
        name="sklearn_gbm",
        pred_col_suffix="gbm",
        want_builtin_importance=False,
        want_perm_importance=compute_perm_importance_gbm,  # ✅ optional
    )

    # ---- XGBoost (optional) ----
    try:
        from xgboost import XGBRegressor
        fit_pipeline(
            XGBRegressor(
                n_estimators=3000,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            ),
            name="xgboost",
            pred_col_suffix="xgb",
            want_builtin_importance=True,
            want_perm_importance=False,
        )
    except Exception as e:
        metrics_rows += [
            {"model": "xgboost", "split": "train", "r2": np.nan, "mae": np.nan, "wape": np.nan},
            {"model": "xgboost", "split": "test", "r2": np.nan, "mae": np.nan, "wape": np.nan},
        ]
        # also note the failure in importance
        imp_rows.append(pd.DataFrame({
            "feature": feature_cols,
            "importance": [np.nan] * len(feature_cols),
            "model": ["xgboost"] * len(feature_cols),
            "method": ["unavailable"] * len(feature_cols),
        }))

    metrics_df = pd.DataFrame(metrics_rows)
    importance_df = pd.concat(imp_rows, ignore_index=True)

    return df_out, metrics_df, importance_df, models
###########################
def save_models(models):
    # ---------- Save models ----------
    saved = {}
    for name, obj in models.items():
        if name.endswith("_error"):
            continue
        if obj is None:
            saved[name] = None
            continue

        path = config.model_dir / f"{name}.pkl"

        # statsmodels results have .save()
        if hasattr(obj, "save") and callable(getattr(obj, "save")):
            # Use statsmodels native saver (writes pickle internally)
            obj.save(str(path))
        else:
            joblib.dump(obj, path)

        saved[name] = str(path)
###########################################
def plot_actual_vs_predicted(
    df: pd.DataFrame,
    date_col: str = "Date",
    target_col: str = "daily_consumption",
    pred_col: str = "pred_gbm",
    meter_col: str = "meter_id",
    data_flag_col: str = "data_flag",
    meter_id: str | None = None,
    aggregate: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    max_points: int = 500,
    title: str | None = None,
):
    """
    Plot Actual vs Predicted values over time.

    Options:
    - meter_id: plot a single meter
    - aggregate=True: plot total consumption across meters
    - start_date / end_date: date filtering
    """

    plot_df = df.copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])

    # Filter dates
    if start_date:
        plot_df = plot_df[plot_df[date_col] >= start_date]
    if end_date:
        plot_df = plot_df[plot_df[date_col] <= end_date]

    # Single meter
    if meter_id is not None:
        plot_df = plot_df[plot_df[meter_col] == meter_id]

    # Aggregate across meters
    if aggregate:
        plot_df = (
            plot_df.groupby([date_col, data_flag_col], observed=True)
            .agg(
                actual=(target_col, "sum"),
                predicted=(pred_col, "sum"),
            )
            .reset_index()
        )
    else:
        plot_df = plot_df.rename(
            columns={target_col: "actual", pred_col: "predicted"}
        )

    # Sort
    plot_df = plot_df.sort_values(date_col)

    # Downsample if needed
    if len(plot_df) > max_points:
        plot_df = plot_df.iloc[:: len(plot_df) // max_points]

    # Plot
    plt.figure(figsize=(14, 5))

    plt.plot(plot_df[date_col], plot_df["actual"], label="Actual", linewidth=2)
    plt.plot(plot_df[date_col], plot_df["predicted"], label="Predicted", linestyle="--")

    # Mark train/test split
    if data_flag_col in plot_df.columns:
        test_start = plot_df.loc[
            plot_df[data_flag_col] == "test", date_col
        ].min()
        if pd.notna(test_start):
            plt.axvline(test_start, color="red", linestyle=":", label="Test start")

    plt.xlabel("Date")
    plt.ylabel("Consumption")
    plt.title(title or f"Actual vs Predicted ({pred_col})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
