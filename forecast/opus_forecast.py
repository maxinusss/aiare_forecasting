"""
Opus forecast: improved forecasting pipeline for AIARE course enrollment.

Key fixes over chat_3_forecast_full_restored.py:
1. Direct multi-step forecasting (no recursive lag degradation)
2. Log1p transform to handle extreme skew (0 to 4768 students)
3. Multiplicative ETS seasonality
4. Seasonal naive baseline
5. Ensemble of top models weighted by CV performance
6. SARIMAX support
7. Calendar-only features for future periods (no lag leakage)
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# Optional models
PROPHET_AVAILABLE = True
XGBOOST_AVAILABLE = True
CATBOOST_AVAILABLE = True

try:
    from prophet import Prophet
except Exception:
    PROPHET_AVAILABLE = False

try:
    from xgboost import XGBRegressor
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
except Exception:
    CATBOOST_AVAILABLE = False


# =========================
# User-configurable inputs
# =========================
INPUT_CSV = "master_data_full.csv"
TARGET_COL = "num_students"
COURSE_COL = "combined_course"
DATE_COL = "date"
FORECAST_END = "2027-08-01"
OUTPUT_DIR = "opus_analysis"
REPORTING_LAG_MONTHS = 2
# Number of top models to ensemble (set to 1 to disable ensembling)
ENSEMBLE_TOP_K = 3


# =========================
# Metrics / helpers
# =========================
def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    out = np.abs((y_true - y_pred) / denom)
    return np.nanmean(out) * 100.0


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def season_from_month(month):
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "fall"


def sanitize_name(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def clean_features(df):
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    numeric_cols = out.select_dtypes(include=[np.number, "bool"]).columns
    if len(numeric_cols) > 0:
        out[numeric_cols] = out[numeric_cols].clip(lower=-1e6, upper=1e6)
    return out


def get_enso_schema(df):
    temp = df.copy()
    if "enso_outlook" not in temp.columns:
        temp["enso_outlook"] = "unknown"
    ser = (
        temp["enso_outlook"]
        .astype("object")
        .where(pd.notna(temp["enso_outlook"]), "unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )
    dummies = pd.get_dummies(ser, prefix="enso")
    return sorted(dummies.columns.tolist())


def align_enso_dummies(df, enso_schema):
    out = df.copy()
    if "enso_outlook" not in out.columns:
        out["enso_outlook"] = "unknown"
    ser = (
        out["enso_outlook"]
        .astype("object")
        .where(pd.notna(out["enso_outlook"]), "unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )
    dummies = pd.get_dummies(ser, prefix="enso")
    if enso_schema is None:
        enso_schema = sorted(dummies.columns.tolist())
    for col in enso_schema:
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[enso_schema]
    out = pd.concat([out.drop(columns=["enso_outlook"], errors="ignore"), dummies], axis=1)
    return out


# =========================
# Feature engineering
# =========================
def add_time_features(df):
    out = df.copy()
    out["month_num"] = out.index.month
    out["year_num"] = out.index.year
    out["quarter"] = out.index.quarter
    out["sin_month_12"] = np.sin(2 * np.pi * out["month_num"] / 12.0)
    out["cos_month_12"] = np.cos(2 * np.pi * out["month_num"] / 12.0)
    out["sin_month_6"] = np.sin(2 * np.pi * out["month_num"] / 6.0)
    out["cos_month_6"] = np.cos(2 * np.pi * out["month_num"] / 6.0)
    out["is_winter"] = out["month_num"].isin([12, 1, 2]).astype(int)
    out["is_spring"] = out["month_num"].isin([3, 4, 5]).astype(int)
    out["is_summer"] = out["month_num"].isin([6, 7, 8]).astype(int)
    out["is_fall"] = out["month_num"].isin([9, 10, 11]).astype(int)
    out["is_core_season"] = out["month_num"].isin([11, 12, 1, 2, 3, 4]).astype(int)
    out["days_in_month"] = out.index.days_in_month
    out["time_idx"] = np.arange(len(out))
    out["time_idx_sq"] = out["time_idx"] ** 2

    out["post_2017_18_reliable"] = (out.index >= pd.Timestamp("2017-07-01")).astype(int)
    out["post_pro_rec_split"] = (out.index >= pd.Timestamp("2018-07-01")).astype(int)
    out["post_roster_timing_change"] = (out.index >= pd.Timestamp("2020-07-01")).astype(int)
    out["covid_year_window"] = (
        (out.index >= pd.Timestamp("2020-03-01"))
        & (out.index <= pd.Timestamp("2022-06-01"))
    ).astype(int)

    out["season_name"] = [season_from_month(m) for m in out.index.month]
    out["season_year"] = np.where(out.index.month >= 7, out.index.year + 1, out.index.year)
    return clean_features(out)


def add_lag_features(df, target_col):
    """Add lag features — only useful for in-sample / backtest with known history."""
    out = df.copy()
    for lag in [1, 2, 3, 6, 12]:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    for window in [3, 6, 12]:
        shifted = out[target_col].shift(1)
        out[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        out[f"rolling_std_{window}"] = shifted.rolling(window).std()
        out[f"rolling_max_{window}"] = shifted.rolling(window).max()
        out[f"rolling_min_{window}"] = shifted.rolling(window).min()
    out["yoy_change"] = out[target_col].shift(1) - out[target_col].shift(13)
    denom = out[target_col].shift(13).replace(0, np.nan)
    out["yoy_ratio"] = out[target_col].shift(1) / denom
    out["lag1_vs_roll3"] = out[target_col].shift(1) - out[target_col].shift(1).rolling(3).mean()
    return clean_features(out)


def add_monthly_profile_features(df, target_col):
    """
    Add historical monthly averages as features — these are known for future months
    and carry strong seasonal signal without requiring lagged target values.
    """
    out = df.copy()
    month_means = out.groupby(out.index.month)[target_col].transform("mean")
    month_medians = out.groupby(out.index.month)[target_col].transform("median")
    out["month_hist_mean"] = month_means
    out["month_hist_median"] = month_medians
    return out


def prepare_monthly_series(raw, course_name, target_col):
    data = raw[raw[COURSE_COL] == course_name].copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    data = data.sort_values(DATE_COL)

    full_idx = pd.date_range(data[DATE_COL].min(), data[DATE_COL].max(), freq="MS")
    monthly = data.set_index(DATE_COL).sort_index().reindex(full_idx)
    monthly.index.name = DATE_COL

    monthly[COURSE_COL] = course_name
    monthly[target_col] = monthly[target_col].fillna(0)

    numeric_context = [
        "covid_flag",
        "cms_loss_flag",
        "unemployment_rate",
        "cpi",
        "gas_price",
        "economic_pressure_index",
        "provider_roster_upload_timing_days",
        "provider_student_fees_delay_days",
    ]
    for col in numeric_context:
        if col in monthly.columns:
            monthly[col] = monthly[col].ffill().bfill()

    if "enso_outlook" not in monthly.columns:
        monthly["enso_outlook"] = "unknown"
    monthly["enso_outlook"] = (
        monthly["enso_outlook"]
        .astype("object")
        .where(pd.notna(monthly["enso_outlook"]), "unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
        .ffill()
        .bfill()
        .fillna("unknown")
    )

    monthly = add_time_features(monthly)

    max_dt = monthly.index.max()
    monthly["is_recent_reporting_window"] = (
        monthly.index >= (max_dt - pd.DateOffset(months=REPORTING_LAG_MONTHS - 1))
    ).astype(int)

    if "cms_loss_flag" not in monthly.columns:
        monthly["cms_loss_flag"] = (monthly.index >= pd.Timestamp("2023-07-01")).astype(int)

    if "covid_flag" not in monthly.columns:
        monthly["covid_flag"] = monthly["covid_year_window"]

    return clean_features(monthly)


# Columns from the raw CSV that won't exist for future months — always drop these
RAW_ONLY_COLS = [
    "month", "year", "enrolled", "mean_student_price",
]


def build_direct_feature_matrix(df, target_col, enso_schema=None, use_log=True):
    """
    Build feature matrix for DIRECT forecasting (no lag features).
    Uses only calendar features and exogenous variables that are known for future months.
    """
    features = df.copy()
    features = align_enso_dummies(features, enso_schema)

    if "season_name" in features.columns:
        dummies = pd.get_dummies(features["season_name"], prefix="season", dummy_na=False)
        features = pd.concat([features.drop(columns=["season_name"]), dummies], axis=1)

    drop_cols = [COURSE_COL, target_col]
    # Drop lag features — they won't be available for future months
    lag_cols = [c for c in features.columns if c.startswith(("lag_", "rolling_", "yoy_", "lag1_vs_"))]
    drop_cols.extend(lag_cols)
    # Drop raw CSV columns that won't exist in future data
    drop_cols.extend(RAW_ONLY_COLS)

    X = features.drop(columns=[c for c in drop_cols if c in features.columns], errors="ignore")

    if use_log:
        y = np.log1p(features[target_col].clip(lower=0)).copy()
        # Log-transform profile features too so they're on the same scale as the target
        for col in ["month_hist_mean", "month_hist_median"]:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))
    else:
        y = features[target_col].copy()

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X = X.select_dtypes(include=[np.number, "bool"]).copy()
    X = clean_features(X)
    return X, y


# =========================
# CV splits
# =========================
def time_series_cv_splits(n_obs, initial_train=36, horizon=6, step=3, min_splits=3):
    splits = []
    train_end = initial_train
    while train_end + horizon <= n_obs:
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, train_end + horizon)
        splits.append((train_idx, val_idx))
        train_end += step

    if len(splits) < min_splits and n_obs >= initial_train + horizon:
        splits = []
        for te in range(initial_train, n_obs - horizon + 1):
            train_idx = np.arange(0, te)
            val_idx = np.arange(te, te + horizon)
            splits.append((train_idx, val_idx))
            if len(splits) >= min_splits:
                break
    return splits


# =========================
# Model implementations
# =========================
def seasonal_naive_forecast(train_y, forecast_steps):
    """Predict using last year's same month. Best baseline for strong seasonality."""
    preds = []
    n = len(train_y)
    for h in range(forecast_steps):
        # Look back 12 months from the forecast point
        lookback_idx = n - 12 + (h % 12)
        if lookback_idx >= 0:
            preds.append(max(0.0, float(train_y.iloc[lookback_idx])))
        else:
            preds.append(max(0.0, float(train_y.mean())))
    return np.array(preds)


def fit_predict_ets(train_y, forecast_steps, trend, seasonal, damped_trend):
    try:
        model = ExponentialSmoothing(
            train_y.astype(float).clip(lower=0.1),  # small floor for multiplicative
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=12 if seasonal else None,
            damped_trend=damped_trend,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True, use_brute=True)
        pred = fitted.forecast(forecast_steps)
        return np.clip(np.asarray(pred, dtype=float), 0.0, None), fitted
    except Exception:
        # Fallback if multiplicative fails (e.g. zeros in data)
        if seasonal == "mul" or trend == "mul":
            return fit_predict_ets(
                train_y, forecast_steps,
                trend="add" if trend else None,
                seasonal="add" if seasonal else None,
                damped_trend=damped_trend,
            )
        raise


def fit_predict_sarimax(train_y, forecast_steps, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12)):
    """SARIMAX: captures seasonal patterns via seasonal differencing."""
    model = SARIMAX(
        train_y.astype(float),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=200)
    pred = fitted.forecast(forecast_steps)
    return np.clip(np.asarray(pred, dtype=float), 0.0, None), fitted


def direct_forecast_sklearn(model, train_df, future_df, target_col, enso_schema, use_log=True):
    """
    Direct forecasting: train on all historical data, predict all future months at once.
    Uses only calendar/exogenous features (no lags) so there's no error compounding.
    """
    # Add monthly profile features to train
    train_with_profile = add_monthly_profile_features(train_df.copy(), target_col)

    # Compute monthly stats from training data to apply to future
    month_stats = train_df.groupby(train_df.index.month)[target_col].agg(["mean", "median"])

    # Add profile features to future using training stats
    future_with_profile = future_df.copy()
    future_with_profile["month_hist_mean"] = future_with_profile.index.month.map(month_stats["mean"]).fillna(0)
    future_with_profile["month_hist_median"] = future_with_profile.index.month.map(month_stats["median"]).fillna(0)

    X_train, y_train = build_direct_feature_matrix(
        train_with_profile, target_col, enso_schema=enso_schema, use_log=use_log
    )
    X_future, _ = build_direct_feature_matrix(
        future_with_profile.assign(**{target_col: 0}),
        target_col,
        enso_schema=enso_schema,
        use_log=use_log,
    )

    # Align columns — use last training value for missing exogenous features
    # (e.g., cpi, gas_price, unemployment_rate) rather than zero
    last_train_values = X_train.iloc[-1]
    for col in X_train.columns:
        if col not in X_future.columns:
            X_future[col] = last_train_values[col]
    for col in X_future.columns:
        if col not in X_train.columns:
            X_train[col] = 0
    X_future = X_future[X_train.columns]

    X_train = clean_features(X_train)
    X_future = clean_features(X_future)

    fitted_model = clone(model)
    fitted_model.fit(X_train.values, y_train.values)
    y_pred = fitted_model.predict(X_future.values)

    if use_log:
        y_pred = np.expm1(y_pred)

    return np.clip(np.asarray(y_pred, dtype=float), 0.0, None), fitted_model, X_train


def fit_predict_prophet(train_df, future_df, target_col, enso_schema):
    if not PROPHET_AVAILABLE:
        raise ImportError("prophet is not installed")

    train_temp = add_time_features(train_df.copy())

    prophet_train = pd.DataFrame({
        "ds": train_temp.index,
        "y": train_temp[target_col].values,
    })

    # Use only a few reliable regressors for Prophet
    safe_regressors = []
    for col in ["covid_flag", "cms_loss_flag", "covid_year_window"]:
        if col in train_temp.columns and train_temp[col].nunique() > 1:
            safe_regressors.append(col)
            prophet_train[col] = train_temp[col].values

    prophet_train = prophet_train.fillna(0)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # multiplicative for proportional seasonality
        changepoint_prior_scale=0.05,
    )
    for c in safe_regressors:
        model.add_regressor(c)
    model.fit(prophet_train)

    # Build future dataframe
    all_dates = train_temp.index.tolist() + future_df.index.tolist()
    prophet_future = pd.DataFrame({"ds": all_dates})
    combined = pd.concat([train_temp, add_time_features(future_df.copy())])
    for c in safe_regressors:
        if c in combined.columns:
            prophet_future[c] = combined[c].reindex(all_dates).fillna(0).values
        else:
            prophet_future[c] = 0
    prophet_future = prophet_future.fillna(0)

    fcst = model.predict(prophet_future)
    pred = (
        fcst.set_index("ds")
        .loc[future_df.index, "yhat"]
        .clip(lower=0)
        .values.astype(float)
    )
    fitted_in_sample = (
        fcst.set_index("ds")
        .loc[train_temp.index, "yhat"]
        .clip(lower=0)
        .values.astype(float)
    )
    return pred, model, safe_regressors, fitted_in_sample


# =========================
# Evaluation
# =========================
def evaluate_direct_sklearn(model, df, target_col, splits, enso_schema, use_log=True):
    """Evaluate ML model using direct forecasting in each CV fold."""
    preds = []
    actuals = []
    fold_rows = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        pred, _, _ = direct_forecast_sklearn(
            model=model,
            train_df=train_df,
            future_df=val_df.drop(columns=[target_col]),
            target_col=target_col,
            enso_schema=enso_schema,
            use_log=use_log,
        )

        y_true = val_df[target_col].values.astype(float)
        preds.extend(pred.tolist())
        actuals.extend(y_true.tolist())
        fold_rows.append({
            "fold": fold_num,
            "mae": mean_absolute_error(y_true, pred),
            "rmse": rmse(y_true, pred),
            "mape": safe_mape(y_true, pred),
            "n_val": len(y_true),
            "train_end": str(train_df.index.max().date()),
            "val_start": str(val_df.index.min().date()),
            "val_end": str(val_df.index.max().date()),
        })

    return {
        "mae": mean_absolute_error(actuals, preds),
        "rmse": rmse(actuals, preds),
        "mape": safe_mape(actuals, preds),
        "n_preds": len(preds),
        "fold_metrics": fold_rows,
    }


def evaluate_seasonal_naive(df, target_col, splits):
    preds = []
    actuals = []
    fold_rows = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_y = df.iloc[train_idx][target_col]
        val_y = df.iloc[val_idx][target_col].values.astype(float)

        pred = seasonal_naive_forecast(train_y, len(val_idx))
        preds.extend(pred.tolist())
        actuals.extend(val_y.tolist())
        fold_rows.append({
            "fold": fold_num,
            "mae": mean_absolute_error(val_y, pred),
            "rmse": rmse(val_y, pred),
            "mape": safe_mape(val_y, pred),
            "n_val": len(val_y),
            "train_end": str(df.iloc[train_idx].index.max().date()),
            "val_start": str(df.iloc[val_idx].index.min().date()),
            "val_end": str(df.iloc[val_idx].index.max().date()),
        })

    return {
        "mae": mean_absolute_error(actuals, preds),
        "rmse": rmse(actuals, preds),
        "mape": safe_mape(actuals, preds),
        "n_preds": len(preds),
        "fold_metrics": fold_rows,
    }


def evaluate_ets_model_with_params(df, target_col, splits, trend, seasonal, damped_trend):
    preds = []
    actuals = []
    fold_rows = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_y = df.iloc[train_idx][target_col].values.astype(float)
        val_y = df.iloc[val_idx][target_col].values.astype(float)

        try:
            pred, _ = fit_predict_ets(
                train_y=train_y,
                forecast_steps=len(val_idx),
                trend=trend,
                seasonal=seasonal,
                damped_trend=damped_trend,
            )
        except Exception:
            pred = np.full(len(val_idx), np.nanmean(train_y))

        pred = np.clip(pred, 0.0, None)
        preds.extend(pred.tolist())
        actuals.extend(val_y.tolist())
        fold_rows.append({
            "fold": fold_num,
            "mae": mean_absolute_error(val_y, pred),
            "rmse": rmse(val_y, pred),
            "mape": safe_mape(val_y, pred),
            "n_val": len(val_y),
            "train_end": str(df.iloc[train_idx].index.max().date()),
            "val_start": str(df.iloc[val_idx].index.min().date()),
            "val_end": str(df.iloc[val_idx].index.max().date()),
        })

    return {
        "mae": mean_absolute_error(actuals, preds),
        "rmse": rmse(actuals, preds),
        "mape": safe_mape(actuals, preds),
        "n_preds": len(preds),
        "fold_metrics": fold_rows,
    }


def evaluate_sarimax(df, target_col, splits, order, seasonal_order):
    preds = []
    actuals = []
    fold_rows = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_y = df.iloc[train_idx][target_col].values.astype(float)
        val_y = df.iloc[val_idx][target_col].values.astype(float)

        try:
            pred, _ = fit_predict_sarimax(
                train_y=train_y,
                forecast_steps=len(val_idx),
                order=order,
                seasonal_order=seasonal_order,
            )
        except Exception:
            pred = np.full(len(val_idx), np.nanmean(train_y))

        pred = np.clip(pred, 0.0, None)
        preds.extend(pred.tolist())
        actuals.extend(val_y.tolist())
        fold_rows.append({
            "fold": fold_num,
            "mae": mean_absolute_error(val_y, pred),
            "rmse": rmse(val_y, pred),
            "mape": safe_mape(val_y, pred),
            "n_val": len(val_y),
            "train_end": str(df.iloc[train_idx].index.max().date()),
            "val_start": str(df.iloc[val_idx].index.min().date()),
            "val_end": str(df.iloc[val_idx].index.max().date()),
        })

    return {
        "mae": mean_absolute_error(actuals, preds),
        "rmse": rmse(actuals, preds),
        "mape": safe_mape(actuals, preds),
        "n_preds": len(preds),
        "fold_metrics": fold_rows,
    }


def evaluate_prophet_model(df, target_col, splits, enso_schema):
    preds = []
    actuals = []
    fold_rows = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        try:
            pred, _, reg_cols, _ = fit_predict_prophet(
                train_df=train_df,
                future_df=val_df.drop(columns=[target_col]),
                target_col=target_col,
                enso_schema=enso_schema,
            )
        except Exception:
            pred = np.full(len(val_idx), train_df[target_col].mean())

        y_true = val_df[target_col].values.astype(float)
        preds.extend(pred.tolist())
        actuals.extend(y_true.tolist())
        fold_rows.append({
            "fold": fold_num,
            "mae": mean_absolute_error(y_true, pred),
            "rmse": rmse(y_true, pred),
            "mape": safe_mape(y_true, pred),
            "n_val": len(y_true),
            "train_end": str(train_df.index.max().date()),
            "val_start": str(val_df.index.min().date()),
            "val_end": str(val_df.index.max().date()),
        })

    return {
        "mae": mean_absolute_error(actuals, preds),
        "rmse": rmse(actuals, preds),
        "mape": safe_mape(actuals, preds),
        "n_preds": len(preds),
        "fold_metrics": fold_rows,
    }


# =========================
# Model search space (streamlined)
# =========================
def build_model_grid():
    grids = []

    # --- Seasonal naive baseline ---
    grids.append({
        "model_family": "seasonal_naive",
        "estimator": None,
        "params": {},
    })

    # --- Ridge (direct, log-transformed target) ---
    for alpha in [0.1, 1.0, 10.0]:
        grids.append({
            "model_family": "ridge",
            "estimator": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha, random_state=42)),
            ]),
            "params": {"alpha": alpha, "use_log": True},
        })

    # --- ElasticNet (direct, log-transformed) ---
    for alpha in [0.01, 0.1, 1.0]:
        for l1_ratio in [0.5, 0.9]:
            grids.append({
                "model_family": "elasticnet",
                "estimator": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)),
                ]),
                "params": {"alpha": alpha, "l1_ratio": l1_ratio, "use_log": True},
            })

    # --- Random Forest (direct, log-transformed) ---
    for n_estimators in [200, 500]:
        for max_depth in [4, 8]:
            for min_samples_leaf in [3, 6]:
                grids.append({
                    "model_family": "random_forest",
                    "estimator": Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=42,
                            n_jobs=-1,
                        )),
                    ]),
                    "params": {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "use_log": True,
                    },
                })

    # --- HistGradientBoosting (direct, log-transformed) ---
    for learning_rate in [0.05, 0.1]:
        for max_depth in [3, 5]:
            for min_samples_leaf in [5, 10]:
                grids.append({
                    "model_family": "hist_gradient_boosting",
                    "estimator": Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", HistGradientBoostingRegressor(
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            max_iter=400,
                            l2_regularization=0.1,
                            random_state=42,
                        )),
                    ]),
                    "params": {
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "use_log": True,
                    },
                })

    # --- XGBoost (if available) ---
    if XGBOOST_AVAILABLE:
        for learning_rate in [0.05, 0.1]:
            for max_depth in [3, 5]:
                for n_estimators in [200, 500]:
                    grids.append({
                        "model_family": "xgboost",
                        "estimator": Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("model", XGBRegressor(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                reg_lambda=1.0,
                                objective="reg:squarederror",
                                random_state=42,
                                n_jobs=4,
                            )),
                        ]),
                        "params": {
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "use_log": True,
                        },
                    })

    # --- CatBoost (if available) ---
    if CATBOOST_AVAILABLE:
        for depth in [4, 6]:
            for learning_rate in [0.05, 0.1]:
                grids.append({
                    "model_family": "catboost",
                    "estimator": Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", CatBoostRegressor(
                            depth=depth,
                            learning_rate=learning_rate,
                            iterations=400,
                            loss_function="RMSE",
                            verbose=0,
                            random_seed=42,
                        )),
                    ]),
                    "params": {
                        "depth": depth,
                        "learning_rate": learning_rate,
                        "use_log": True,
                    },
                })

    # --- ETS variants (including multiplicative) ---
    for trend in ["add", "mul", None]:
        for seasonal in ["add", "mul", None]:
            for damped_trend in [True, False]:
                if trend is None and damped_trend:
                    continue
                if seasonal is None and trend is None and not damped_trend:
                    continue  # skip trivial no-seasonality no-trend
                grids.append({
                    "model_family": "ets",
                    "estimator": None,
                    "params": {
                        "trend": trend,
                        "seasonal": seasonal,
                        "damped_trend": damped_trend,
                    },
                })

    # --- SARIMAX ---
    for order in [(1, 0, 1), (1, 1, 1), (0, 1, 1)]:
        for seasonal_order in [(1, 1, 1, 12), (0, 1, 1, 12), (1, 1, 0, 12)]:
            grids.append({
                "model_family": "sarimax",
                "estimator": None,
                "params": {
                    "order": order,
                    "seasonal_order": seasonal_order,
                },
            })

    # --- Prophet (if available) ---
    if PROPHET_AVAILABLE:
        grids.append({
            "model_family": "prophet",
            "estimator": None,
            "params": {
                "yearly_seasonality": True,
                "seasonality_mode": "multiplicative",
            },
        })

    return grids


# =========================
# Fit best model(s) and forecast
# =========================
def fit_best_and_forecast(course_df, course_name, target_col, forecast_end, output_dir):
    course_dir = output_dir / sanitize_name(course_name)
    course_dir.mkdir(parents=True, exist_ok=True)

    df = course_df.copy().sort_index()
    enso_schema = get_enso_schema(df)

    selection_df = df.copy()
    if REPORTING_LAG_MONTHS > 0 and len(selection_df) > REPORTING_LAG_MONTHS + 24:
        selection_df = selection_df.iloc[:-REPORTING_LAG_MONTHS].copy()

    splits = time_series_cv_splits(
        n_obs=len(selection_df),
        initial_train=max(24, min(36, len(selection_df) // 2)),
        horizon=6,
        step=3,
        min_splits=3,
    )

    if not splits:
        raise ValueError(f"Not enough history to create validation splits for {course_name}")

    model_grid = build_model_grid()
    results = []

    for i, item in enumerate(model_grid, start=1):
        try:
            if item["model_family"] == "seasonal_naive":
                score = evaluate_seasonal_naive(selection_df, target_col, splits)

            elif item["model_family"] == "ets":
                score = evaluate_ets_model_with_params(
                    df=selection_df,
                    target_col=target_col,
                    splits=splits,
                    trend=item["params"]["trend"],
                    seasonal=item["params"]["seasonal"],
                    damped_trend=item["params"]["damped_trend"],
                )

            elif item["model_family"] == "sarimax":
                score = evaluate_sarimax(
                    df=selection_df,
                    target_col=target_col,
                    splits=splits,
                    order=tuple(item["params"]["order"]),
                    seasonal_order=tuple(item["params"]["seasonal_order"]),
                )

            elif item["model_family"] == "prophet":
                score = evaluate_prophet_model(
                    df=selection_df,
                    target_col=target_col,
                    splits=splits,
                    enso_schema=enso_schema,
                )

            else:
                use_log = item["params"].get("use_log", True)
                score = evaluate_direct_sklearn(
                    model=item["estimator"],
                    df=selection_df,
                    target_col=target_col,
                    splits=splits,
                    enso_schema=enso_schema,
                    use_log=use_log,
                )

            results.append({
                "course": course_name,
                "model_family": item["model_family"],
                "mae": score["mae"],
                "rmse": score["rmse"],
                "mape": score["mape"],
                "n_preds": score["n_preds"],
                "params_json": json.dumps(item["params"], default=str),
                "error": None,
            })

            with open(course_dir / f'cv_folds_{item["model_family"]}_{i:03d}.json', "w") as f:
                json.dump(score["fold_metrics"], f, indent=2)

        except Exception as e:
            results.append({
                "course": course_name,
                "model_family": item["model_family"],
                "mae": np.nan,
                "rmse": np.nan,
                "mape": np.nan,
                "n_preds": np.nan,
                "params_json": json.dumps(item["params"], default=str),
                "error": str(e),
            })

    results_df = pd.DataFrame(results).sort_values(["rmse", "mae", "mape"], na_position="last")
    results_df.to_csv(course_dir / "model_search_results.csv", index=False)

    valid_results = results_df.dropna(subset=["rmse"]).copy()
    if valid_results.empty:
        raise ValueError(f"All candidate models failed for {course_name}")

    # --- Generate forecasts from top-K models and ensemble ---
    final_train = df.copy()
    if REPORTING_LAG_MONTHS > 0 and len(final_train) > REPORTING_LAG_MONTHS + 24:
        final_train = final_train.iloc[:-REPORTING_LAG_MONTHS].copy()

    last_obs = df.index.max()
    future_idx = pd.date_range(last_obs + pd.offsets.MonthBegin(1), pd.Timestamp(forecast_end), freq="MS")
    future_df = pd.DataFrame(index=future_idx)

    for col in [
        "covid_flag", "cms_loss_flag", "unemployment_rate", "cpi", "gas_price",
        "economic_pressure_index", "provider_roster_upload_timing_days",
        "provider_student_fees_delay_days", "enso_outlook",
    ]:
        if col in df.columns:
            future_df[col] = df[col].iloc[-1]

    if "enso_outlook" not in future_df.columns:
        future_df["enso_outlook"] = "unknown"

    future_df[COURSE_COL] = course_name
    future_df = add_time_features(future_df)

    if "covid_flag" in future_df.columns:
        future_df["covid_flag"] = 0
    if "covid_year_window" in future_df.columns:
        future_df["covid_year_window"] = 0
    if "cms_loss_flag" in future_df.columns:
        future_df["cms_loss_flag"] = (future_df.index >= pd.Timestamp("2023-07-01")).astype(int)
    future_df["is_recent_reporting_window"] = 0

    # ---- Generate seasonal naive as the anchor forecast ----
    naive_pred = seasonal_naive_forecast(final_train[target_col], len(future_idx))

    # ---- Collect forecasts from top-K models ----
    top_k = min(ENSEMBLE_TOP_K, len(valid_results))
    top_models = valid_results.head(top_k)
    all_model_preds = []
    model_weights = []
    model_families_used = []
    best_family = None
    best_params = None

    # Always include seasonal naive in ensemble with a baseline weight
    naive_rmse_row = valid_results[valid_results["model_family"] == "seasonal_naive"]
    naive_rmse = float(naive_rmse_row["rmse"].iloc[0]) if len(naive_rmse_row) else 500.0
    all_model_preds.append(naive_pred)
    model_weights.append(1.0 / (naive_rmse + 1e-8))
    model_families_used.append("seasonal_naive")

    for rank, (_, row) in enumerate(top_models.iterrows()):
        family = row["model_family"]
        params = json.loads(row["params_json"])
        weight = 1.0 / (row["rmse"] + 1e-8)  # inverse-RMSE weighting

        if rank == 0:
            best_family = family
            best_params = params

        # Skip seasonal_naive since we already included it
        if family == "seasonal_naive":
            continue

        try:
            pred = _generate_forecast(
                family, params, final_train, future_df, future_idx,
                target_col, enso_schema,
            )
            # Sanity check: reject forecasts that are degenerate
            # (peak month should be at least 10% of seasonal naive peak)
            if naive_pred.max() > 0 and pred.max() < 0.1 * naive_pred.max():
                print(f"  Rejecting {family} forecast: peak={pred.max():.1f} "
                      f"vs naive peak={naive_pred.max():.1f} (< 10%)")
                continue

            all_model_preds.append(pred)
            model_weights.append(weight)
            model_families_used.append(family)
        except Exception as e:
            print(f"  Warning: failed to generate forecast for {family} ({params}): {e}")

    # Weighted ensemble
    model_weights = np.array(model_weights)
    model_weights /= model_weights.sum()
    ensemble_pred = np.zeros(len(future_idx))
    for pred, w in zip(all_model_preds, model_weights):
        ensemble_pred += w * pred
    future_pred = np.clip(ensemble_pred, 0.0, None)

    print(f"  Ensemble: {len(all_model_preds)} models, families={model_families_used}")
    print(f"  Ensemble weights: {dict(zip(model_families_used, model_weights.round(3)))}")

    # Also get the single best model prediction for comparison
    single_best_pred = all_model_preds[0] if best_family == "seasonal_naive" else (
        all_model_preds[1] if len(all_model_preds) > 1 else all_model_preds[0]
    )

    # ---- Backtest the best model on last CV fold ----
    last_train_idx, last_val_idx = splits[-1]
    bt_train = selection_df.iloc[last_train_idx].copy()
    bt_val = selection_df.iloc[last_val_idx].copy()

    bt_pred = _generate_forecast(
        best_family, best_params, bt_train,
        bt_val.drop(columns=[target_col], errors="ignore"),
        bt_val.index, target_col, enso_schema,
    )

    # ---- Get fitted in-sample values ----
    fitted_in_sample = _get_fitted_in_sample(
        best_family, best_params, final_train, target_col, enso_schema,
    )

    # ---- Save outputs ----
    plot_backtest = pd.DataFrame({
        "date": bt_val.index,
        "actual": bt_val[target_col].values,
        "predicted": bt_pred,
    })

    forecast_df = pd.DataFrame({
        "date": future_idx,
        "combined_course": course_name,
        "forecast_num_students": future_pred,
        "forecast_single_best": single_best_pred,
        "best_model_family": best_family,
        "best_model_params": json.dumps(best_params),
        "ensemble_n_models": len(all_model_preds),
    })
    forecast_df.to_csv(course_dir / "future_forecast.csv", index=False)
    plot_backtest.to_csv(course_dir / "backtest_actual_vs_pred.csv", index=False)

    fitted_df = pd.DataFrame({
        "date": final_train.index,
        "actual": final_train[target_col].values.astype(float),
        "fitted_or_train_series": fitted_in_sample,
    })
    fitted_df.to_csv(course_dir / "train_fitted_values.csv", index=False)

    best = valid_results.iloc[0].to_dict()
    summary = {
        "course": course_name,
        "best_model_family": best_family,
        "best_model_params": best_params,
        "best_cv_rmse": float(best["rmse"]),
        "best_cv_mae": float(best["mae"]),
        "best_cv_mape": float(best["mape"]),
        "ensemble_n_models": len(all_model_preds),
        "ensemble_families": model_families_used,
        "ensemble_weights": dict(zip(model_families_used, model_weights.round(3).tolist())),
        "train_start": str(final_train.index.min().date()),
        "train_end": str(final_train.index.max().date()),
        "forecast_start": str(future_idx.min().date()) if len(future_idx) else None,
        "forecast_end": str(future_idx.max().date()) if len(future_idx) else None,
        "enso_schema": enso_schema,
        "notes": [
            "Direct forecasting (no recursive lag degradation).",
            "Log1p-transformed target for ML models to handle extreme skew.",
            "Multiplicative ETS and SARIMAX added for proportional seasonality.",
            "Seasonal naive baseline included.",
            "Top-K model ensemble weighted by inverse CV RMSE.",
            "Monthly historical profile features (month_hist_mean/median) replace lag features for future.",
        ],
        "package_availability": {
            "prophet": PROPHET_AVAILABLE,
            "xgboost": XGBOOST_AVAILABLE,
            "catboost": CATBOOST_AVAILABLE,
        },
    }
    with open(course_dir / "best_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Plots ----
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df[target_col].values, label="Actual full history", color="black")
    plt.plot(final_train.index, fitted_in_sample, label="Fitted (best model)", alpha=0.7)
    plt.axvline(final_train.index.max(), linestyle="--", alpha=0.7, color="gray", label="Forecast start")
    if len(future_idx):
        plt.plot(future_idx, future_pred, label=f"Ensemble forecast (n={len(all_model_preds)})", linewidth=2)
        plt.plot(future_idx, single_best_pred, label=f"Best model ({best_family})", linestyle="--", alpha=0.6)
    plt.title(f"{course_name}: history and forecast")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(course_dir / "history_and_forecast.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(plot_backtest["date"], plot_backtest["actual"], marker="o", label="Actual")
    plt.plot(plot_backtest["date"], plot_backtest["predicted"], marker="o", label="Predicted")
    plt.title(f"{course_name}: last backtest window actual vs forecast")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(course_dir / "backtest_actual_vs_forecast.png", dpi=150)
    plt.close()

    return {
        "course": course_name,
        "best_model_family": best_family,
        "best_model_params": best_params,
        "best_cv_rmse": float(best["rmse"]),
        "best_cv_mae": float(best["mae"]),
        "best_cv_mape": float(best["mape"]),
        "ensemble_n_models": len(all_model_preds),
    }


def _generate_forecast(family, params, train_df, future_df, future_idx,
                        target_col, enso_schema):
    """Generate a forecast array for a given model family and params."""
    if family == "seasonal_naive":
        return seasonal_naive_forecast(train_df[target_col], len(future_idx))

    elif family == "ets":
        pred, _ = fit_predict_ets(
            train_y=train_df[target_col].values,
            forecast_steps=len(future_idx),
            trend=params["trend"],
            seasonal=params["seasonal"],
            damped_trend=params["damped_trend"],
        )
        return np.clip(pred, 0.0, None)

    elif family == "sarimax":
        pred, _ = fit_predict_sarimax(
            train_y=train_df[target_col].values,
            forecast_steps=len(future_idx),
            order=tuple(params["order"]),
            seasonal_order=tuple(params["seasonal_order"]),
        )
        return np.clip(pred, 0.0, None)

    elif family == "prophet":
        pred, _, _, _ = fit_predict_prophet(
            train_df=train_df,
            future_df=future_df,
            target_col=target_col,
            enso_schema=enso_schema,
        )
        return pred

    else:
        # ML model — reconstruct estimator
        use_log = params.get("use_log", True)
        estimator = _rebuild_estimator(family, params)
        pred, _, _ = direct_forecast_sklearn(
            model=estimator,
            train_df=train_df,
            future_df=future_df.drop(columns=[target_col], errors="ignore"),
            target_col=target_col,
            enso_schema=enso_schema,
            use_log=use_log,
        )
        return pred


def _rebuild_estimator(family, params):
    """Reconstruct a sklearn Pipeline from family name and params."""
    if family == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=params["alpha"], random_state=42)),
        ])
    elif family == "elasticnet":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                max_iter=10000, random_state=42,
            )),
        ])
    elif family == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=42, n_jobs=-1,
            )),
        ])
    elif family == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_iter=400, l2_regularization=0.1, random_state=42,
            )),
        ])
    elif family == "xgboost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective="reg:squarederror", random_state=42, n_jobs=4,
            )),
        ])
    elif family == "catboost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostRegressor(
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                iterations=400, loss_function="RMSE", verbose=0, random_seed=42,
            )),
        ])
    else:
        raise ValueError(f"Unknown model family: {family}")


def _get_fitted_in_sample(family, params, train_df, target_col, enso_schema):
    """Get in-sample fitted values for plotting."""
    n = len(train_df)
    try:
        if family == "seasonal_naive":
            fitted = []
            y = train_df[target_col].values
            for i in range(n):
                if i >= 12:
                    fitted.append(max(0.0, float(y[i - 12])))
                else:
                    fitted.append(float(y[i]))
            return np.array(fitted)

        elif family == "ets":
            model = ExponentialSmoothing(
                train_df[target_col].astype(float).clip(lower=0.1),
                trend=params["trend"],
                seasonal=params["seasonal"],
                seasonal_periods=12 if params["seasonal"] else None,
                damped_trend=params["damped_trend"],
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=True)
            return np.clip(model.fittedvalues.values.astype(float), 0.0, None)

        elif family == "sarimax":
            model = SARIMAX(
                train_df[target_col].astype(float),
                order=tuple(params["order"]),
                seasonal_order=tuple(params["seasonal_order"]),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
            return np.clip(model.fittedvalues.values.astype(float), 0.0, None)

        elif family == "prophet":
            _, _, _, fitted = fit_predict_prophet(
                train_df=train_df,
                future_df=pd.DataFrame(index=pd.DatetimeIndex([])),  # no future needed
                target_col=target_col,
                enso_schema=enso_schema,
            )
            return fitted

        else:
            use_log = params.get("use_log", True)
            estimator = _rebuild_estimator(family, params)
            train_with_profile = add_monthly_profile_features(train_df.copy(), target_col)
            X, y = build_direct_feature_matrix(
                train_with_profile, target_col, enso_schema=enso_schema, use_log=use_log,
            )
            fitted_model = clone(estimator)
            fitted_model.fit(X.values, y.values)
            y_hat = fitted_model.predict(X.values)
            if use_log:
                y_hat = np.expm1(y_hat)
            return np.clip(y_hat, 0.0, None)

    except Exception:
        return train_df[target_col].values.astype(float)


# =========================
# Main
# =========================
def main():
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir.parent / "data/cleaned_data" / INPUT_CSV
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL])

    all_course_summaries = []
    all_forecasts = []

    for course_name in sorted(raw[COURSE_COL].dropna().unique()):
        print(f"\n{'='*60}")
        print(f"Running course: {course_name}")
        print(f"{'='*60}")
        course_df = prepare_monthly_series(raw, course_name, TARGET_COL)
        summary = fit_best_and_forecast(
            course_df=course_df,
            course_name=course_name,
            target_col=TARGET_COL,
            forecast_end=FORECAST_END,
            output_dir=output_dir,
        )
        all_course_summaries.append(summary)
        print(f"  Best: {summary['best_model_family']} "
              f"(RMSE={summary['best_cv_rmse']:.1f}, "
              f"ensemble={summary['ensemble_n_models']} models)")

        course_folder = output_dir / sanitize_name(course_name)
        fc = pd.read_csv(course_folder / "future_forecast.csv")
        all_forecasts.append(fc)

    leaderboard = pd.DataFrame(all_course_summaries).sort_values(["best_cv_rmse", "best_cv_mae"])
    leaderboard.to_csv(output_dir / "course_model_leaderboard.csv", index=False)

    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
    combined_forecasts.to_csv(output_dir / "all_courses_future_forecasts.csv", index=False)

    plt.figure(figsize=(14, 7))
    for course_name in combined_forecasts["combined_course"].unique():
        temp = combined_forecasts[combined_forecasts["combined_course"] == course_name]
        plt.plot(pd.to_datetime(temp["date"]), temp["forecast_num_students"], label=course_name)
    plt.title("Forecasts through Aug 2027 by course (ensemble)")
    plt.xlabel("Date")
    plt.ylabel(TARGET_COL)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "combined_course_forecasts.png", dpi=150)
    plt.close()

    availability = {
        "prophet": PROPHET_AVAILABLE,
        "xgboost": XGBOOST_AVAILABLE,
        "catboost": CATBOOST_AVAILABLE,
    }
    with open(output_dir / "package_availability.json", "w") as f:
        json.dump(availability, f, indent=2)

    print(f"\nDone. Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
