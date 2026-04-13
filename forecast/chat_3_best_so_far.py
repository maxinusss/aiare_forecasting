
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
FORECAST_END = "2027-08-01"   # monthly forecast through Aug 2027
OUTPUT_DIR = "chat_3_analysis_updated_4_13"
REPORTING_LAG_MONTHS = 2       # last 2 months can be under-reported


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
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")

def clean_features(X):
    X = X.copy()
    
    # Replace inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Clip extreme values (prevents overflow in tree models)
    X = X.clip(lower=-1e6, upper=1e6)
    
    return X
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

    # Reliability / structural-change notes from user
    out["post_2017_18_reliable"] = (out.index >= pd.Timestamp("2017-07-01")).astype(int)

    # Approximate split flag based on user's note
    out["post_pro_rec_split"] = (out.index >= pd.Timestamp("2018-07-01")).astype(int)

    # New roster timing requirement from 20/21 season onward
    out["post_roster_timing_change"] = (out.index >= pd.Timestamp("2020-07-01")).astype(int)

    # COVID era explicit handling
    out["covid_year_window"] = (
        (out.index >= pd.Timestamp("2020-03-01")) &
        (out.index <= pd.Timestamp("2022-06-01"))
    ).astype(int)

    # Season labels
    out["season_name"] = [season_from_month(m) for m in out.index.month]
    out["season_year"] = np.where(out.index.month >= 7, out.index.year + 1, out.index.year)
    return out


def add_lag_features(df, target_col):
    out = df.copy()

    for lag in [1, 2, 3, 6, 12]:
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    for window in [3, 6, 12]:
        out[f"rolling_mean_{window}"] = out[target_col].shift(1).rolling(window).mean()
        out[f"rolling_std_{window}"] = out[target_col].shift(1).rolling(window).std()
        out[f"rolling_max_{window}"] = out[target_col].shift(1).rolling(window).max()
        out[f"rolling_min_{window}"] = out[target_col].shift(1).rolling(window).min()

    out["yoy_change"] = out[target_col].shift(1) - out[target_col].shift(13)
    denom = out[target_col].shift(13).replace(0, np.nan)
    out["yoy_ratio"] = out[target_col].shift(1) / denom
    out["lag1_vs_roll3"] = out[target_col].shift(1) - out[target_col].shift(1).rolling(3).mean()
    return out


def prepare_monthly_series(raw, course_name, target_col):
    data = raw[raw[COURSE_COL] == course_name].copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    data = data.sort_values(DATE_COL)

    full_idx = pd.date_range(data[DATE_COL].min(), data[DATE_COL].max(), freq="MS")
    monthly = (
        data.set_index(DATE_COL)
            .sort_index()
            .reindex(full_idx)
    )
    monthly.index.name = DATE_COL

    monthly[COURSE_COL] = course_name
    monthly[target_col] = monthly[target_col].fillna(0)

    # fill known external/context columns if present
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

    # categorical context
    if "enso_outlook" in monthly.columns:
        if monthly["enso_outlook"].dtype == "O":
            monthly["enso_outlook"] = monthly["enso_outlook"].ffill().bfill().fillna("unknown")
        else:
            monthly["enso_outlook"] = monthly["enso_outlook"].ffill().bfill()

    monthly = add_time_features(monthly)
    monthly = add_lag_features(monthly, target_col)

    max_dt = monthly.index.max()
    monthly["is_recent_reporting_window"] = (
        monthly.index >= (max_dt - pd.DateOffset(months=REPORTING_LAG_MONTHS - 1))
    ).astype(int)

    # If dataset does not already have cms_loss_flag, create one based on note
    if "cms_loss_flag" not in monthly.columns:
        monthly["cms_loss_flag"] = (monthly.index >= pd.Timestamp("2023-07-01")).astype(int)

    # If dataset does not already have covid_flag, create one
    if "covid_flag" not in monthly.columns:
        monthly["covid_flag"] = monthly["covid_year_window"]

    return monthly


def build_feature_matrix(df, target_col):
    features = df.copy()

    if "enso_outlook" in features.columns and features["enso_outlook"].dtype == "O":
        dummies = pd.get_dummies(features["enso_outlook"], prefix="enso", dummy_na=True)
        features = pd.concat([features.drop(columns=["enso_outlook"]), dummies], axis=1)

    if "season_name" in features.columns:
        dummies = pd.get_dummies(features["season_name"], prefix="season", dummy_na=False)
        features = pd.concat([features.drop(columns=["season_name"]), dummies], axis=1)

    drop_cols = [COURSE_COL, target_col]
    X = features.drop(columns=[c for c in drop_cols if c in features.columns])
    y = features[target_col].copy()

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)

        X = X.select_dtypes(include=[np.number, "bool"]).copy()
        X = clean_features(X)
    return X, y


def get_prophet_regressor_cols(df, target_col):
    temp = df.copy()
    if "season_name" in temp.columns:
        temp = temp.drop(columns=["season_name"])
    if "enso_outlook" in temp.columns and temp["enso_outlook"].dtype == "O":
        temp = pd.concat([temp.drop(columns=["enso_outlook"]),
                          pd.get_dummies(temp["enso_outlook"], prefix="enso", dummy_na=True)], axis=1)
    temp = temp.drop(columns=[c for c in [COURSE_COL, target_col] if c in temp.columns], errors="ignore")
    temp = temp.select_dtypes(include=[np.number, "bool"]).replace([np.inf, -np.inf], np.nan)
    good_cols = []
    for c in temp.columns:
        # Prophet regressors should not be all missing or constant
        if temp[c].notna().sum() > 6 and temp[c].nunique(dropna=True) > 1:
            good_cols.append(c)
    return good_cols


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
def fit_predict_ets(train_y, forecast_steps, trend, seasonal, damped_trend):
    model = ExponentialSmoothing(
        train_y.astype(float),
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=12 if seasonal else None,
        damped_trend=damped_trend,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True, use_brute=True)
    pred = fitted.forecast(forecast_steps)
    return np.asarray(pred, dtype=float), fitted


def recursive_forecast_sklearn(model, history_df, future_df, target_col):
    all_df = pd.concat([history_df.copy(), future_df.copy()], axis=0).sort_index()

    if target_col not in all_df.columns:
        all_df[target_col] = np.nan

    preds = []
    future_idx = future_df.index

    for current_dt in future_idx:
        temp = add_lag_features(all_df.loc[:current_dt].copy(), target_col)
        temp = add_time_features(temp)
        X_all, _ = build_feature_matrix(temp, target_col)

        train_mask = temp[target_col].notna() & (temp.index < current_dt)
        X_train = X_all.loc[train_mask]
        y_train = temp.loc[train_mask, target_col]

        X_pred = X_all.loc[[current_dt]]

        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        yhat = float(fitted_model.predict(X_pred)[0])
        yhat = max(0.0, yhat)
        all_df.loc[current_dt, target_col] = yhat
        preds.append(yhat)

    return np.asarray(preds), all_df


def fit_predict_prophet(train_df, future_df, target_col):
    if not PROPHET_AVAILABLE:
        raise ImportError("prophet is not installed")

    train_temp = train_df.copy()
    train_temp = add_lag_features(train_temp, target_col)
    train_temp = add_time_features(train_temp)

    future_temp = pd.concat([train_df.copy(), future_df.copy()], axis=0).sort_index()
    future_temp = add_lag_features(future_temp, target_col)
    future_temp = add_time_features(future_temp)

    reg_cols = get_prophet_regressor_cols(train_temp, target_col)

    prophet_train = pd.DataFrame({
        "ds": train_temp.index,
        "y": train_temp[target_col].values
    })

    for c in reg_cols:
        prophet_train[c] = train_temp[c].values

    prophet_train = prophet_train.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").fillna(0)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )
    for c in reg_cols:
        model.add_regressor(c)

    model.fit(prophet_train)

    prophet_future = pd.DataFrame({"ds": future_temp.index})
    for c in reg_cols:
        prophet_future[c] = future_temp[c].values

    prophet_future = prophet_future.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").fillna(0)
    fcst = model.predict(prophet_future)

    pred = fcst.set_index("ds").loc[future_df.index, "yhat"].clip(lower=0).values.astype(float)
    return pred, model, reg_cols


# =========================
# Evaluation
# =========================
def evaluate_sklearn_model(model, df, target_col, splits):
    preds = []
    actuals = []
    fold_rows = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        pred, _ = recursive_forecast_sklearn(
            model=model,
            history_df=train_df,
            future_df=val_df.drop(columns=[target_col]),
            target_col=target_col,
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

    score = {
        "mae": mean_absolute_error(actuals, preds),
        "rmse": rmse(actuals, preds),
        "mape": safe_mape(actuals, preds),
        "n_preds": len(preds),
        "fold_metrics": fold_rows,
    }
    return score


def evaluate_ets_model(df, target_col, splits, trend, seasonal, damped_trend):
    preds = []
    actuals = []
    fold_rows = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_y = df.iloc[train_idx][target_col].values.astype(float)
        val_y = df.iloc[val_idx][target_col].values.astype(float)

        pred, _ = fit_predict_ets(
            train_y=train_y,
            forecast_steps=len(val_idx),
            trend=trend,
            seasonal=seasonal,
            damped_trend=damped_trend,
        )
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

    score = {
        "mae": mean_absolute_error(actuals, preds),
        "rmse": rmse(actuals, preds),
        "mape": safe_mape(actuals, preds),
        "n_preds": len(preds),
        "fold_metrics": fold_rows,
    }
    return score


def evaluate_prophet_model(df, target_col, splits):
    preds = []
    actuals = []
    fold_rows = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        pred, _, reg_cols = fit_predict_prophet(
            train_df=train_df,
            future_df=val_df.drop(columns=[target_col]),
            target_col=target_col,
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
            "n_regressors": len(reg_cols),
        })

    score = {
        "mae": mean_absolute_error(actuals, preds),
        "rmse": rmse(actuals, preds),
        "mape": safe_mape(actuals, preds),
        "n_preds": len(preds),
        "fold_metrics": fold_rows,
    }
    return score


# =========================
# Model search space
# =========================
def build_model_grid():
    grids = []

    # Linear models
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        grids.append({
            "model_family": "ridge",
            "estimator": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha, random_state=42)),
            ]),
            "params": {"alpha": alpha},
        })

    for alpha in [0.001, 0.01, 0.1, 1.0]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            grids.append({
                "model_family": "elasticnet",
                "estimator": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)),
                ]),
                "params": {"alpha": alpha, "l1_ratio": l1_ratio},
            })

    # Tree models
    for n_estimators in [200, 500]:
        for max_depth in [4, 8, None]:
            for min_samples_leaf in [1, 3, 6]:
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
                    },
                })

    for learning_rate in [0.03, 0.05, 0.1]:
        for max_depth in [3, 5, 8]:
            for min_samples_leaf in [5, 10, 20]:
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
                    },
                })

    # XGBoost
    if XGBOOST_AVAILABLE:
        for learning_rate in [0.03, 0.05, 0.1]:
            for max_depth in [3, 5, 8]:
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
                        },
                    })

    # CatBoost
    if CATBOOST_AVAILABLE:
        for depth in [4, 6, 8]:
            for learning_rate in [0.03, 0.05, 0.1]:
                for iterations in [300, 600]:
                    grids.append({
                        "model_family": "catboost",
                        "estimator": Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("model", CatBoostRegressor(
                                depth=depth,
                                learning_rate=learning_rate,
                                iterations=iterations,
                                loss_function="RMSE",
                                verbose=0,
                                random_seed=42,
                            )),
                        ]),
                        "params": {
                            "depth": depth,
                            "learning_rate": learning_rate,
                            "iterations": iterations,
                        },
                    })

    # ETS
    for trend in ["add", None]:
        for seasonal in ["add", None]:
            for damped_trend in [True, False]:
                if trend is None and damped_trend:
                    continue
                grids.append({
                    "model_family": "ets",
                    "estimator": None,
                    "params": {
                        "trend": trend,
                        "seasonal": seasonal,
                        "damped_trend": damped_trend,
                    },
                })

    # Prophet
    if PROPHET_AVAILABLE:
        grids.append({
            "model_family": "prophet",
            "estimator": None,
            "params": {
                "yearly_seasonality": True,
                "seasonality_mode": "additive",
            },
        })

    return grids


# =========================
# Fit best model and forecast
# =========================
def fit_best_and_forecast(course_df, course_name, target_col, forecast_end, output_dir):
    course_dir = output_dir / sanitize_name(course_name)
    course_dir.mkdir(parents=True, exist_ok=True)

    df = course_df.copy().sort_index()

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
            if item["model_family"] == "ets":
                score = evaluate_ets_model(
                    df=selection_df,
                    target_col=target_col,
                    splits=splits,
                    trend=item["params"]["trend"],
                    seasonal=item["params"]["seasonal"],
                    damped_trend=item["params"]["damped_trend"],
                )
            elif item["model_family"] == "prophet":
                score = evaluate_prophet_model(
                    df=selection_df,
                    target_col=target_col,
                    splits=splits,
                )
            else:
                score = evaluate_sklearn_model(
                    model=item["estimator"],
                    df=selection_df,
                    target_col=target_col,
                    splits=splits,
                )

            row = {
                "course": course_name,
                "model_family": item["model_family"],
                "mae": score["mae"],
                "rmse": score["rmse"],
                "mape": score["mape"],
                "n_preds": score["n_preds"],
                "params_json": json.dumps(item["params"], default=str),
                "error": None,
            }
            results.append(row)

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

    best = valid_results.iloc[0].to_dict()
    best_family = best["model_family"]
    best_params = json.loads(best["params_json"])

    final_train = df.copy()
    if REPORTING_LAG_MONTHS > 0 and len(final_train) > REPORTING_LAG_MONTHS + 24:
        final_train = final_train.iloc[:-REPORTING_LAG_MONTHS].copy()

    last_obs = df.index.max()
    future_idx = pd.date_range(last_obs + pd.offsets.MonthBegin(1), pd.Timestamp(forecast_end), freq="MS")
    future_df = pd.DataFrame(index=future_idx)

    for col in [
        "covid_flag", "cms_loss_flag", "unemployment_rate", "cpi", "gas_price",
        "economic_pressure_index", "provider_roster_upload_timing_days",
        "provider_student_fees_delay_days", "enso_outlook"
    ]:
        if col in df.columns:
            future_df[col] = df[col].iloc[-1]

    future_df[COURSE_COL] = course_name
    future_df = add_time_features(future_df)

    if "covid_flag" in future_df.columns:
        future_df["covid_flag"] = 0
    if "covid_year_window" in future_df.columns:
        future_df["covid_year_window"] = 0
    if "cms_loss_flag" in future_df.columns:
        future_df["cms_loss_flag"] = (future_df.index >= pd.Timestamp("2023-07-01")).astype(int)
    future_df["is_recent_reporting_window"] = 0

    last_train_idx, last_val_idx = splits[-1]
    bt_train = selection_df.iloc[last_train_idx].copy()
    bt_val = selection_df.iloc[last_val_idx].copy()

    if best_family == "ets":
        bt_pred, _ = fit_predict_ets(
            train_y=bt_train[target_col].values,
            forecast_steps=len(bt_val),
            trend=best_params["trend"],
            seasonal=best_params["seasonal"],
            damped_trend=best_params["damped_trend"],
        )
        bt_pred = np.clip(bt_pred, 0.0, None)

        full_fitted_model = ExponentialSmoothing(
            final_train[target_col].astype(float),
            trend=best_params["trend"],
            seasonal=best_params["seasonal"],
            seasonal_periods=12 if best_params["seasonal"] else None,
            damped_trend=best_params["damped_trend"],
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)
        future_pred = np.clip(full_fitted_model.forecast(len(future_idx)).values.astype(float), 0.0, None)
        fitted_in_sample = np.clip(full_fitted_model.fittedvalues.values.astype(float), 0.0, None)

    elif best_family == "prophet":
        bt_pred, bt_model, reg_cols = fit_predict_prophet(
            train_df=bt_train,
            future_df=bt_val.drop(columns=[target_col]),
            target_col=target_col,
        )
        future_pred, full_model, full_reg_cols = fit_predict_prophet(
            train_df=final_train,
            future_df=future_df,
            target_col=target_col,
        )

        train_prophet_df = final_train.copy()
        train_prophet_df = add_lag_features(train_prophet_df, target_col)
        train_prophet_df = add_time_features(train_prophet_df)
        prophet_train = pd.DataFrame({"ds": train_prophet_df.index})
        for c in full_reg_cols:
            prophet_train[c] = train_prophet_df[c].values
        prophet_train = prophet_train.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").fillna(0)
        fitted_full = full_model.predict(prophet_train)
        fitted_in_sample = fitted_full.set_index("ds").loc[final_train.index, "yhat"].clip(lower=0).values.astype(float)

    else:
        if best_family == "ridge":
            best_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=best_params["alpha"], random_state=42)),
            ])
        elif best_family == "elasticnet":
            best_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(
                    alpha=best_params["alpha"],
                    l1_ratio=best_params["l1_ratio"],
                    max_iter=10000,
                    random_state=42,
                )),
            ])
        elif best_family == "random_forest":
            best_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=best_params["n_estimators"],
                    max_depth=best_params["max_depth"],
                    min_samples_leaf=best_params["min_samples_leaf"],
                    random_state=42,
                    n_jobs=-1,
                )),
            ])
        elif best_family == "hist_gradient_boosting":
            best_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(
                    learning_rate=best_params["learning_rate"],
                    max_depth=best_params["max_depth"],
                    min_samples_leaf=best_params["min_samples_leaf"],
                    max_iter=400,
                    l2_regularization=0.1,
                    random_state=42,
                )),
            ])
        elif best_family == "xgboost":
            best_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(
                    n_estimators=best_params["n_estimators"],
                    learning_rate=best_params["learning_rate"],
                    max_depth=best_params["max_depth"],
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=4,
                )),
            ])
        elif best_family == "catboost":
            best_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", CatBoostRegressor(
                    depth=best_params["depth"],
                    learning_rate=best_params["learning_rate"],
                    iterations=best_params["iterations"],
                    loss_function="RMSE",
                    verbose=0,
                    random_seed=42,
                )),
            ])
        else:
            raise ValueError(f"Unsupported best_family: {best_family}")

        bt_pred, _ = recursive_forecast_sklearn(
            model=best_model,
            history_df=bt_train,
            future_df=bt_val.drop(columns=[target_col]),
            target_col=target_col,
        )

        full_future_pred, full_all = recursive_forecast_sklearn(
            model=best_model,
            history_df=final_train,
            future_df=future_df.drop(columns=[target_col], errors="ignore"),
            target_col=target_col,
        )
        future_pred = np.asarray(full_future_pred)
        fitted_in_sample = full_all.loc[final_train.index, target_col].values.astype(float)

    plot_backtest = pd.DataFrame({
        "date": bt_val.index,
        "actual": bt_val[target_col].values,
        "predicted": bt_pred,
    })

    forecast_df = pd.DataFrame({
        "date": future_idx,
        "combined_course": course_name,
        "forecast_num_students": future_pred,
        "best_model_family": best_family,
        "best_model_params": json.dumps(best_params),
    })
    forecast_df.to_csv(course_dir / "future_forecast.csv", index=False)

    plot_backtest.to_csv(course_dir / "backtest_actual_vs_pred.csv", index=False)

    fitted_df = pd.DataFrame({
        "date": final_train.index,
        "actual": final_train[target_col].values.astype(float),
        "fitted_or_train_series": fitted_in_sample,
    })
    fitted_df.to_csv(course_dir / "train_fitted_values.csv", index=False)

    summary = {
        "course": course_name,
        "best_model_family": best_family,
        "best_model_params": best_params,
        "best_cv_rmse": float(best["rmse"]),
        "best_cv_mae": float(best["mae"]),
        "best_cv_mape": float(best["mape"]),
        "train_start": str(final_train.index.min().date()),
        "train_end": str(final_train.index.max().date()),
        "forecast_start": str(future_idx.min().date()) if len(future_idx) else None,
        "forecast_end": str(future_idx.max().date()) if len(future_idx) else None,
        "notes_used": [
            "COVID spike handled via explicit flags and flexible models.",
            "Post-2017/18 history treated as the reliable era via feature flag.",
            "Post-2020/21 provider roster timing change represented via feature flag.",
            "Last few months excluded from model selection to reduce under-reporting bias.",
            "CMS loss after 2023/24 represented via cms_loss_flag.",
            "Per-course model search across classical TS, linear, tree, boosting, Prophet, XGBoost, and CatBoost.",
        ],
        "package_availability": {
            "prophet": PROPHET_AVAILABLE,
            "xgboost": XGBOOST_AVAILABLE,
            "catboost": CATBOOST_AVAILABLE,
        }
    }
    with open(course_dir / "best_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # plots
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df[target_col].values, label="Actual full history")
    plt.plot(final_train.index, fitted_df["fitted_or_train_series"].values, label="Fitted/train series")
    plt.axvline(final_train.index.max(), linestyle="--", alpha=0.7, label="Forecast start")
    if len(future_idx):
        plt.plot(future_idx, future_pred, label="Forecast")
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
    }


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
        print(f"Running course: {course_name}")
        course_df = prepare_monthly_series(raw, course_name, TARGET_COL)
        summary = fit_best_and_forecast(
            course_df=course_df,
            course_name=course_name,
            target_col=TARGET_COL,
            forecast_end=FORECAST_END,
            output_dir=output_dir,
        )
        all_course_summaries.append(summary)

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
    plt.title("Forecasts through Aug 2027 by course")
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

    print(f"Done. Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
