import logging
from pathlib import Path
from itertools import product
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet, Lasso, HuberRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
)
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

from pandas.tseries.holiday import USFederalHolidayCalendar
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Set working directory to project root (parent of eda folder)
os.chdir(Path(__file__).parent.parent)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DATA_PATH = Path("data/cleaned_data/master_data_full.csv")
DATE_COL = "date"
COURSE_COL = "combined_course"
TARGET_COL = "num_students"   # change to "enrolled" if desired

FORECAST_HORIZON_MONTHS = 12
FORECAST_END_DATE = pd.Timestamp("2027-07-31")
HYPERPARAMETER_TUNING = True

LAGS = [1, 2, 3, 12]
ROLL_WINDOWS = [3, 6, 12]

MIN_TRAIN_MONTHS = 24
VALID_MONTHS = 3

RUN_DIAGNOSTICS = True
DIAGNOSTICS_LAGS = 24
CONFIDENCE_LEVELS = [0.80, 0.95]

MODEL_SPECS = {
    "ridge": Ridge(),
    "lasso": Lasso(max_iter=10000),
    "elasticnet": ElasticNet(max_iter=10000, random_state=42),
    "huber": HuberRegressor(max_iter=300),
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "extra_trees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
    "gbr": GradientBoostingRegressor(random_state=42),
    "hgb": HistGradientBoostingRegressor(random_state=42),
    "bagging": BaggingRegressor(random_state=42, n_jobs=-1),
    "svr": SVR(kernel="rbf"),
}

MODEL_PARAM_GRID = {
    "ridge": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
    },
    "lasso": {
        "model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    },
    "elasticnet": {
        "model__alpha": [0.01, 0.05, 0.1, 0.5],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "huber": {
        "model__epsilon": [1.1, 1.35, 1.5, 2.0],
        "model__alpha": [0.0001, 0.001, 0.01],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": [1, 3, 5],
    },
    "extra_trees": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": [1, 3, 5],
    },
    "gbr": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 4, 5, 6],
        "model__subsample": [0.8, 1.0],
    },
    "hgb": {
        "model__max_iter": [100, 200, 400],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 4, 6],
        "model__min_samples_leaf": [5, 10, 20],
    },
    "bagging": {
        "model__n_estimators": [10, 25, 50],
        "model__max_samples": [0.7, 0.9, 1.0],
        "model__max_features": [0.7, 0.9, 1.0],
    },
    "svr": {
        "model__C": [0.1, 1.0, 10.0, 50.0],
        "model__epsilon": [0.01, 0.1, 0.5, 1.0],
        "model__gamma": ["scale", "auto"],
    },
}


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred, epsilon=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), epsilon)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


# -----------------------------------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values(DATE_COL).reset_index(drop=True)

    out["year_num"] = out[DATE_COL].dt.year
    out["month_num"] = out[DATE_COL].dt.month
    out["quarter"] = out[DATE_COL].dt.quarter

    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12)
    out["quarter_sin"] = np.sin(2 * np.pi * out["quarter"] / 4)
    out["quarter_cos"] = np.cos(2 * np.pi * out["quarter"] / 4)

    out["is_winter"] = out["month_num"].isin([12, 1, 2]).astype(int)
    out["is_spring"] = out["month_num"].isin([3, 4, 5]).astype(int)
    out["is_summer"] = out["month_num"].isin([6, 7, 8]).astype(int)
    out["is_fall"] = out["month_num"].isin([9, 10, 11]).astype(int)

    logging.info(
        "Added time features | rows=%s | min_date=%s | max_date=%s",
        len(out),
        out[DATE_COL].min(),
        out[DATE_COL].max(),
    )
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values(DATE_COL).reset_index(drop=True)

    out["is_month_start"] = out[DATE_COL].dt.is_month_start.astype(int)
    out["is_month_end"] = out[DATE_COL].dt.is_month_end.astype(int)
    out["days_in_month"] = out[DATE_COL].dt.days_in_month

    month_start = out[DATE_COL].dt.to_period("M").dt.to_timestamp()
    month_end = month_start + pd.offsets.MonthEnd(0)
    out["month_length_days"] = (month_end - month_start).dt.days + 1

    cal = USFederalHolidayCalendar()
    holiday_dates = cal.holidays(
        start=out[DATE_COL].min() - pd.Timedelta(days=31),
        end=out[DATE_COL].max() + pd.Timedelta(days=31),
    )
    holiday_df = pd.DataFrame({"holiday_date": pd.to_datetime(holiday_dates)})
    holiday_df["month_start"] = holiday_df["holiday_date"].dt.to_period("M").dt.to_timestamp()
    monthly_holiday_counts = (
        holiday_df.groupby("month_start").size().rename("holiday_count_in_month").reset_index()
    )

    out["month_start"] = out[DATE_COL].dt.to_period("M").dt.to_timestamp()
    out = out.merge(monthly_holiday_counts, on="month_start", how="left")
    out["holiday_count_in_month"] = out["holiday_count_in_month"].fillna(0)
    out = out.drop(columns=["month_start"])

    return out


def add_trend_and_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(DATE_COL).reset_index(drop=True)

    out["time_idx"] = np.arange(len(out))
    out["time_idx_sq"] = out["time_idx"] ** 2
    out["year_month_index"] = out["year_num"] * 12 + out["month_num"]

    first_date = pd.to_datetime(out[DATE_COL]).min()
    out["months_since_start"] = (
        (out[DATE_COL].dt.year - first_date.year) * 12
        + (out[DATE_COL].dt.month - first_date.month)
    )

    out["summer_trend"] = out["is_summer"] * out["time_idx"]
    out["winter_trend"] = out["is_winter"] * out["time_idx"]
    out["spring_trend"] = out["is_spring"] * out["time_idx"]
    out["fall_trend"] = out["is_fall"] * out["time_idx"]

    if "covid_flag" in out.columns:
        out["covid_x_month_sin"] = out["covid_flag"] * out["month_sin"]
        out["covid_x_month_cos"] = out["covid_flag"] * out["month_cos"]

    return out


def add_lag_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(DATE_COL).reset_index(drop=True)

    for lag in LAGS:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)

    for window in ROLL_WINDOWS:
        shifted = out[target_col].shift(1)
        out[f"{target_col}_roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"{target_col}_roll_std_{window}"] = shifted.rolling(window).std()
        out[f"{target_col}_roll_min_{window}"] = shifted.rolling(window).min()
        out[f"{target_col}_roll_max_{window}"] = shifted.rolling(window).max()

    out[f"{target_col}_seasonal_diff_12"] = out[target_col].shift(1) - out[target_col].shift(13)
    # seasonal_ratio_12 removed: at forecast time target is NaN → ratio is NaN → imputed to training median, distorting predictions
    out[f"{target_col}_yoy_change"] = out[target_col].shift(1) - out[target_col].shift(13)
    out[f"{target_col}_lag_12_vs_6"] = out[target_col].shift(12) - out[target_col].shift(6)

    lag_cols = [c for c in out.columns if "_lag_" in c or "_roll_" in c or "_seasonal_" in c or "_yoy_" in c]
    logging.info(
        "Added lag/rolling/seasonal features | target=%s | lag_feature_count=%s",
        target_col,
        len(lag_cols),
    )
    return out


def build_course_frame(course_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    course_name = course_df[COURSE_COL].iloc[0]
    logging.info("Building feature frame for course=%s", course_name)

    out = course_df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values(DATE_COL).reset_index(drop=True)

    out = add_time_features(out)
    out = add_calendar_features(out)
    out = add_trend_and_interaction_features(out)
    out = add_lag_features(out, target_col=target_col)

    logging.info(
        "Built feature frame for course=%s | final_rows=%s | columns=%s",
        course_name,
        len(out),
        len(out.columns),
    )
    return out


def get_feature_columns(df: pd.DataFrame, target_col: str):
    exclude_cols = {
        target_col,
        DATE_COL,
        COURSE_COL,
        "year",
        "month",
    }

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    logging.info(
        "Feature split complete | numeric_cols=%s | categorical_cols=%s",
        len(numeric_cols),
        len(categorical_cols),
    )
    logging.info("Numeric columns: %s", numeric_cols)
    logging.info("Categorical columns: %s", categorical_cols)

    return numeric_cols, categorical_cols


def make_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    logging.info(
        "Created preprocessor | numeric_transformers=%s | categorical_transformers=%s",
        int(bool(numeric_cols)),
        int(bool(categorical_cols)),
    )
    return ColumnTransformer(transformers=transformers)


# -----------------------------------------------------------------------------
# DIAGNOSTICS
# -----------------------------------------------------------------------------
def safe_adf(series: pd.Series):
    try:
        stat, pvalue, _, _, _, _ = adfuller(series.dropna(), autolag="AIC")
        return stat, pvalue
    except Exception as e:
        logging.warning("ADF failed: %s", e)
        return np.nan, np.nan


def safe_kpss(series: pd.Series):
    try:
        stat, pvalue, _, _ = kpss(series.dropna(), regression="c", nlags="auto")
        return stat, pvalue
    except Exception as e:
        logging.warning("KPSS failed: %s", e)
        return np.nan, np.nan


def run_course_diagnostics(course_df: pd.DataFrame, target_col: str, outdir: Path):
    course_name = course_df[COURSE_COL].iloc[0]
    diag_dir = outdir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    series_df = course_df[[DATE_COL, target_col]].dropna().copy()
    series_df[DATE_COL] = pd.to_datetime(series_df[DATE_COL])
    series_df = series_df.sort_values(DATE_COL)
    series = series_df.set_index(DATE_COL)[target_col].asfreq("MS")

    if len(series.dropna()) < 24:
        logging.warning("Skipping diagnostics for course=%s because series is too short", course_name)
        return

    adf_stat, adf_pvalue = safe_adf(series)
    kpss_stat, kpss_pvalue = safe_kpss(series)

    diag_row = {
        "course": course_name,
        "n_obs": len(series.dropna()),
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_pvalue,
    }

    try:
        stl = STL(series.interpolate(limit_direction="both"), period=12, robust=True).fit()
        diag_row["trend_strength"] = 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid)
        diag_row["seasonal_strength"] = 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid)

        stl_plot_path = diag_dir / f"{course_name}_stl.png"
        fig = stl.plot()
        fig.set_size_inches(12, 8)
        plt.tight_layout()
        plt.savefig(stl_plot_path)
        plt.close(fig)
    except Exception as e:
        logging.warning("STL failed for course=%s: %s", course_name, e)
        diag_row["trend_strength"] = np.nan
        diag_row["seasonal_strength"] = np.nan

    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(series.dropna(), lags=min(DIAGNOSTICS_LAGS, len(series.dropna()) - 1), ax=axes[0])
        plot_pacf(series.dropna(), lags=min(DIAGNOSTICS_LAGS, len(series.dropna()) // 2 - 1), ax=axes[1], method="ywm")
        axes[0].set_title(f"ACF - {course_name}")
        axes[1].set_title(f"PACF - {course_name}")
        plt.tight_layout()
        plt.savefig(diag_dir / f"{course_name}_acf_pacf.png")
        plt.close(fig)
    except Exception as e:
        logging.warning("ACF/PACF failed for course=%s: %s", course_name, e)

    pd.DataFrame([diag_row]).to_csv(diag_dir / f"{course_name}_diagnostics.csv", index=False)
    logging.info("Saved diagnostics for course=%s", course_name)


# -----------------------------------------------------------------------------
# TIME-SERIES VALIDATION
# -----------------------------------------------------------------------------
def make_expanding_splits(df: pd.DataFrame, min_train_months: int, valid_months: int):
    unique_months = sorted(df[DATE_COL].dropna().unique())
    unique_months = pd.to_datetime(unique_months)

    logging.info(
        "Creating expanding splits | unique_months=%s | min_train_months=%s | valid_months=%s",
        len(unique_months),
        min_train_months,
        valid_months,
    )

    splits = []
    train_end = min_train_months

    while train_end + valid_months <= len(unique_months):
        train_months = unique_months[:train_end]
        valid_months_slice = unique_months[train_end: train_end + valid_months]

        train_idx = df[df[DATE_COL].isin(train_months)].index.to_numpy()
        valid_idx = df[df[DATE_COL].isin(valid_months_slice)].index.to_numpy()

        if len(train_idx) > 0 and len(valid_idx) > 0:
            splits.append((train_idx, valid_idx))
            logging.info(
                "Created split %s | train_rows=%s | valid_rows=%s | train_end=%s | valid_start=%s | valid_end=%s",
                len(splits),
                len(train_idx),
                len(valid_idx),
                train_months.max(),
                valid_months_slice.min(),
                valid_months_slice.max(),
            )

        train_end += valid_months

    logging.info("Total CV splits created: %s", len(splits))
    return splits


def evaluate_model(X, y, pipeline, splits, course_name: str, model_name: str, date_series: pd.Series):
    fold_rows = []
    oof_rows = []

    logging.info(
        "Starting CV evaluation | course=%s | model=%s | n_splits=%s",
        course_name,
        model_name,
        len(splits),
    )

    for fold_num, (train_idx, valid_idx) in enumerate(splits, start=1):
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_valid = X.loc[valid_idx]
        y_valid = y.loc[valid_idx]
        valid_dates = pd.to_datetime(date_series.loc[valid_idx]).reset_index(drop=True)

        logging.info(
            "Fitting fold | course=%s | model=%s | fold=%s | train_shape=%s | valid_shape=%s",
            course_name,
            model_name,
            fold_num,
            X_train.shape,
            X_valid.shape,
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_valid)
        preds = np.maximum(preds, 0)

        fold_rmse = rmse(y_valid, preds)
        fold_mae = mean_absolute_error(y_valid, preds)
        fold_mape = mape(y_valid, preds)

        logging.info(
            "Finished fold | course=%s | model=%s | fold=%s | rmse=%.4f | mae=%.4f | mape=%.2f",
            course_name,
            model_name,
            fold_num,
            fold_rmse,
            fold_mae,
            fold_mape,
        )

        fold_rows.append(
            {
                "fold": fold_num,
                "rmse": fold_rmse,
                "mae": fold_mae,
                "mape": fold_mape,
            }
        )

        for horizon_num, (dt, actual, pred) in enumerate(zip(valid_dates, y_valid.to_numpy(), preds), start=1):
            oof_rows.append(
                {
                    "course": course_name,
                    "model_name": model_name,
                    "fold": fold_num,
                    DATE_COL: dt,
                    "horizon": horizon_num,
                    "actual": actual,
                    "prediction": pred,
                    "residual": actual - pred,
                    "abs_error": abs(actual - pred),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)

    summary = {
        "rmse_mean": fold_df["rmse"].mean(),
        "mae_mean": fold_df["mae"].mean(),
        "mape_mean": fold_df["mape"].mean(),
        "fold_df": fold_df,
        "oof_df": oof_df,
    }

    logging.info(
        "Completed CV | course=%s | model=%s | rmse_mean=%.4f | mae_mean=%.4f | mape_mean=%.2f",
        course_name,
        model_name,
        summary["rmse_mean"],
        summary["mae_mean"],
        summary["mape_mean"],
    )

    return summary


def build_residual_quantiles(oof_df: pd.DataFrame):
    residual_quantiles = {}
    if oof_df.empty:
        return residual_quantiles

    for horizon, grp in oof_df.groupby("horizon"):
        residuals = grp["residual"].dropna().to_numpy()
        if len(residuals) == 0:
            continue
        residual_quantiles[horizon] = {}
        for conf in CONFIDENCE_LEVELS:
            alpha = 1 - conf
            lower_q = float(np.quantile(residuals, alpha / 2))
            upper_q = float(np.quantile(residuals, 1 - alpha / 2))
            residual_quantiles[horizon][conf] = (lower_q, upper_q)

    return residual_quantiles


# -----------------------------------------------------------------------------
# MODEL TRAINING PER COURSE
# -----------------------------------------------------------------------------
def train_best_model_for_course(course_df: pd.DataFrame, target_col: str, outdir: Path):
    course_name = course_df[COURSE_COL].iloc[0]
    logging.info("--------------------------------------------------")
    logging.info("Training started for course=%s", course_name)
    logging.info("Raw course rows=%s", len(course_df))

    if RUN_DIAGNOSTICS:
        run_course_diagnostics(course_df, target_col=target_col, outdir=outdir)

    df = build_course_frame(course_df, target_col=target_col)
    df = df[df[target_col].notna()].copy()
    logging.info("Rows after dropping missing target | course=%s | rows=%s", course_name, len(df))

    lag_cols = [
        c for c in df.columns
        if "_lag_" in c or "_roll_" in c or "_seasonal_" in c or "_yoy_" in c
    ]
    logging.info("Lag columns for course=%s: %s", course_name, lag_cols)

    df = df.dropna(subset=lag_cols, how="any").copy()
    logging.info(
        "Rows after dropping lag-missing rows | course=%s | rows=%s",
        course_name,
        len(df),
    )

    if len(df) < MIN_TRAIN_MONTHS + VALID_MONTHS:
        logging.warning(
            "Not enough data after lagging for course=%s | rows=%s | required_min=%s",
            course_name,
            len(df),
            MIN_TRAIN_MONTHS + VALID_MONTHS,
        )
        return None

    numeric_cols, categorical_cols = get_feature_columns(df, target_col=target_col)
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)

    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    date_series = pd.to_datetime(df[DATE_COL]).copy()

    logging.info(
        "Prepared modeling matrices | course=%s | X_shape=%s | y_len=%s",
        course_name,
        X.shape,
        len(y),
    )

    splits = make_expanding_splits(
        df=df,
        min_train_months=MIN_TRAIN_MONTHS,
        valid_months=VALID_MONTHS,
    )

    if not splits:
        logging.warning("No CV splits available for course=%s", course_name)
        return None

    results = []
    best_name = None
    best_pipeline = None
    best_rmse = None
    best_oof_df = pd.DataFrame()
    best_residual_quantiles = {}

    for model_name, model in MODEL_SPECS.items():
        logging.info("Testing model | course=%s | model=%s", course_name, model_name)

        param_grid = MODEL_PARAM_GRID.get(model_name, {}) or {}

        if HYPERPARAMETER_TUNING and param_grid:
            keys = list(param_grid.keys())
            values_product = list(product(*param_grid.values()))
            candidates = [dict(zip(keys, v)) for v in values_product]
        else:
            candidates = [{}]

        for candidate_params in candidates:
            candidate_model = clone(model).set_params(
                **{k.replace("model__", ""): v for k, v in candidate_params.items()}
            )

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", candidate_model),
                ]
            )

            candidate_label = model_name
            if candidate_params:
                candidate_label = f"{model_name}_" + "_".join(
                    [f"{k.split('__')[-1]}={v}" for k, v in candidate_params.items()]
                )

            metrics = evaluate_model(
                X=X,
                y=y,
                pipeline=pipeline,
                splits=splits,
                course_name=course_name,
                model_name=candidate_label,
                date_series=date_series,
            )

            results.append(
                {
                    "course": course_name,
                    "model_name": candidate_label,
                    "rmse_mean": metrics["rmse_mean"],
                    "mae_mean": metrics["mae_mean"],
                    "mape_mean": metrics["mape_mean"],
                }
            )

            if best_rmse is None or metrics["rmse_mean"] < best_rmse:
                best_rmse = metrics["rmse_mean"]
                best_name = candidate_label
                best_pipeline = pipeline
                best_oof_df = metrics["oof_df"].copy()
                best_residual_quantiles = build_residual_quantiles(best_oof_df)
                logging.info(
                    "New best model found | course=%s | model=%s | rmse=%.4f",
                    course_name,
                    best_name,
                    best_rmse,
                )

    if best_pipeline is None:
        logging.error("No model could be selected for course=%s. Skipping.", course_name)
        return None

    logging.info("Refitting best model on full course history | course=%s | model=%s", course_name, best_name)
    best_pipeline.fit(X, y)

    feature_dir = outdir
    feature_dir.mkdir(parents=True, exist_ok=True)

    try:
        importance = permutation_importance(best_pipeline, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        feature_importance_df = (
            pd.DataFrame({
                "feature": feature_cols,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

        feature_importance_file = feature_dir / f"{course_name}_feature_importance.csv"
        feature_importance_df.to_csv(feature_importance_file, index=False)
        logging.info(
            "Feature importance saved | course=%s | file=%s | top_features=%s",
            course_name,
            feature_importance_file,
            feature_importance_df.head(7)["feature"].tolist(),
        )
    except Exception as e:
        logging.warning("Feature importance computation failed for course=%s: %s", course_name, e)

    if not best_oof_df.empty:
        best_oof_df.to_csv(feature_dir / f"{course_name}_best_model_oof_predictions.csv", index=False)

    residual_rows = []
    for horizon, horizon_map in best_residual_quantiles.items():
        row = {"course": course_name, "horizon": horizon}
        for conf, (lower_q, upper_q) in horizon_map.items():
            pct = int(conf * 100)
            row[f"residual_q_low_{pct}"] = lower_q
            row[f"residual_q_high_{pct}"] = upper_q
        residual_rows.append(row)
    if residual_rows:
        pd.DataFrame(residual_rows).to_csv(feature_dir / f"{course_name}_residual_quantiles.csv", index=False)

    results_df = pd.DataFrame(results).sort_values(
        ["rmse_mean", "mae_mean", "mape_mean"]
    ).reset_index(drop=True)

    logging.info("Training finished for course=%s", course_name)
    logging.info("Best result table for course=%s:\n%s", course_name, results_df.to_string(index=False))

    return {
        "course": course_name,
        "training_df": df,
        "feature_cols": feature_cols,
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "best_residual_quantiles": best_residual_quantiles,
        "results_df": results_df,
    }


# -----------------------------------------------------------------------------
# RECURSIVE FORECAST PER COURSE
# -----------------------------------------------------------------------------
def build_future_stub(course_df: pd.DataFrame, target_col: str, horizon_months: int):
    course_name = course_df[COURSE_COL].iloc[0]
    course_df = course_df.copy()
    course_df[DATE_COL] = pd.to_datetime(course_df[DATE_COL])
    course_df = course_df.sort_values(DATE_COL).reset_index(drop=True)

    last_row = course_df.iloc[-1].copy()
    last_date = course_df[DATE_COL].max()

    start_date = last_date + pd.offsets.MonthBegin(1)
    end_date = min(start_date + pd.DateOffset(months=horizon_months - 1), FORECAST_END_DATE)

    if start_date > end_date:
        logging.warning(
            "No forecast range for course=%s because start_date=%s is after FORECAST_END_DATE=%s",
            course_name,
            start_date,
            FORECAST_END_DATE,
        )
        return pd.DataFrame(columns=[DATE_COL, COURSE_COL, "prediction", "year", "month"])

    future_dates = pd.date_range(start_date, end=end_date, freq="MS")

    logging.info(
        "Building future stub | course=%s | last_date=%s | horizon_months=%s",
        course_name,
        last_date,
        horizon_months,
    )

    future_rows = []
    for d in future_dates:
        row = last_row.copy()
        row[DATE_COL] = d
        row["year"] = d.year
        row["month"] = d.month
        row[target_col] = np.nan

        if "covid_flag" in row.index:
            row["covid_flag"] = 0

        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    logging.info("Future stub created | course=%s | rows=%s", course_name, len(future_df))
    return future_df


def add_prediction_intervals(pred: float, step_num: int, residual_quantiles: dict):
    interval_values = {}

    horizon_key = step_num if step_num in residual_quantiles else max(residual_quantiles.keys(), default=None)
    if horizon_key is None:
        for conf in CONFIDENCE_LEVELS:
            pct = int(conf * 100)
            interval_values[f"lower_{pct}"] = np.nan
            interval_values[f"upper_{pct}"] = np.nan
        return interval_values

    growth_factor = np.sqrt(step_num / max(horizon_key, 1))

    for conf in CONFIDENCE_LEVELS:
        pct = int(conf * 100)
        if conf not in residual_quantiles[horizon_key]:
            interval_values[f"lower_{pct}"] = np.nan
            interval_values[f"upper_{pct}"] = np.nan
            continue

        low_q, high_q = residual_quantiles[horizon_key][conf]
        lower_bound = max(0, pred + low_q * growth_factor)
        upper_bound = max(0, pred + high_q * growth_factor)
        interval_values[f"lower_{pct}"] = lower_bound
        interval_values[f"upper_{pct}"] = upper_bound

    return interval_values


def recursive_forecast_course(course_df: pd.DataFrame, fitted_pipeline, target_col: str, horizon_months: int, residual_quantiles: dict):
    course_name = course_df[COURSE_COL].iloc[0]
    logging.info("Starting recursive forecast | course=%s", course_name)

    history = course_df.copy()
    history[DATE_COL] = pd.to_datetime(history[DATE_COL])

    future_stub = build_future_stub(history, target_col=target_col, horizon_months=horizon_months)

    work_df = pd.concat([history, future_stub], ignore_index=True, sort=False)
    work_df = work_df.sort_values(DATE_COL).reset_index(drop=True)

    forecast_dates = sorted(future_stub[DATE_COL].unique())
    forecast_rows = []

    for step_num, forecast_date in enumerate(forecast_dates, start=1):
        logging.info("Forecasting step | course=%s | forecast_date=%s", course_name, forecast_date)

        featured = build_course_frame(work_df, target_col=target_col)
        current_row = featured[featured[DATE_COL] == forecast_date].copy()

        lag_cols = [
            c for c in featured.columns
            if "_lag_" in c or "_roll_" in c or "_seasonal_" in c or "_yoy_" in c
        ]

        if current_row.empty:
            logging.warning(
                "Skipping forecast step because row not found | course=%s | forecast_date=%s",
                course_name,
                forecast_date,
            )
            continue

        if current_row[lag_cols].isna().all(axis=1).iloc[0]:
            logging.warning(
                "Forecast step has all lag/roll_na on row; applying persistence fallback | course=%s | forecast_date=%s",
                course_name,
                forecast_date,
            )
            previous_value = work_df[work_df[DATE_COL] < forecast_date][target_col].dropna()
            if not previous_value.empty:
                pred = float(previous_value.iloc[-1])
                interval_values = add_prediction_intervals(pred, step_num, residual_quantiles)
                forecast_rows.append(
                    {DATE_COL: forecast_date, COURSE_COL: course_name, "prediction": pred, **interval_values}
                )
                work_df.loc[work_df[DATE_COL] == forecast_date, target_col] = pred
                continue
            logging.warning(
                "No previous value available for persistence fallback | course=%s | forecast_date=%s",
                course_name,
                forecast_date,
            )
            continue

        numeric_cols, categorical_cols = get_feature_columns(featured, target_col=target_col)
        X_current = current_row[numeric_cols + categorical_cols].copy()

        pred = fitted_pipeline.predict(X_current)[0]
        pred = max(pred, 0)
        interval_values = add_prediction_intervals(pred, step_num, residual_quantiles)

        logging.info(
            "Forecast complete | course=%s | forecast_date=%s | prediction=%.3f",
            course_name,
            forecast_date,
            pred,
        )

        forecast_rows.append(
            {
                DATE_COL: forecast_date,
                COURSE_COL: course_name,
                "prediction": pred,
                **interval_values,
            }
        )

        work_df.loc[work_df[DATE_COL] == forecast_date, target_col] = pred

    forecast_df = pd.DataFrame(forecast_rows)
    if not forecast_df.empty:
        forecast_df["year"] = pd.to_datetime(forecast_df[DATE_COL]).dt.year
        forecast_df["month"] = pd.to_datetime(forecast_df[DATE_COL]).dt.month

    logging.info(
        "Finished recursive forecast | course=%s | forecast_rows=%s",
        course_name,
        len(forecast_df),
    )
    return forecast_df


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def plot_forecast_vs_actual(actual_df: pd.DataFrame, forecast_df: pd.DataFrame):
    outdir = Path("forecast/forecast_metrics")
    outdir.mkdir(parents=True, exist_ok=True)

    model_results_path = outdir / "per_course_model_results.csv"
    if not model_results_path.exists():
        logging.warning("Model results file not found: %s", model_results_path)
        return
    model_results = pd.read_csv(model_results_path)

    for course in sorted(actual_df[COURSE_COL].dropna().unique()):
        actual_course = actual_df[actual_df[COURSE_COL] == course].sort_values(DATE_COL)
        if actual_course.empty:
            continue

        course_models = model_results[model_results["course"] == course].sort_values("rmse_mean").head(5)
        if course_models.empty:
            continue

        plt.figure(figsize=(12, 7))
        plt.plot(actual_course[DATE_COL], actual_course[TARGET_COL], marker="o", label="actual", color="black", alpha=0.8)

        model_forecast = forecast_df[(forecast_df[COURSE_COL] == course)].copy()
        if not model_forecast.empty:
            plt.plot(model_forecast[DATE_COL], model_forecast["prediction"], marker="x", linestyle="--", label="best forecast")

            if {"lower_80", "upper_80"}.issubset(model_forecast.columns):
                plt.fill_between(
                    model_forecast[DATE_COL],
                    model_forecast["lower_80"],
                    model_forecast["upper_80"],
                    alpha=0.20,
                    label="80% interval",
                )

            if {"lower_95", "upper_95"}.issubset(model_forecast.columns):
                plt.fill_between(
                    model_forecast[DATE_COL],
                    model_forecast["lower_95"],
                    model_forecast["upper_95"],
                    alpha=0.12,
                    label="95% interval",
                )

        plt.title(f"Forecast vs Actual for {course}")
        plt.xlabel("Date")
        plt.ylabel(TARGET_COL)
        plt.legend()
        plt.tight_layout()

        filepath = outdir / f"{course.replace(' ', '_')}_forecast_vs_actual.png"
        plt.savefig(filepath)
        plt.close()
        logging.info("Saved forecast plot | course=%s | file=%s", course, filepath)


def main():
    logging.info("Script started")
    logging.info("Reading data from %s", DATA_PATH.resolve())

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}")

    outdir = Path("forecast/forecast_metrics")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    logging.info("Data loaded successfully | shape=%s", df.shape)
    logging.info("Columns: %s", list(df.columns))

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not found in dataset")

    logging.info("Using target column: %s", TARGET_COL)
    logging.info("Date range: %s to %s", df[DATE_COL].min(), df[DATE_COL].max())
    logging.info("Number of courses: %s", df[COURSE_COL].nunique())
    logging.info("Courses found: %s", sorted(df[COURSE_COL].dropna().unique()))

    all_model_results = []
    all_forecasts = []
    best_model_summary = []

    for course in sorted(df[COURSE_COL].dropna().unique()):
        logging.info("==================================================")
        logging.info("Starting course run: %s", course)

        course_df = df[df[COURSE_COL] == course].copy().sort_values(DATE_COL)
        logging.info(
            "Course subset ready | course=%s | rows=%s | min_date=%s | max_date=%s",
            course,
            len(course_df),
            course_df[DATE_COL].min(),
            course_df[DATE_COL].max(),
        )

        trained = train_best_model_for_course(course_df, target_col=TARGET_COL, outdir=outdir)
        if trained is None:
            logging.warning("Skipping course=%s because training returned None", course)
            continue

        course_results = trained["results_df"].copy()
        all_model_results.append(course_results)

        best_model_summary.append(
            {
                "course": course,
                "best_model_name": trained["best_model_name"],
                "best_rmse": course_results.iloc[0]["rmse_mean"],
                "best_mae": course_results.iloc[0]["mae_mean"],
                "best_mape": course_results.iloc[0]["mape_mean"],
            }
        )

        logging.info(
            "Best model selected | course=%s | model=%s | rmse=%.4f | mae=%.4f | mape=%.2f",
            course,
            trained["best_model_name"],
            course_results.iloc[0]["rmse_mean"],
            course_results.iloc[0]["mae_mean"],
            course_results.iloc[0]["mape_mean"],
        )

        forecast_df = recursive_forecast_course(
            course_df=course_df,
            fitted_pipeline=trained["best_pipeline"],
            target_col=TARGET_COL,
            horizon_months=FORECAST_HORIZON_MONTHS,
            residual_quantiles=trained["best_residual_quantiles"],
        )
        if not forecast_df.empty:
            all_forecasts.append(forecast_df)
        else:
            logging.warning("Skipping empty forecast for course=%s", course)

    if all_model_results:
        model_results_df = pd.concat(all_model_results, ignore_index=True)
        model_results_df.to_csv(outdir / "per_course_model_results.csv", index=False)
        logging.info("Saved per_course_model_results.csv | rows=%s", len(model_results_df))

    if best_model_summary:
        best_model_df = pd.DataFrame(best_model_summary).sort_values("course")
        best_model_df.to_csv(outdir / "per_course_best_models.csv", index=False)
        logging.info("Saved per_course_best_models.csv | rows=%s", len(best_model_df))
        logging.info("Best models summary:\n%s", best_model_df.to_string(index=False))

    if all_forecasts:
        forecast_df = pd.concat(all_forecasts, ignore_index=True)
        if forecast_df.empty:
            logging.warning("All course forecasts are empty; skipping output files")
        else:
            forecast_df = forecast_df.sort_values([COURSE_COL, DATE_COL]).reset_index(drop=True)
            forecast_df.to_csv(outdir / "per_course_monthly_forecast_12m.csv", index=False)
            logging.info("Saved per_course_monthly_forecast_12m.csv | rows=%s", len(forecast_df))

            annual_summary = (
                forecast_df.groupby([COURSE_COL, "year"], as_index=False)["prediction"]
                .sum()
                .rename(columns={"prediction": "forecast_students"})
            )
            annual_summary.to_csv(outdir / "per_course_annual_forecast.csv", index=False)
            logging.info("Saved per_course_annual_forecast.csv | rows=%s", len(annual_summary))
            logging.info("Annual forecast summary:\n%s", annual_summary.to_string(index=False))

            plot_forecast_vs_actual(df, forecast_df)

    logging.info("Script finished successfully")


if __name__ == "__main__":
    main()
