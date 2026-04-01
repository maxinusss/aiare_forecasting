import logging
from pathlib import Path

import numpy as np
import pandas as pd

import os

# Set working directory to project root (parent of eda folder)
os.chdir(Path(__file__).parent.parent)

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor


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

# Keep this light
LAGS = [1, 2, 3, 12]
ROLL_WINDOWS = [3, 12]

MIN_TRAIN_MONTHS = 24
VALID_MONTHS = 3

# Lightweight model list
MODEL_SPECS = {
    "ridge_small": Ridge(alpha=1.0),
    "ridge_medium": Ridge(alpha=10.0),
    "elasticnet_small": ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=10000, random_state=42),
    "elasticnet_medium": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42),
    "hgb_small": HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=100,
        min_samples_leaf=5,
        random_state=42,
    ),
    "hgb_medium": HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=150,
        min_samples_leaf=5,
        random_state=42,
    ),
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

    logging.info(
        "Added time features | rows=%s | min_date=%s | max_date=%s",
        len(out),
        out[DATE_COL].min(),
        out[DATE_COL].max(),
    )
    return out


def add_lag_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(DATE_COL).reset_index(drop=True)

    for lag in LAGS:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)

    for window in ROLL_WINDOWS:
        out[f"{target_col}_roll_mean_{window}"] = out[target_col].shift(1).rolling(window).mean()

    lag_cols = [c for c in out.columns if "_lag_" in c or "_roll_" in c]
    logging.info(
        "Added lag/rolling features | target=%s | lag_feature_count=%s",
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


def evaluate_model(X, y, pipeline, splits, course_name: str, model_name: str):
    fold_rows = []

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

    fold_df = pd.DataFrame(fold_rows)

    summary = {
        "rmse_mean": fold_df["rmse"].mean(),
        "mae_mean": fold_df["mae"].mean(),
        "mape_mean": fold_df["mape"].mean(),
        "fold_df": fold_df,
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


# -----------------------------------------------------------------------------
# MODEL TRAINING PER COURSE
# -----------------------------------------------------------------------------
def train_best_model_for_course(course_df: pd.DataFrame, target_col: str):
    course_name = course_df[COURSE_COL].iloc[0]
    logging.info("--------------------------------------------------")
    logging.info("Training started for course=%s", course_name)
    logging.info("Raw course rows=%s", len(course_df))

    df = build_course_frame(course_df, target_col=target_col)

    df = df[df[target_col].notna()].copy()
    logging.info("Rows after dropping missing target | course=%s | rows=%s", course_name, len(df))

    lag_cols = [c for c in df.columns if "_lag_" in c or "_roll_" in c]
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

    for model_name, model in MODEL_SPECS.items():
        logging.info("Testing model | course=%s | model=%s", course_name, model_name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        metrics = evaluate_model(
            X=X,
            y=y,
            pipeline=pipeline,
            splits=splits,
            course_name=course_name,
            model_name=model_name,
        )

        results.append(
            {
                "course": course_name,
                "model_name": model_name,
                "rmse_mean": metrics["rmse_mean"],
                "mae_mean": metrics["mae_mean"],
                "mape_mean": metrics["mape_mean"],
            }
        )

        if best_rmse is None or metrics["rmse_mean"] < best_rmse:
            best_rmse = metrics["rmse_mean"]
            best_name = model_name
            best_pipeline = pipeline
            logging.info(
                "New best model found | course=%s | model=%s | rmse=%.4f",
                course_name,
                best_name,
                best_rmse,
            )

    logging.info("Refitting best model on full course history | course=%s | model=%s", course_name, best_name)
    best_pipeline.fit(X, y)

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

    # Limit the forecast horizon to July 2027 at most
    start_date = last_date + pd.offsets.MonthBegin(1)
    end_date = min(start_date + pd.DateOffset(months=horizon_months-1), FORECAST_END_DATE)

    if start_date > end_date:
        logging.warning(
            "No forecast range for course=%s because start_date=%s is after FORECAST_END_DATE=%s",
            course_name,
            start_date,
            FORECAST_END_DATE,
        )
        return pd.DataFrame(columns=[DATE_COL, COURSE_COL, "prediction", "year", "month"])

    future_dates = pd.date_range(
        start_date,
        end=end_date,
        freq="MS",
    )

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


def recursive_forecast_course(course_df: pd.DataFrame, fitted_pipeline, target_col: str, horizon_months: int):
    course_name = course_df[COURSE_COL].iloc[0]
    logging.info("Starting recursive forecast | course=%s", course_name)

    history = course_df.copy()
    history[DATE_COL] = pd.to_datetime(history[DATE_COL])

    future_stub = build_future_stub(history, target_col=target_col, horizon_months=horizon_months)

    work_df = pd.concat([history, future_stub], ignore_index=True, sort=False)
    work_df = work_df.sort_values(DATE_COL).reset_index(drop=True)

    forecast_dates = sorted(future_stub[DATE_COL].unique())
    forecast_rows = []

    for forecast_date in forecast_dates:
        logging.info("Forecasting step | course=%s | forecast_date=%s", course_name, forecast_date)

        featured = build_course_frame(work_df, target_col=target_col)
        current_row = featured[featured[DATE_COL] == forecast_date].copy()

        lag_cols = [c for c in featured.columns if "_lag_" in c or "_roll_" in c]
        current_row = current_row.dropna(subset=lag_cols, how="any")

        if current_row.empty:
            logging.warning(
                "Skipping forecast step due to missing lag features | course=%s | forecast_date=%s",
                course_name,
                forecast_date,
            )
            continue

        numeric_cols, categorical_cols = get_feature_columns(featured, target_col=target_col)
        X_current = current_row[numeric_cols + categorical_cols].copy()

        pred = fitted_pipeline.predict(X_current)[0]
        pred = max(pred, 0)

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
def main():
    logging.info("Script started")
    logging.info("Reading data from %s", DATA_PATH.resolve())

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}")

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

        trained = train_best_model_for_course(course_df, target_col=TARGET_COL)
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
        )
        if not forecast_df.empty:
            all_forecasts.append(forecast_df)
        else:
            logging.warning("Skipping empty forecast for course=%s", course)

    if all_model_results:
        model_results_df = pd.concat(all_model_results, ignore_index=True)
        model_results_df.to_csv("per_course_model_results.csv", index=False)
        logging.info(
            "Saved per_course_model_results.csv | rows=%s",
            len(model_results_df),
        )

    if best_model_summary:
        best_model_df = pd.DataFrame(best_model_summary).sort_values("course")
        best_model_df.to_csv("per_course_best_models.csv", index=False)
        logging.info(
            "Saved per_course_best_models.csv | rows=%s",
            len(best_model_df),
        )
        logging.info("Best models summary:\n%s", best_model_df.to_string(index=False))

    if all_forecasts:
        forecast_df = pd.concat(all_forecasts, ignore_index=True)
        if forecast_df.empty:
            logging.warning("All course forecasts are empty; skipping output files")
        else:
            forecast_df = forecast_df.sort_values([COURSE_COL, DATE_COL]).reset_index(drop=True)
            forecast_df.to_csv("per_course_monthly_forecast_12m.csv", index=False)
            logging.info(
                "Saved per_course_monthly_forecast_12m.csv | rows=%s",
                len(forecast_df),
            )

            annual_summary = (
                forecast_df.groupby([COURSE_COL, "year"], as_index=False)["prediction"]
                .sum()
                .rename(columns={"prediction": "forecast_students"})
            )
            annual_summary.to_csv("per_course_annual_forecast.csv", index=False)
            logging.info(
                "Saved per_course_annual_forecast.csv | rows=%s",
                len(annual_summary),
            )
            logging.info("Annual forecast summary:\n%s", annual_summary.to_string(index=False))

main()