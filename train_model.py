from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset" / "train.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
TARGET = "SalePrice"
RANDOM_STATE = 42


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add project features while keeping the original columns available."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(X.columns):
            X["TotalSF"] = (
                X["TotalBsmtSF"].fillna(0)
                + X["1stFlrSF"].fillna(0)
                + X["2ndFlrSF"].fillna(0)
            )
        if {"YrSold", "YearBuilt"}.issubset(X.columns):
            X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        if {"YrSold", "YearRemodAdd"}.issubset(X.columns):
            X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
        if "GarageArea" in X.columns:
            X["HasGarage"] = (X["GarageArea"].fillna(0) > 0).astype(int)
        if "PoolArea" in X.columns:
            X["HasPool"] = (X["PoolArea"].fillna(0) > 0).astype(int)
        return X


class DropMostlyMissingColumns(BaseEstimator, TransformerMixin):
    """Drop columns with more than 80% missing values, learned at fit time."""

    def __init__(self, threshold=0.80):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.keep_columns_ = X.columns[X.isna(
        ).mean() <= self.threshold].tolist()
        return self

    def transform(self, X):
        return X.reindex(columns=self.keep_columns_)


class AmesMissingValueHandler(BaseEstimator, TransformerMixin):
    """Fill categorical blanks with 'None' where appropriate; other values are imputed later."""

    NONE_CATEGORIES = {
        "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
        "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
        "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
        "MasVnrType",
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            if col in self.NONE_CATEGORIES:
                X[col] = X[col].fillna("None")
        return X


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place the Ames Housing train.csv file at Dataset/train.csv."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found in {DATA_PATH.name}.")

    # Match the practical's specified outlier rule.
    if {"GrLivArea", TARGET}.issubset(df.columns):
        outlier_mask = (df["GrLivArea"] > 4000) & (df[TARGET] < 300000)
        removed_outliers = int(outlier_mask.sum())
        df = df.loc[~outlier_mask].copy()
    else:
        removed_outliers = 0

    y = np.log1p(df[TARGET].astype(float))
    X = df.drop(columns=[TARGET, "Id"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline,
             make_column_selector(dtype_include=np.number)),
            ("categorical", categorical_pipeline, make_column_selector(
                dtype_include=["object", "category", "bool"])),
        ],
        remainder="drop",
    )

    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",
        )
        model_name = "XGBoost Regressor"
    except ImportError:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=10,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        model_name = "Random Forest Regressor (xgboost unavailable)"

    pipeline = Pipeline(steps=[
        ("feature_engineering", HouseFeatureEngineer()),
        ("drop_mostly_missing", DropMostlyMissingColumns(threshold=0.80)),
        ("special_missing_values", AmesMissingValueHandler()),
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    print(f"Dataset: {DATA_PATH}")
    print(f"Rows after outlier removal: {len(df)}")
    print(f"Removed outlier rows: {removed_outliers}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Training model: {model_name}")

    start = time.time()
    pipeline.fit(X_train, y_train)
    training_seconds = time.time() - start

    # Convert predictions and actual values back to the original SalePrice scale.
    predictions = np.expm1(pipeline.predict(X_test))
    actual = np.expm1(y_test)

    metrics = {
        "model": model_name,
        "target": TARGET,
        "target_transform": "log1p; predictions converted back using expm1",
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "mae": float(mean_absolute_error(actual, predictions)),
        "r2": float(r2_score(actual, predictions)),
        "training_seconds": round(training_seconds, 3),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "removed_outlier_rows": removed_outliers,
        "random_state": RANDOM_STATE,
        "model_file": str(MODEL_PATH.relative_to(BASE_DIR)),
    }

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print("\nModel training completed successfully.")
    print(f"Saved model:   {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("\nTest-set metrics (original SalePrice scale):")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"MAE:  {metrics['mae']:,.2f}")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"Training time: {training_seconds:.2f} seconds")

    # Quick smoke test: reload the saved model and predict a few test rows.
    loaded_model = joblib.load(MODEL_PATH)
    sample_predictions = np.expm1(loaded_model.predict(X_test.head(5)))
    print("\nSaved-model check (first 5 test rows):")
    for row_number, (actual_price, predicted_price) in enumerate(
        zip(actual.head(5), sample_predictions), start=1
    ):
        print(
            f"{row_number}. Actual: ${actual_price:,.0f} | "
            f"Predicted: ${predicted_price:,.0f}"
        )


if __name__ == "__main__":
    main()
