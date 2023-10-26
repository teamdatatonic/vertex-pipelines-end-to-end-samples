# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import hypertune

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from xgboost import XGBRegressor

logging.basicConfig(level=logging.DEBUG)

# used for monitoring during prediction time
TRAINING_DATASET_INFO = "training_dataset.json"
# numeric/categorical features in Chicago trips dataset to be preprocessed
NUM_COLS = ["dayofweek", "hourofday", "trip_distance", "trip_miles", "trip_seconds"]
ORD_COLS = ["company"]
OHE_COLS = ["payment_type"]


def split_xy(df: pd.DataFrame, label: str) -> (pd.DataFrame, pd.Series):
    """Split dataframe into X and y."""
    return df.drop(columns=[label]), df[label]


def indices_in_list(elements: list, base_list: list) -> list:
    """Get indices of specific elements in a base list"""
    return [idx for idx, elem in enumerate(base_list) if elem in elements]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--valid-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
        help="Number of estimators (default: 200)",
    )
    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=10,
        help="Early stopping rounds (default: 10)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="reg:squarederror",
        help="Objective function (default: reg:squarederror)",
    )
    parser.add_argument(
        "--booster", type=str, default="gbtree", help="Booster type (default: gbtree)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.3, help="Learning rate (default: 0.3)"
    )
    parser.add_argument(
        "--min_split_loss",
        type=float,
        default=0,
        help="Minimum split loss (default: 0)",
    )
    parser.add_argument(
        "--max_depth", type=int, default=6, help="Maximum depth (default: 6)"
    )
    parser.add_argument(
        "--label", type=str, required=True, help="Name of the label column (required)"
    )

    args = parser.parse_args()

    logging.info("Read csv files into dataframes")
    df_train = pd.read_csv(args.train_data)
    df_valid = pd.read_csv(args.valid_data)
    df_test = pd.read_csv(args.test_data)

    logging.info("Split dataframes")
    label = args.label
    X_train, y_train = split_xy(df_train, label)
    X_valid, y_valid = split_xy(df_valid, label)
    X_test, y_test = split_xy(df_test, label)

    logging.info("Get the number of unique categories for ordinal encoded columns")
    ordinal_columns = X_train[ORD_COLS]
    n_unique_cat = ordinal_columns.nunique()

    logging.info("Get indices of columns in base data")
    col_list = X_train.columns.tolist()
    num_indices = indices_in_list(NUM_COLS, col_list)
    cat_indices_onehot = indices_in_list(OHE_COLS, col_list)
    cat_indices_ordinal = indices_in_list(ORD_COLS, col_list)

    ordinal_transformers = [
        (
            f"ordinal encoding for {ord_col}",
            OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=n_unique_cat[ord_col]
            ),
            [ord_index],
        )
        for ord_col in ORD_COLS
        for ord_index in cat_indices_ordinal
    ]
    all_transformers = [
        ("numeric_scaling", StandardScaler(), num_indices),
        (
            "one_hot_encoding",
            OneHotEncoder(handle_unknown="ignore"),
            cat_indices_onehot,
        ),
    ] + ordinal_transformers

    logging.info("Build sklearn preprocessing steps")
    preprocesser = ColumnTransformer(transformers=all_transformers)
    logging.info("Build sklearn pipeline with XGBoost model")
    xgb_model = XGBRegressor(
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        objective=args.objective,
        booster=args.booster,
        learning_rate=args.learning_rate,
        min_split_loss=args.min_split_loss,
        max_depth=args.max_depth,
    )

    pipeline = Pipeline(
        steps=[("feature_engineering", preprocesser), ("train_model", xgb_model)]
    )

    logging.info("Transform validation data")
    valid_preprocesser = preprocesser.fit(X_train)
    X_valid_transformed = valid_preprocesser.transform(X_valid)

    logging.info("Fit model")
    pipeline.fit(
        X_train, y_train, train_model__eval_set=[(X_valid_transformed, y_valid)]
    )

    logging.info("Predict test data")
    y_pred = pipeline.predict(X_test)
    y_pred = y_pred.clip(0)

    metrics = {
        "problemType": "regression",
        "rootMeanSquaredError": np.sqrt(skmetrics.mean_squared_error(y_test, y_pred)),
        "meanAbsoluteError": skmetrics.mean_absolute_error(y_test, y_pred),
        "meanAbsolutePercentageError": skmetrics.mean_absolute_percentage_error(
            y_test, y_pred
        ),
        "rSquared": skmetrics.r2_score(y_test, y_pred),
        "rootMeanSquaredLogError": np.sqrt(
            skmetrics.mean_squared_log_error(y_test, y_pred)
        ),
    }

    # DEFINE METRIC
    hp_metric = metrics["rootMeanSquaredError"]

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="rootMeanSquaredError",
        metric_value=hp_metric,
    )


if __name__ == "__main__":
    main()
