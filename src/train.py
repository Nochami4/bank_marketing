import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


RANDOM_STATE = 42
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


def load_data():
    dataset = fetch_ucirepo(id=222)
    features = dataset.data.features.copy()
    target = dataset.data.targets
    if isinstance(target, pd.DataFrame):
        target = target.iloc[:, 0]
    target = target.map({"yes": 1, "no": 0})

    if "duration" in features.columns:
        features = features.drop(columns=["duration"])

    return features, target


def build_model(model_name: str):
    if model_name == "logistic":
        model = LogisticRegression(max_iter=1000, n_jobs=None, random_state=RANDOM_STATE)
    elif model_name == "hgb":
        model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Backward-compat for older scikit-learn versions.
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=skf,
        scoring={
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        },
        n_jobs=None,
        return_train_score=False,
    )

    roc_auc_mean = cv_results["test_roc_auc"].mean()
    roc_auc_std = cv_results["test_roc_auc"].std()
    ap_mean = cv_results["test_average_precision"].mean()
    ap_std = cv_results["test_average_precision"].std()

    print("Cross-validation metrics (5-fold):")
    print(f"  ROC-AUC: {roc_auc_mean:.4f} ± {roc_auc_std:.4f}")
    print(f"  PR-AUC : {ap_mean:.4f} ± {ap_std:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    preds = (proba >= 0.5).astype(int)

    print("Holdout metrics (20% test split):")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC : {ap:.4f}")
    print("Classification report (threshold=0.5):")
    print(classification_report(y_test, preds, digits=4))

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train Bank Marketing model")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "hgb"],
        help="Model type: logistic (default) or hgb",
    )
    args = parser.parse_args()

    X, y = load_data()

    model = build_model(args.model)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        col for col in X.columns if col not in numeric_features
    ]

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    pipeline = Pipeline(
        steps=[("preprocess", preprocessor), ("model", model)]
    )

    print(f"Training model: {args.model}")
    trained_pipeline = evaluate(pipeline, X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_pipeline, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
