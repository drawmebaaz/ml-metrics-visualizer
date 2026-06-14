from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ModelFactory = Callable[[int], ClassifierMixin]


SUPPORTED_MODELS: dict[str, ModelFactory] = {
    "Logistic Regression": lambda random_state: LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    ),
    "Decision Tree": lambda random_state: DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=random_state,
    ),
    "Random Forest": lambda random_state: RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    ),
}


@dataclass
class TrainingResult:
    model_name: str
    target: str
    positive_class: str
    classes: list[str]
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    roc_curve: dict[str, list[float | None]]
    feature_columns: list[str]
    test_predictions: pd.DataFrame
    pipeline: Pipeline


def train_classifier(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """Train a binary classifier and return metrics needed by the app."""
    validate_inputs(df, target, model_name, test_size)

    clean_df = df.dropna(subset=[target]).copy()
    x = clean_df.drop(columns=[target])
    y_raw = clean_df[target].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = label_encoder.classes_.tolist()

    if len(classes) != 2:
        raise ValueError(
            f"Target column must contain exactly 2 classes. Found {len(classes)} classes: {classes}"
        )

    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in x.columns if column not in numeric_features]
    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns found after removing the target column.")

    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", SUPPORTED_MODELS[model_name](random_state)),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_score = predict_positive_score(pipeline, x_test)
    metrics = calculate_metrics(y_test, y_pred, y_score)
    roc_data = calculate_roc_curve(y_test, y_score)
    prediction_table = build_prediction_table(
        x_test=x_test,
        y_test=y_test,
        y_pred=y_pred,
        y_score=y_score,
        label_encoder=label_encoder,
    )

    return TrainingResult(
        model_name=model_name,
        target=target,
        positive_class=str(classes[1]),
        classes=[str(class_name) for class_name in classes],
        metrics=metrics,
        confusion_matrix=confusion_matrix(y_test, y_pred).astype(int).tolist(),
        roc_curve=roc_data,
        feature_columns=x.columns.tolist(),
        test_predictions=prediction_table,
        pipeline=pipeline,
    )


def validate_inputs(df: pd.DataFrame, target: str, model_name: str, test_size: float) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' does not exist in the dataset.")
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {supported}.")
    if not 0 < test_size < 1:
        raise ValueError("Test size must be between 0 and 1.")


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def predict_positive_score(pipeline: Pipeline, x_test: pd.DataFrame) -> np.ndarray | None:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x_test)[:, 1]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(x_test)
        return np.asarray(scores)
    return None


def calculate_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": np.nan,
    }
    if y_score is not None and len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, y_score)
    return {name: float(value) for name, value in metrics.items()}


def calculate_roc_curve(
    y_test: np.ndarray,
    y_score: np.ndarray | None,
) -> dict[str, list[float | None]]:
    if y_score is None or len(np.unique(y_test)) != 2:
        return {"fpr": [], "tpr": [], "thresholds": []}

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    return {
        "fpr": clean_float_list(fpr),
        "tpr": clean_float_list(tpr),
        "thresholds": clean_float_list(thresholds),
    }


def clean_float_list(values: np.ndarray) -> list[float | None]:
    cleaned = []
    for value in values:
        value = float(value)
        cleaned.append(value if np.isfinite(value) else None)
    return cleaned


def build_prediction_table(
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    table = x_test.copy()
    table.insert(0, "actual", label_encoder.inverse_transform(y_test))
    table.insert(1, "predicted", label_encoder.inverse_transform(y_pred))
    if y_score is not None:
        table.insert(2, "positive_probability", np.round(y_score, 4))
    return table.reset_index(drop=True)
