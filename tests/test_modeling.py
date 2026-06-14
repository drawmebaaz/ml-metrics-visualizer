import numpy as np
import pandas as pd
import pytest

from ml_metrics_visualizer.modeling import SUPPORTED_MODELS, train_classifier


def make_training_frame(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    study_hours = rng.normal(5, 1.5, rows).clip(0.5, 9.5)
    attendance_rate = rng.normal(0.78, 0.12, rows).clip(0.35, 0.99)
    previous_score = rng.normal(62, 14, rows).clip(20, 95)
    assignments_submitted = rng.integers(2, 11, rows)
    internet_access = rng.choice(["yes", "no"], rows, p=[0.75, 0.25])

    signal = (
        study_hours * 0.35
        + attendance_rate * 4.0
        + previous_score * 0.06
        + assignments_submitted * 0.18
        + (internet_access == "yes") * 0.45
    )
    passed = np.where(signal > np.median(signal), "yes", "no")

    return pd.DataFrame(
        {
            "study_hours": study_hours,
            "attendance_rate": attendance_rate,
            "previous_score": previous_score,
            "assignments_submitted": assignments_submitted,
            "internet_access": internet_access,
            "passed": passed,
        }
    )


def test_train_classifier_returns_metrics_and_predictions() -> None:
    df = make_training_frame()

    result = train_classifier(df, target="passed", model_name="Logistic Regression")

    assert result.model_name == "Logistic Regression"
    assert result.target == "passed"
    assert set(result.metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= result.metrics[name] <= 1.0 for name in ["accuracy", "precision", "recall", "f1"])
    assert result.confusion_matrix and len(result.confusion_matrix) == 2
    assert {"actual", "predicted", "positive_probability"}.issubset(result.test_predictions.columns)


def test_all_supported_models_can_train() -> None:
    df = make_training_frame()

    for model_name in SUPPORTED_MODELS:
        result = train_classifier(df, target="passed", model_name=model_name)
        assert result.metrics["accuracy"] >= 0.0


def test_target_must_be_binary() -> None:
    df = make_training_frame()
    df["passed"] = np.resize(["low", "medium", "high"], len(df))

    with pytest.raises(ValueError, match="exactly 2 classes"):
        train_classifier(df, target="passed", model_name="Logistic Regression")


def test_unknown_model_is_rejected() -> None:
    df = make_training_frame()

    with pytest.raises(ValueError, match="Unsupported model"):
        train_classifier(df, target="passed", model_name="Naive Bayes")

