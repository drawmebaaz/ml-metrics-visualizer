from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_metrics_visualizer.modeling import SUPPORTED_MODELS, train_classifier  # noqa: E402


SAMPLE_DATA = PROJECT_ROOT / "data" / "kaggle_student_performance_pass_fail.csv"


def load_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(SAMPLE_DATA)


def render_metric_cards(metrics: dict[str, float]) -> None:
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        display = "N/A" if pd.isna(value) else f"{value:.3f}"
        col.metric(name.replace("_", " ").title(), display)


def render_confusion_matrix(matrix: list[list[int]], classes: list[str]) -> None:
    labels = [f"Actual {class_name}" for class_name in classes]
    columns = [f"Predicted {class_name}" for class_name in classes]
    matrix_df = pd.DataFrame(matrix, index=labels, columns=columns)
    fig = px.imshow(
        matrix_df,
        color_continuous_scale="Blues",
        aspect="auto",
        title="Confusion Matrix",
    )
    fig.update_traces(text=matrix_df.values, texttemplate="%{text}")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_roc_curve(roc_curve: dict[str, list[float | None]]) -> None:
    if not roc_curve["fpr"] or not roc_curve["tpr"]:
        st.info("ROC curve is unavailable for this run.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc_curve["fpr"],
            y=roc_curve["tpr"],
            mode="lines+markers",
            name="Model",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(page_title="ML Metrics Visualizer", layout="wide")
st.title("ML Metrics Visualizer")
st.caption("Train a binary classifier and inspect the metrics that prove whether it works.")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    df = load_dataset(uploaded_file)

    target_column = st.selectbox(
        "Target column",
        options=list(df.columns),
        index=max(len(df.columns) - 1, 0),
    )
    model_name = st.selectbox("Model", options=list(SUPPORTED_MODELS.keys()))
    test_size = st.slider("Test size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    train_clicked = st.button("Train model", type="primary", use_container_width=True)

st.subheader("Data Preview")
st.dataframe(df.head(50), use_container_width=True)
st.caption(f"{df.shape[0]} rows, {df.shape[1]} columns")

if train_clicked:
    try:
        result = train_classifier(
            df=df,
            target=target_column,
            model_name=model_name,
            test_size=test_size,
            random_state=int(random_state),
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.subheader("Model Quality")
        st.caption(f"Positive class: {result.positive_class}")
        render_metric_cards(result.metrics)

        left, right = st.columns(2)
        with left:
            render_confusion_matrix(result.confusion_matrix, result.classes)
        with right:
            render_roc_curve(result.roc_curve)

        st.subheader("Test Predictions")
        st.dataframe(result.test_predictions, use_container_width=True)
else:
    st.info("Choose a target and model, then train the model.")
