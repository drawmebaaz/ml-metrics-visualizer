from io import BytesIO
from pathlib import Path
import hashlib
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


@st.cache_data(show_spinner=False)
def load_default_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


def load_dataset(uploaded_file_name: str | None, uploaded_file_bytes: bytes | None) -> pd.DataFrame:
    if uploaded_file_bytes is not None:
        return load_uploaded_dataset(uploaded_file_name or "uploaded.csv", uploaded_file_bytes)
    return load_default_dataset(str(SAMPLE_DATA))


def dataset_fingerprint(uploaded_file_name: str | None, uploaded_file_bytes: bytes | None, df: pd.DataFrame) -> str:
    if uploaded_file_bytes is not None:
        digest = hashlib.sha1(uploaded_file_bytes).hexdigest()[:12]
        return f"upload:{uploaded_file_name}:{digest}:{df.shape}"
    return f"default:{SAMPLE_DATA.name}:{df.shape}"


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
    uploaded_file_name = uploaded_file.name if uploaded_file is not None else None
    uploaded_file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None

try:
    df = load_dataset(uploaded_file_name, uploaded_file_bytes)
except FileNotFoundError:
    st.error(
        "The default dataset was not found. Run "
        "`python scripts\\download_kaggle_student_performance.py` to regenerate it."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("The loaded dataset has no usable rows or columns.")
    st.stop()

fingerprint = dataset_fingerprint(uploaded_file_name, uploaded_file_bytes, df)
if st.session_state.get("dataset_fingerprint") != fingerprint:
    st.session_state["dataset_fingerprint"] = fingerprint
    st.session_state.pop("training_result", None)
    st.session_state.pop("training_error", None)

with st.sidebar:
    st.caption(f"Loaded {df.shape[0]:,} rows and {df.shape[1]:,} columns.")
    with st.form("training_controls"):
        st.subheader("Training")
        target_column = st.selectbox(
            "Target column",
            options=list(df.columns),
            index=max(len(df.columns) - 1, 0),
        )
        model_name = st.selectbox("Model", options=list(SUPPORTED_MODELS.keys()))
        test_size = st.slider("Test size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
        train_clicked = st.form_submit_button("Train model", type="primary", use_container_width=True)

st.subheader("Data Preview")
st.dataframe(df.head(50), use_container_width=True)
st.caption(f"{df.shape[0]} rows, {df.shape[1]} columns")

if train_clicked:
    try:
        with st.spinner("Training model and calculating metrics..."):
            st.session_state["training_result"] = train_classifier(
                df=df,
                target=target_column,
                model_name=model_name,
                test_size=test_size,
                random_state=int(random_state),
            )
            st.session_state["training_error"] = None
    except ValueError as exc:
        st.session_state["training_result"] = None
        st.session_state["training_error"] = str(exc)

if st.session_state.get("training_error"):
    st.error(st.session_state["training_error"])
elif st.session_state.get("training_result") is not None:
    result = st.session_state["training_result"]
    st.subheader("Model Quality")
    st.caption(f"Positive class: {result.positive_class}")
    render_metric_cards(result.metrics)

    left, right = st.columns(2)
    with left:
        render_confusion_matrix(result.confusion_matrix, result.classes)
    with right:
        render_roc_curve(result.roc_curve)

    st.subheader("Test Predictions")
    preview_rows = 250
    st.dataframe(result.test_predictions.head(preview_rows), use_container_width=True)
    if len(result.test_predictions) > preview_rows:
        st.caption(f"Showing first {preview_rows:,} of {len(result.test_predictions):,} test rows.")
else:
    st.info("Choose a target and model, then train the model.")
