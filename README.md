# ML Metrics Visualizer

A beginner-friendly Streamlit app for training binary classifiers and understanding the metrics behind model quality.

The app lets you upload a CSV file, choose a binary target column, train a classical ML model, and inspect accuracy, precision, recall, F1, ROC AUC, a confusion matrix, a ROC curve, and row-level test predictions.

## What It Does

- Uses a Kaggle-derived student performance dataset by default.
- Accepts uploaded CSV files with any binary target column.
- Handles numeric and categorical features with a reusable preprocessing pipeline.
- Trains Logistic Regression, Decision Tree, or Random Forest classifiers.
- Shows the difference between accuracy, precision, recall, F1, and ROC AUC.
- Displays confusion matrix and ROC curve visualizations.
- Keeps reusable training logic in `src/ml_metrics_visualizer/modeling.py`.
- Includes pytest coverage for model training and validation behavior.

## Dataset

The default CSV is:

```text
data/kaggle_student_performance_pass_fail.csv
```

It is generated from Kaggle dataset:

```text
uciml/student-alcohol-consumption
```

The original final grade is converted into the binary target:

```text
passed = yes when G3 >= 10
passed = no when G3 < 10
```

To avoid target leakage, the generated training dataset removes `G1`, `G2`, and `G3` from the feature columns.

The script `scripts/download_kaggle_student_performance.py` can regenerate the dataset when needed.

## Project Structure

```text
ml-metrics-visualizer/
|-- app.py
|-- requirements.txt
|-- pyproject.toml
|-- data/
|   `-- kaggle_student_performance_pass_fail.csv
|-- scripts/
|   `-- download_kaggle_student_performance.py
|-- src/
|   `-- ml_metrics_visualizer/
|       |-- __init__.py
|       `-- modeling.py
`-- tests/
    `-- test_modeling.py
```

## Testing

The project includes pytest coverage for the reusable training code, supported model names, binary target validation, and error handling.

## Why Metrics Can Become 1.000

Perfect accuracy, precision, recall, and F1 can happen for a few reasons:

- The dataset may be easy for the selected model.
- A feature may leak the target directly or indirectly.
- The test split may be small or unusually simple.
- The model may be overfitting, especially with flexible models.

For this project, the generated dataset removes the most obvious leakage columns (`G1`, `G2`, `G3`). If you upload your own CSV and see every metric at `1.000`, inspect the feature columns, increase the test size, try another random state, and compare a simple model against a more flexible one.

## Learning Checklist

For every run, be able to explain:

1. What binary classification problem is being solved.
2. Which columns are features and which column is the target.
3. How numeric and categorical features are preprocessed.
4. What each metric says about model behavior.
5. Why a confusion matrix can reveal mistakes hidden by accuracy.
6. What could cause suspiciously perfect metrics.

## License

This project is for learning and portfolio use.
