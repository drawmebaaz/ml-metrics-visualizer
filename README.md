# ML Metrics Visualizer

Level 0 project from the ML product roadmap.

Upload a CSV, choose a binary target column, train a classical ML model, and inspect the metrics that prove whether it works.

The default dataset is prepared from Kaggle's Student Alcohol Consumption dataset.

## What This Builds

- CSV loading and preview
- Automatic numeric and categorical preprocessing
- Logistic Regression, Decision Tree, and Random Forest classifiers
- Accuracy, precision, recall, F1, and ROC AUC
- Confusion matrix and ROC curve
- Test-set prediction table
- Kaggle download/preparation script
- Reusable training code with pytest coverage

## Dataset

The default CSV is:

`data/kaggle_student_performance_pass_fail.csv`

It is generated from Kaggle dataset `uciml/student-alcohol-consumption`, which contains student demographic, social, school, and grade features for Math and Portuguese courses.

The binary target is:

```text
passed = yes when G3 >= 10, else no
```

To avoid target leakage, the generated training CSV removes `G1`, `G2`, and `G3` from the feature columns. The removed columns are highly related to the final grade, especially `G1` and `G2`.

Regenerate the dataset:

```powershell
python scripts\download_kaggle_student_performance.py
```

## Run Locally

```powershell
cd "C:\Users\Sourabh\OneDrive\Pictures\Documents\ml learning\level0-ml-metrics-visualizer"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Run Tests

```powershell
pytest
```

## Learning Checklist

For every model run, explain:

1. What problem the model solves.
2. What data flows through preprocessing and prediction.
3. Which algorithm is used.
4. Which metric proves it works.
5. What can break, such as class imbalance or bad target choice.
6. How you would explain the result in an interview.

## Suggested First Experiments

- Try the included sample dataset.
- Regenerate the Kaggle dataset using the script.
- Change the target column and notice what errors appear.
- Compare Logistic Regression vs Decision Tree vs Random Forest.
- Look at precision and recall, not only accuracy.
- Inspect false positives and false negatives in the prediction table.
