# Building the ML Metrics Visualizer: What Went Wrong and What It Taught Us

This project started as the first small step in a larger ML product-building roadmap. The goal was simple: build a Streamlit app where a user can upload a CSV, select a binary target column, train a classical ML model, and inspect useful metrics like accuracy, precision, recall, F1 score, ROC AUC, confusion matrix, and ROC curve.

The project looks small from the outside, but even this first Level 0 app exposed several important development lessons. That is exactly why it was worth building.

## 1. The First Dataset Was Too Perfect

The first version used a tiny hand-written student performance CSV with only 40 rows. The features were things like study hours, attendance rate, previous score, assignments submitted, and whether the student passed.

When we trained the model, every metric became `1.000`.

At first glance, that looks excellent. In reality, it was suspicious.

The dataset was too small, too clean, and too strongly separated. With a test size of `0.20`, the model was testing on only 8 rows. The target was also very easy to infer from the features: students with high study hours, high attendance, and high previous scores almost always passed.

So the issue was not that the model had magically become brilliant. The real problem was that the validation setup was too easy to trust.

The fix was to replace the toy CSV with a more realistic dataset from Kaggle: the Student Alcohol Consumption / Student Performance dataset. That gave us 1,044 rows after combining the Math and Portuguese course files, with a more realistic class balance:

```text
passed = yes: 814
passed = no: 230
```

After that change, the metrics became much more believable:

```text
Logistic Regression accuracy: 0.703
Decision Tree accuracy: 0.737
Random Forest accuracy: 0.794
```

This was the first major ML lesson of the project:

> Perfect metrics are not always a win. Sometimes they are a warning.

## 2. Avoiding Target Leakage

The Kaggle dataset includes three grade columns:

```text
G1, G2, G3
```

`G3` is the final grade. We created the binary target like this:

```text
passed = yes when G3 >= 10, else no
```

That immediately created a risk: if `G3` stayed in the feature columns, the model would be given the answer while training.

There was also a softer leakage problem with `G1` and `G2`. These are earlier period grades and are highly related to the final grade. Keeping them would make the prediction much easier, but less useful for learning honest model evaluation.

So the dataset preparation script removes:

```text
G1, G2, G3
```

The model now has to predict pass/fail from demographic, social, school, and study-related features instead of directly reading grade columns.

This made the project more realistic and more educational.

## 3. The Repo Structure Changed Midway

The first implementation was created in the original workspace folder. Then the requirement changed: the repo should live in `Downloads`, with phase-wise folders and project-wise folders inside it.

Initially, a large multi-phase repo structure was created with many future project directories. Then the requirement changed again:

> Only make the very first project of the first phase, like Phase 0, for now.

So the structure was cleaned up and reduced to:

```text
ml-ai-product-roadmap/
  phase-00-ml-playground-projects/
    ml-metrics-visualizer/
```

This was a useful product-development reminder: project structure should match the current scope. Too much empty scaffolding creates noise.

## 4. PowerShell Wildcard Move Error

While moving the first project into the new Downloads repo, a PowerShell command tried to move files using this pattern:

```powershell
Move-Item -LiteralPath "...\level0-ml-metrics-visualizer\*" -Destination ...
```

That failed because `-LiteralPath` treats the wildcard `*` literally instead of expanding it.

The error looked like this:

```text
Cannot move item because the item at '...\level0-ml-metrics-visualizer\*' does not exist.
```

The fix was to enumerate the source directory first and move each item by its real full path:

```powershell
Get-ChildItem -Force -LiteralPath $source | ForEach-Object {
  Move-Item -LiteralPath $_.FullName -Destination $dest
}
```

The lesson:

> In PowerShell, use `-Path` for wildcard expansion and `-LiteralPath` for exact paths. Mixing them carelessly can break file operations.

## 5. Bash Heredoc Habit in PowerShell

At one point, a quick Python inspection command was written using Bash-style heredoc syntax:

```bash
python - <<'PY'
...
PY
```

But the shell was PowerShell, not Bash. PowerShell does not support that syntax, so it threw parsing errors.

The fix was to use PowerShell’s here-string style:

```powershell
@'
print("hello")
'@ | python -
```

The lesson:

> Always adapt quick debugging commands to the actual shell environment.

## 6. Kaggle CSV Parsing Assumption

The Kaggle dataset was downloaded as a zip containing:

```text
student-mat.csv
student-por.csv
student-merge.R
```

The first version of the preparation script assumed the CSV files were semicolon-separated:

```python
pd.read_csv(csv_file, sep=";")
```

That caused this error:

```text
KeyError: 'G3'
```

The reason was simple: in this Kaggle download, the files were comma-separated. Because the wrong delimiter was used, pandas did not read the columns correctly, so `G3` was not found.

The fix was to read the file with pandas’ default comma separator:

```python
pd.read_csv(csv_file)
```

The lesson:

> Never assume a dataset delimiter. Inspect the columns before building processing logic around them.

## 7. Python Import Path Issue

The tests passed because `pyproject.toml` configured pytest to include `src` in the Python path:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

But when running a separate quick Python metrics script, the import failed:

```text
ModuleNotFoundError: No module named 'ml_metrics_visualizer'
```

The fix was to set `PYTHONPATH=src` for that command:

```powershell
$env:PYTHONPATH = 'src'
```

Long term, installing the package in editable mode also solves it:

```powershell
pip install -e .
```

The lesson:

> Passing tests does not always mean every execution path has the same import environment.

## 8. Streamlit Hydration Timing

When checking the local app in the browser, the first page read sometimes showed only Streamlit’s initial shell text, such as:

```text
You need to enable JavaScript to run this app.
```

That did not mean the app was broken. It meant Streamlit had not finished hydrating the frontend yet.

The fix was to wait a little longer and then read the page again. After hydration, the browser showed the expected app title and controls:

```text
ML Metrics Visualizer
Train model
Data Preview
```

The lesson:

> Frontend verification needs to account for app boot time, especially with Streamlit and other JavaScript-rendered UIs.

## 9. Browser Automation Quirks

During browser verification, a few tooling-specific issues appeared.

One wait mode was unsupported:

```text
playwright_wait_for_load_state does not support networkidle
```

The fix was to use `domcontentloaded` instead and then wait briefly for Streamlit to render.

Another issue came from reusing a JavaScript variable name in the browser automation session:

```text
Identifier 'info' has already been declared
```

The fix was to use a different variable name.

Later, after a browser session reset, this appeared:

```text
tab is not defined
```

That meant the browser automation state had been cleared, not that the Streamlit app itself was broken.

The lesson:

> Verification tools have their own state and failure modes. Separate app bugs from testing-tool quirks.

## 10. Generated Cache Cleanup

Running tests created generated folders like:

```text
.pytest_cache
__pycache__
```

These are normal, but they should not clutter the project or Git status.

The fix was to remove them after verification and add common generated paths to `.gitignore`:

```text
.pytest_cache/
__pycache__/
*.pyc
.cache/
```

The Kaggle zip download is also cached under `.cache/kaggle`, but that should remain untracked because it is a generated download artifact.

The lesson:

> A clean repo is part of the engineering work, not a cosmetic extra.

## 11. Why This Project Became Better

The first version worked, but it taught the wrong lesson because the metrics were unrealistically perfect.

The current version is better because:

- It uses a real Kaggle dataset.
- It creates a clear binary target.
- It removes leakage columns.
- It has reusable training logic outside the UI.
- It has tests for the modeling code.
- It has a script to regenerate the dataset.
- It produces realistic metrics instead of fake-perfect ones.

The app is still simple, but it now teaches the right ML habits:

- Do not trust accuracy alone.
- Check precision, recall, F1, ROC AUC, and confusion matrix.
- Watch for leakage.
- Be suspicious of perfect scores.
- Keep data preparation reproducible.
- Separate model logic from UI code.

## Final Reflection

This was only a Phase 0 project, but it already showed the difference between making a demo and building like an ML engineer.

A demo says:

> The model runs.

An ML engineering project asks:

> Is the data honest? Are the metrics meaningful? Can I reproduce the dataset? Can I explain what broke and why?

That is the real value of this first project. The bugs were not distractions from learning. They were the learning.

