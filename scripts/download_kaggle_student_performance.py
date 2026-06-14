from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache" / "kaggle"

KAGGLE_DATASET = "uciml/student-alcohol-consumption"
KAGGLE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
OUTPUT_CSV = DATA_DIR / "kaggle_student_performance_pass_fail.csv"
METADATA_JSON = DATA_DIR / "kaggle_student_performance_metadata.json"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = CACHE_DIR / "student-alcohol-consumption.zip"
    if not zip_path.exists():
        print(f"Downloading Kaggle dataset: {KAGGLE_DATASET}")
        urlretrieve(KAGGLE_URL, zip_path)
    else:
        print(f"Using cached Kaggle zip: {zip_path}")

    math_df = read_subject(zip_path, "student-mat.csv", "math")
    portuguese_df = read_subject(zip_path, "student-por.csv", "portuguese")
    combined = pd.concat([math_df, portuguese_df], ignore_index=True)

    combined["passed"] = combined["G3"].apply(lambda grade: "yes" if grade >= 10 else "no")

    leakage_columns = ["G1", "G2", "G3"]
    model_df = combined.drop(columns=leakage_columns)
    feature_columns = [column for column in model_df.columns if column != "passed"]
    model_df = model_df[feature_columns + ["passed"]]
    model_df.to_csv(OUTPUT_CSV, index=False)

    metadata = {
        "source": "Kaggle Student Alcohol Consumption dataset",
        "kaggle_dataset": KAGGLE_DATASET,
        "kaggle_url": "https://www.kaggle.com/datasets/uciml/student-alcohol-consumption",
        "rows": int(model_df.shape[0]),
        "columns": int(model_df.shape[1]),
        "target": "passed",
        "target_rule": "passed = yes when final grade G3 >= 10 else no",
        "removed_leakage_columns": leakage_columns,
        "class_counts": model_df["passed"].value_counts().to_dict(),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Rows: {metadata['rows']}, columns: {metadata['columns']}")
    print(f"Class counts: {metadata['class_counts']}")


def read_subject(zip_path: Path, filename: str, course: str) -> pd.DataFrame:
    with ZipFile(zip_path) as archive:
        with archive.open(filename) as csv_file:
            df = pd.read_csv(csv_file)
    df.insert(0, "course", course)
    return df


if __name__ == "__main__":
    main()
