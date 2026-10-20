import ast
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


DATA_PATH = Path(
    "/home/chad/Desktop/ChadBlock/data/cleaned_classified_data.parquet"
)
TEXT_FIELDS = ("url", "title", "snippet", "language", "meta_description")


class TextColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column].fillna("").astype(str).tolist()


def parse_labels(value):
    """Handle label lists, list-like strings, and comma-separated strings."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = list(value)
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    else:
        raw = str(value).strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            values = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
        except (ValueError, SyntaxError):
            values = re.split(r"\s*,\s*", raw)

    labels = []
    for item in values:
        label = str(item).strip().strip("\"'")
        if label and label not in labels:
            labels.append(label)
    return labels


def make_model():
    features = FeatureUnion([
        ("words", Pipeline([
            ("select", TextColumn("combined")),
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=250_000,
                sublinear_tf=True,
                strip_accents="unicode",
                dtype=np.float32,
            )),
        ])),
        ("url_characters", Pipeline([
            ("select", TextColumn("url")),
            ("tfidf", TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=150_000,
                sublinear_tf=True,
                dtype=np.float32,
            )),
        ])),
    ])

    return Pipeline([
        ("features", features),
        ("classifier", OneVsRestClassifier(
            LogisticRegression(C=4.0, solver="liblinear", max_iter=1000),
            n_jobs=1,
        )),
    ])


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    print(f"Loading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns:")
    print(", ".join(df.columns))

    target_column = next(
        (column for column in ("category", "category_list") if column in df.columns),
        None,
    )
    if target_column is None:
        raise ValueError(
            "Could not find a 'category' or 'category_list' target column. "
            f"Available columns: {list(df.columns)}"
        )

    for field in TEXT_FIELDS:
        if field not in df.columns:
            print(f"Column '{field}' is missing; filling it with empty text.")
            df[field] = ""

        df[field] = df[field].fillna("").astype(str)

    df["_labels"] = df[target_column].map(parse_labels)
    df = df[df["_labels"].map(bool)].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable labels found in '{target_column}'.")

    df["combined"] = df.apply(
        lambda row: " ".join(
            f"{field} {row[field]}" for field in TEXT_FIELDS
        ),
        axis=1,
    )

    label_binarizer = MultiLabelBinarizer()
    y = label_binarizer.fit_transform(df["_labels"])
    classes = list(label_binarizer.classes_)
    X = df[[*TEXT_FIELDS, "combined"]]

    # Make reproducible 70% train, 15% validation, 15% test splits.
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        shuffle=True,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    print(
        f"Rows: train={len(X_train):,}, validation={len(X_val):,}, "
        f"test={len(X_test):,}"
    )

    model = make_model()
    print("Training model...")
    model.fit(X_train, y_train)

    for split_name, split_X, split_y in (
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ):
        predictions = model.predict_proba(split_X) >= 0.5
        print(
            f"\n{split_name} micro-F1: "
            f"{f1_score(split_y, predictions, average='micro', zero_division=0):.4f}"
        )
        print(
            f"{split_name} macro-F1: "
            f"{f1_score(split_y, predictions, average='macro', zero_division=0):.4f}"
        )
        print(classification_report(
            split_y,
            predictions,
            target_names=classes,
            zero_division=0,
        ))

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "classes": classes,
            "threshold": 0.5,
            "text_fields": list(TEXT_FIELDS),
        },
        output_dir / "model.joblib",
    )

    print(f"Saved trained model to {output_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()