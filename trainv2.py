"""
Train a web-topic classifier on WebOrganizer's Topic Annotations dataset.

Model:
    HashingVectorizer + SGDClassifier using logistic loss.

Why this model:
    The dataset has about one million long web pages. HashingVectorizer
    creates a fixed-size sparse feature representation without keeping a
    vocabulary in memory. SGDClassifier can learn from batches, so the
    entire dataset does not need to be loaded into RAM.

The target is `top_choice`, the single topic assigned to each page.
"""

from itertools import islice
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score


# Hugging Face dataset repository.
DATASET_ID = "WebOrganizer/TopicAnnotations-Llama-3.1-8B"

# Training options.
RANDOM_SEED = 42
BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 20_000

# Web pages can contain hundreds of thousands of characters. Using every
# character is slow and often unnecessary for topic classification, so this
# script keeps the first 12,000 characters of page text.
MAX_TEXT_CHARACTERS = 12_000

# Read an initial sample to discover the label names before incremental
# training starts. The dataset has 24 topics; this sample should contain them.
LABEL_DISCOVERY_ROWS = 50_000

# The saved model bundle will be written to this file.
MODEL_PATH = Path("weborganizer_topic_model.joblib")


def clean_label(value):
    """
    Convert a dataset topic string into a consistent label.

    Some topic strings may include a descriptive suffix such as
    " - Includes: ...". Keep the topic name and remove that explanation.
    """
    if value is None:
        return ""

    label = str(value).strip()
    label = label.split(" - Includes:", maxsplit=1)[0].strip()
    return label


def make_model_text(row):
    """
    Combine page text and URL into one input string.

    Including the URL helps the model learn topic hints from domains and
    paths, while the page text provides the main semantic evidence.
    """
    url = str(row.get("url") or "").strip()
    text = str(row.get("text") or "").strip()

    # Cap very long documents to keep processing time and memory reasonable.
    text = text[:MAX_TEXT_CHARACTERS]

    return f"url {url} page {text}"


def batches(dataset, batch_size):
    """Yield lists of dataset rows, one batch at a time."""
    iterator = iter(dataset)

    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def vectorize_rows(rows, vectorizer):
    """Turn a batch of rows into sparse numeric features."""
    texts = [make_model_text(row) for row in rows]
    return vectorizer.transform(texts)


def discover_labels(train_stream):
    """
    Discover topic names from an initial slice of the streamed training data.

    SGDClassifier needs the complete list of possible classes on its first
    partial_fit call. Sampling the first 50,000 training rows avoids loading
    the full 3 GB dataset into memory just to find label names.
    """
    labels = set()

    for row in islice(iter(train_stream), LABEL_DISCOVERY_ROWS):
        label = clean_label(row.get("top_choice"))
        if label:
            labels.add(label)

    if len(labels) < 2:
        raise RuntimeError(
            f"Only found {len(labels)} labels in the initial scan. "
            "Check that the dataset loaded correctly."
        )

    return sorted(labels)


def evaluate_split(split_name, split_stream, vectorizer, classifier, labels):
    """Evaluate the model on a streamed validation or test split."""
    true_ids = []
    predicted_ids = []
    row_count = 0

    for rows in batches(split_stream, BATCH_SIZE):
        # Convert topic names into the integer IDs used during training.
        batch_labels = [clean_label(row.get("top_choice")) for row in rows]

        # A label absent from the discovery sample means the discovery window
        # was too small. Stop with a clear message rather than mislabel data.
        unknown = sorted(set(batch_labels) - set(labels))
        if unknown:
            raise RuntimeError(
                f"Found labels not seen during label discovery: {unknown}. "
                "Increase LABEL_DISCOVERY_ROWS and run again."
            )

        y_true = np.asarray([labels.index(label) for label in batch_labels])
        X = vectorize_rows(rows, vectorizer)
        y_pred = classifier.predict(X)

        true_ids.extend(y_true.tolist())
        predicted_ids.extend(y_pred.tolist())
        row_count += len(rows)

    if not true_ids:
        raise RuntimeError(f"The {split_name} split contained no rows.")

    print(f"\n{split_name} results ({row_count:,} pages)")
    print(
        "Micro-F1:",
        f"{f1_score(true_ids, predicted_ids, average='micro', zero_division=0):.4f}",
    )
    print(
        "Macro-F1:",
        f"{f1_score(true_ids, predicted_ids, average='macro', zero_division=0):.4f}",
    )
    print(
        classification_report(
            true_ids,
            predicted_ids,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
        )
    )


def main():
    print(f"Opening {DATASET_ID} in streaming mode...")
    dataset = load_dataset(DATASET_ID, streaming=True)

    # This repository publishes train, validation, and test splits.
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(dataset.keys())
    if missing_splits:
        raise RuntimeError(
            f"Dataset is missing expected splits: {sorted(missing_splits)}. "
            f"Available splits: {list(dataset.keys())}"
        )

    train_stream = dataset["train"]

    print(f"Finding topic labels from the first {LABEL_DISCOVERY_ROWS:,} rows...")
    labels = discover_labels(train_stream)
    label_to_id = {label: index for index, label in enumerate(labels)}

    print(f"Found {len(labels)} topic labels:")
    for label in labels:
        print(f"  - {label}")

    # HashingVectorizer uses a fixed feature space, so it does not need to
    # learn or store a vocabulary from the million documents.
    vectorizer = HashingVectorizer(
        n_features=2**20,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        dtype=np.float32,
    )

    # Logistic loss makes the classifier probabilistic and supports
    # incremental learning with partial_fit.
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        random_state=RANDOM_SEED,
        average=True,
    )

    # Shuffling with a bounded buffer mixes nearby rows without loading the
    # entire dataset into memory.
    shuffled_train = train_stream.shuffle(
        seed=RANDOM_SEED,
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )

    print("Training in batches...")
    trained = False
    seen_rows = 0

    for batch_number, rows in enumerate(batches(shuffled_train, BATCH_SIZE), start=1):
        batch_labels = [clean_label(row.get("top_choice")) for row in rows]

        unknown = sorted(set(batch_labels) - set(label_to_id))
        if unknown:
            raise RuntimeError(
                f"Found labels not seen during label discovery: {unknown}. "
                "Increase LABEL_DISCOVERY_ROWS and run again."
            )

        y = np.asarray([label_to_id[label] for label in batch_labels])
        X = vectorize_rows(rows, vectorizer)

        # The first partial_fit call must receive the complete class list.
        if not trained:
            classifier.partial_fit(X, y, classes=np.arange(len(labels)))
            trained = True
        else:
            classifier.partial_fit(X, y)

        seen_rows += len(rows)

        if batch_number % 100 == 0:
            print(f"  Trained on {seen_rows:,} pages...")

    if not trained:
        raise RuntimeError("No training rows were found.")

    print(f"Finished training on {seen_rows:,} pages.")

    # Evaluate using the dataset's held-out validation and test partitions.
    evaluate_split(
        "Validation",
        dataset["validation"],
        vectorizer,
        classifier,
        labels,
    )
    evaluate_split(
        "Test",
        dataset["test"],
        vectorizer,
        classifier,
        labels,
    )

    # Save both components and the settings needed by a prediction script.
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "classes": labels,
            "max_text_characters": MAX_TEXT_CHARACTERS,
            "dataset_id": DATASET_ID,
        },
        MODEL_PATH,
    )

    print(f"Saved model to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()