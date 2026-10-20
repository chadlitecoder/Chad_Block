import argparse
import json
import re

import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin


MODEL_PATH = "artifacts/model.joblib"


# Needed because the saved model references this class from the training script.
class TextColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column].fillna("").astype(str).tolist()


def get_page_info(url):
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; WebClassifier/1.0)"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.content[:2_000_000], "html.parser")

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    description = ""
    meta = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if not meta:
        meta = soup.find(
            "meta",
            attrs={"property": re.compile(r"^og:description$", re.I)},
        )
    if meta:
        description = meta.get("content", "").strip()

    html_tag = soup.find("html")
    language = html_tag.get("lang", "") if html_tag else ""

    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    page_text = " ".join(soup.stripped_strings)
    snippet = f"{description} {page_text}".strip()[:5000]

    return {
        "url": url,
        "title": title,
        "snippet": snippet,
        "language": language,
        "meta_description": description,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classify a webpage using its URL."
    )
    parser.add_argument("url", help="Webpage URL, such as https://example.com")
    args = parser.parse_args()

    if not args.url.startswith(("http://", "https://")):
        parser.error("URL must start with http:// or https://")

    bundle = joblib.load(MODEL_PATH)
    page = get_page_info(args.url)

    text_fields = bundle.get(
        "text_fields",
        ["url", "title", "snippet", "language", "meta_description"],
    )
    page["combined"] = " ".join(
        f"{field} {page.get(field, '')}" for field in text_fields
    )

    model_input = pd.DataFrame([page])
    probabilities = bundle["model"].predict_proba(model_input)[0]
    classes = bundle["classes"]
    threshold = bundle.get("threshold", 0.5)

    predictions = [
        {"category": label, "score": round(float(score), 4)}
        for label, score in zip(classes, probabilities)
        if score >= threshold
    ]

    if not predictions and len(probabilities):
        best = int(probabilities.argmax())
        predictions = [{
            "category": classes[best],
            "score": round(float(probabilities[best]), 4),
        }]

    predictions.sort(key=lambda item: item["score"], reverse=True)

    print(json.dumps({
        "url": args.url,
        "title": page["title"],
        "language": page["language"],
        "predictions": predictions,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
    