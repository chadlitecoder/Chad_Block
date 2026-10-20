import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import joblib
import requests
from bs4 import BeautifulSoup


# Look for the saved model beside this script.
MODEL_PATH = Path(__file__).resolve().parent / "modelv2.joblib"

# Avoid downloading extremely large web pages just to classify them.
MAX_DOWNLOAD_BYTES = 2_000_000


def fetch_page(url):
    """Download a page and return text in roughly the same form as training."""
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Enter a complete URL beginning with http:// or https://")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; WebTopicClassifier/1.0)"
    }

    # Stream the response so we can stop after a reasonable amount of data.
    chunks = []
    downloaded = 0

    with requests.get(
        url,
        headers=headers,
        timeout=25,
        stream=True,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue

            remaining = MAX_DOWNLOAD_BYTES - downloaded
            chunks.append(chunk[:remaining])
            downloaded += min(len(chunk), remaining)

            if downloaded >= MAX_DOWNLOAD_BYTES:
                break

        final_url = response.url
        content_type = response.headers.get("Content-Type", "")

    # Reject obvious non-HTML resources such as PDFs and images.
    if "html" not in content_type.lower():
        raise ValueError(
            f"The URL did not return an HTML page (Content-Type: {content_type or 'unknown'})."
        )

    html = b"".join(chunks)
    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that do not contain useful visible page content.
    for element in soup(["script", "style", "noscript", "svg", "iframe"]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    visible_text = " ".join(soup.stripped_strings)

    # Include the title explicitly, then cap the text to the same limit used
    # by the training script.
    page_text = f"{title} {visible_text}".strip()

    return final_url, page_text


def main():
    parser = argparse.ArgumentParser(
        description="Classify a webpage by supplying its URL."
    )
    parser.add_argument("url", help="URL of the webpage to classify")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of highest scoring topics to display (default: 5)",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find the trained model at {MODEL_PATH}. "
            "Run train_weborganizer.py first."
        )

    print("Loading trained model...")
    bundle = joblib.load(MODEL_PATH)

    print(f"Fetching page: {args.url}")
    final_url, page_text = fetch_page(args.url)

    # Match the feature format used in train_weborganizer.py.
    max_chars = bundle.get("max_text_characters", 12_000)
    model_text = f"url {final_url} page {page_text[:max_chars]}"

    # HashingVectorizer is stateless, so transform the new text directly.
    features = bundle["vectorizer"].transform([model_text])

    # The classifier was trained with log loss, so it provides class scores
    # that can be displayed as approximate probabilities.
    scores = bundle["classifier"].predict_proba(features)[0]
    classes = bundle["classes"]

    best_indices = scores.argsort()[::-1][:args.top_k]
    predictions = [
        {
            "topic": classes[index],
            "score": round(float(scores[index]), 4),
        }
        for index in best_indices
    ]

    print(json.dumps(
        {
            "url": final_url,
            "topics": predictions,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()