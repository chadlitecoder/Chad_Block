"""Local web UI backend for the saved WebOrganizer topic model."""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse

import joblib
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelv2.joblib"
MAX_DOWNLOAD_BYTES = 2_000_000
MAX_REDIRECTS = 5

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


def is_public_web_url(url: str) -> bool:
    """Allow HTTP(S) URLs that resolve only to public IP addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        return bool(addresses) and all(
            ipaddress.ip_address(item[4][0]).is_global for item in addresses
        )
    except (OSError, ValueError):
        return False


def fetch_page(url: str) -> tuple[str, str, str]:
    """Fetch a public HTML page, following only a few validated redirects."""
    current_url = url
    headers = {"User-Agent": "Mozilla/5.0 (compatible; WebTopicClassifier/1.0)"}

    for redirect_number in range(MAX_REDIRECTS + 1):
        if not is_public_web_url(current_url):
            raise ValueError("Enter a public website URL beginning with http:// or https://")

        with requests.get(
            current_url,
            headers=headers,
            timeout=20,
            stream=True,
            allow_redirects=False,
        ) as response:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location or redirect_number == MAX_REDIRECTS:
                    raise ValueError("The website redirected too many times.")
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and "html" not in content_type and "text/plain" not in content_type:
                raise ValueError("That URL did not return an HTML page.")

            chunks = []
            downloaded = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                keep = chunk[: MAX_DOWNLOAD_BYTES - downloaded]
                chunks.append(keep)
                downloaded += len(keep)
                if downloaded >= MAX_DOWNLOAD_BYTES:
                    break

            html = b"".join(chunks)
            encoding = response.encoding or "utf-8"
            soup = BeautifulSoup(html.decode(encoding, errors="replace"), "html.parser")

            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            for element in soup(["script", "style", "noscript", "svg", "iframe"]):
                element.decompose()
            visible_text = " ".join(soup.stripped_strings)
            return response.url, title, visible_text

    raise ValueError("Could not retrieve the webpage.")


def get_model_bundle():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Copy your trained "
            "weborganizer_topic_model.joblib file beside app.py."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    url = str(body.get("url", "")).strip()
    if not url:
        return jsonify({"error": "Enter a website URL."}), 400

    try:
        final_url, title, page_text = fetch_page(url)
        bundle = get_model_bundle()
        max_chars = int(bundle.get("max_text_characters", 12_000))

        # Match the exact text shape used by train_weborganizer.py.
        model_text = f"url {final_url} page {page_text[:max_chars]}"
        features = bundle["vectorizer"].transform([model_text])
        scores = bundle["classifier"].predict_proba(features)[0]
        best_index = int(scores.argmax())

        return jsonify({
            "url": final_url,
            "title": title,
            "category": bundle["classes"][best_index],
            "probability": float(scores[best_index]),
        })
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except requests.RequestException as exc:
        return jsonify({"error": f"Could not fetch that page: {exc}"}), 502
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    # Local-only server; open http://127.0.0.1:5000 in your browser.
    app.run(host="127.0.0.1", port=5000, debug=False)
