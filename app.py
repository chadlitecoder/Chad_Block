"""Local web UI backend for the saved WebOrganizer topic model."""

from __future__ import annotations

import ipaddress
import json
import socket
from pathlib import Path
from functools import lru_cache
from threading import RLock
from urllib.parse import urljoin, urlparse

import joblib
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelv2.joblib"
CUSTOM_LABELS_PATH = BASE_DIR / "custom_categories.json"
MAX_DOWNLOAD_BYTES = 2_000_000
MAX_REDIRECTS = 5
IDENTIFICATION_THRESHOLD = 0.25
CUSTOM_LABELS_LOCK = RLock()

app = Flask(__name__, static_folder=None)


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


def normalize_url(value: str) -> str:
    """Accept a bare domain and default it to HTTPS."""
    url = value.strip()
    if not url:
        raise ValueError("Enter a website domain or URL.")
    if "://" not in url:
        url = f"https://{url}"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a valid domain, such as example.com.")
    return url


def domain_for_url(url: str) -> str:
    """Return the host used to associate a saved category with a site."""
    hostname = urlparse(url).hostname
    if not hostname:
        raise ValueError("Could not determine the website domain.")
    return hostname.rstrip(".").lower()


def load_custom_labels() -> dict[str, str]:
    """Read user-saved domain labels, returning an empty map if none exist."""
    with CUSTOM_LABELS_LOCK:
        try:
            data = json.loads(CUSTOM_LABELS_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            app.logger.exception("Could not read saved custom categories")
            return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(domain).lower(): str(category)
        for domain, category in data.items()
        if isinstance(domain, str) and isinstance(category, str)
    }


def save_custom_label(domain: str, category: str) -> None:
    """Atomically persist one user-defined domain category."""
    with CUSTOM_LABELS_LOCK:
        labels = load_custom_labels()
        labels[domain] = category
        temporary_path = CUSTOM_LABELS_PATH.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(CUSTOM_LABELS_PATH)


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


@lru_cache(maxsize=1)
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

    try:
        url = normalize_url(url)
        final_url, title, page_text = fetch_page(url)
        bundle = get_model_bundle()
        max_chars = int(bundle.get("max_text_characters", 12_000))

        # Match the exact text shape used by train_weborganizer.py.
        model_text = f"url {final_url} page {page_text[:max_chars]}"
        features = bundle["vectorizer"].transform([model_text])
        scores = bundle["classifier"].predict_proba(features)[0]
        best_index = int(scores.argmax())
        probability = float(scores[best_index])
        candidate_category = str(bundle["classes"][best_index])
        identified = probability >= IDENTIFICATION_THRESHOLD
        domain = domain_for_url(final_url)
        saved_category = load_custom_labels().get(domain) if not identified else None
        if saved_category:
            result_category = saved_category
            label_source = "saved_custom"
        elif identified:
            result_category = candidate_category
            label_source = "model"
        else:
            result_category = "Not able to identify"
            label_source = "unknown"

        return jsonify({
            "url": final_url,
            "domain": domain,
            "title": title,
            "category": result_category,
            "candidate_category": candidate_category,
            "identified": identified,
            "saved_category": saved_category,
            "label_source": label_source,
            "probability": probability,
            "threshold": IDENTIFICATION_THRESHOLD,
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


@app.post("/api/labels")
def save_label():
    """Save a user-provided category for a domain for future low-score results."""
    body = request.get_json(silent=True) or {}
    try:
        url = normalize_url(str(body.get("url", "")))
        category = str(body.get("category", "")).strip()
        if not category:
            raise ValueError("Enter a category name.")
        if len(category) > 60 or any(ord(character) < 32 for character in category):
            raise ValueError("Use a category name of 60 characters or fewer.")

        domain = domain_for_url(url)
        save_custom_label(domain, category)
        return jsonify({"domain": domain, "category": category, "saved": True})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError as exc:
        app.logger.exception("Could not save custom category")
        return jsonify({"error": f"Could not save the category: {exc}"}), 500


if __name__ == "__main__":
    # Local-only server; open http://127.0.0.1:5000 in your browser.
    app.run(host="127.0.0.1", port=5000, debug=False)
