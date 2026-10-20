# Web Topic Classifier UI

A local webpage that sends a URL to the saved WebOrganizer topic model and shows the single highest-scoring category.

## Setup

Copy these files into your `ChadBlock` project folder, beside `weborganizer_topic_model.joblib`. The model file is created by `train_weborganizer.py` and is not included here.

Install the app dependencies in your project's virtual environment:

```bash
pip install -r requirements.txt
```

Start the local backend:

```bash
python app.py
```

Open this address in your browser:

```text
http://127.0.0.1:5000
```

The server binds to localhost. It fetches public HTTP/HTTPS pages, extracts visible page text, and feeds the same `url ... page ...` text format used by the training script into the saved hashing-vectorizer/SGD model.
