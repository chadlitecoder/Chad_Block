This is a prototype right now, it runs locally and it can classify sites into different categories.

## Setup

Keep the app entry points (`app.py`, `trainv2.py`, and `predictionv2.py`), `requirements.txt`, and `modelv2.joblib` in the project root. Supporting files are organized like this:

```text
ChadBlock/
├── app.py
├── modelv2.joblib
├── requirements.txt
├── trainv2.py
├── predictionv2.py
├── catalog/
│   ├── categories.json
│   └── popular_sites.json
├── static/
│   └── index.html
└── storage/                 # created/updated by the app; ignored by Git
    ├── custom_categories.json
    └── prediction_cache.json
```

The model file is produced by the training script and is not included in source control.

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

The lookup order is:

1. A user-saved category for the domain.
2. The bundled `catalog/popular_sites.json` catalog.
3. A previous model result in `storage/prediction_cache.json`.
4. The model, only when the domain has no saved match.

`catalog/popular_sites.json` is a curated starter list, not an exhaustive or live ranking of every popular site. Its categories use the model's existing topics where possible and add `Social Media` and `Search & Portals`. Add more entries to its `sites` object as needed. The prediction cache and custom categories are ignored by Git because they are local user data.

For an uncached domain, the server fetches a public HTTP/HTTPS page, extracts visible page text, and feeds it to the saved model. A category chosen in the optional category control overrides catalog and model results for that domain on later visits.
