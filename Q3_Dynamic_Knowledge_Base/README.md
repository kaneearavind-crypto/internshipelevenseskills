# Q3 — Dynamic Knowledge-Base Expansion

Implementation files: `dynamic_kb.py`, `refresh_sources.py`, and `sources.example.json`.

**Features:** ingestion of trusted HTTPS pages and RSS/Atom feeds, manual text and CSV Q&A import, persistent JSONL records, and vector-index rebuilding/search.

Run through the root application: `streamlit run app.py` → **Q3 · Dynamic knowledge base**.

For scheduled refresh, copy `sources.example.json` to `data/sources.json`, add trusted feed URLs, then configure a scheduler/cron job to run `python scripts/refresh_sources.py`. The script is provided; a cloud scheduler still must be configured on the chosen hosting provider.
