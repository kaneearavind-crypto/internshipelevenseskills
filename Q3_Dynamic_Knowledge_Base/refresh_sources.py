"""Scheduled Q3 refresh runner. Configure HTTPS URLs in data/sources.json."""
import json
from pathlib import Path
from modules.dynamic_kb import add_url_source, add_rss_source, rebuild_index

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"data"/"sources.json"
if not CONFIG.exists():
    raise SystemExit("Create data/sources.json with {'web_pages': [], 'feeds': []}.")
config=json.loads(CONFIG.read_text(encoding="utf-8"))
total=0
for url in config.get("web_pages",[]):
    total += add_url_source(url)
for url in config.get("feeds",[]):
    total += add_rss_source(url)
print(f"Added {total} records; indexed {rebuild_index()} total records.")
