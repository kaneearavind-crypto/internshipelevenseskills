"""Persistent knowledge-base ingestion and scheduled/manual re-indexing."""
import json, os, re
from datetime import datetime, timezone
from pathlib import Path
import requests
import feedparser
from bs4 import BeautifulSoup
from shared.retrieval import SemanticIndex

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "knowledge_records.jsonl"
INDEX_META = ROOT / "data" / "knowledge_index_meta.json"

def load_records():
    if not STORE.exists(): return []
    rows=[]
    for line in STORE.read_text(encoding="utf-8").splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: continue
    return rows

def _save(record):
    STORE.parent.mkdir(parents=True, exist_ok=True)
    with STORE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def add_text_source(title, text, source="manual"):
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if not clean: return 0
    # Chunk long content to improve retrieval precision.
    words=clean.split()
    chunks=[" ".join(words[i:i+220]) for i in range(0,len(words),190)]
    for n, chunk in enumerate(chunks):
        _save({"title": title if len(chunks)==1 else f"{title} (part {n+1})",
               "text": chunk, "source": source, "updated_at": datetime.now(timezone.utc).isoformat()})
    return len(chunks)

def add_url_source(url):
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS public URLs are accepted.")
    response=requests.get(url, timeout=20, headers={"User-Agent":"KnowledgeBot/1.0"})
    response.raise_for_status()
    soup=BeautifulSoup(response.text,"html.parser")
    for tag in soup(["script","style","nav","footer","header"]): tag.decompose()
    title=soup.title.get_text(" ",strip=True) if soup.title else url
    text=soup.get_text(" ",strip=True)
    return add_text_source(title,text,url)

def add_rss_source(url):
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS RSS/Atom URLs are accepted.")
    feed=feedparser.parse(url)
    if not feed.entries: raise ValueError("No feed entries found.")
    count=0
    for entry in feed.entries[:50]:
        summary=BeautifulSoup(entry.get("summary",""),"html.parser").get_text(" ",strip=True)
        body=entry.get("title","") + ". " + summary
        count += add_text_source(entry.get("title","Feed item"),body,entry.get("link",url))
    return count

def rebuild_index():
    rows=load_records()
    if not rows: return 0
    idx=SemanticIndex()
    idx.build([r["title"] + "\n" + r["text"] for r in rows])
    import faiss
    path=ROOT/"data"/"knowledge.faiss"
    faiss.write_index(idx.index,str(path))
    INDEX_META.write_text(json.dumps({"count":len(rows),"built_at":datetime.now(timezone.utc).isoformat(),
                                     "model":"sentence-transformers/all-MiniLM-L6-v2"},indent=2),encoding="utf-8")
    return len(rows)

def search_records(query, top_k=5):
    rows=load_records()
    index_path=ROOT/"data"/"knowledge.faiss"
    if not rows or not index_path.exists(): return []
    from sentence_transformers import SentenceTransformer
    import faiss, numpy as np
    model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index=faiss.read_index(str(index_path))
    vector=model.encode([query],normalize_embeddings=True,convert_to_numpy=True).astype("float32")
    scores, ids=index.search(vector,min(top_k,index.ntotal))
    return [{"title":rows[int(i)]["title"],"text":rows[int(i)]["text"],"source":rows[int(i)]["source"],
             "updated_at":rows[int(i)]["updated_at"],"score":float(s)}
            for s,i in zip(scores[0],ids[0]) if i>=0]
