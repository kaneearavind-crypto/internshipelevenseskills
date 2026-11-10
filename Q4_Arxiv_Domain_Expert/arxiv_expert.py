"""arXiv Kaggle CSV loading, CS filtering, retrieval, summaries and concept map."""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _first(row, names, default=""):
    for name in names:
        if name in row and pd.notna(row[name]): return str(row[name])
    return default

def load_arxiv_csv(file):
    df=pd.read_csv(file).fillna("")
    df.columns=[str(c).strip().lower() for c in df.columns]
    records=[]
    for _,row in df.iterrows():
        d=row.to_dict()
        title=_first(d,["title"])
        abstract=_first(d,["abstract","summary"])
        categories=_first(d,["categories","category","arxiv_category"])
        if not title or not abstract: continue
        # Keep Computer Science papers; arXiv categories commonly start with cs.
        if categories and not any(x.strip().startswith("cs.") for x in re.split(r"[ ,;]+",categories)):
            continue
        records.append({"id":_first(d,["id","paper_id"],str(len(records))),
                        "title":title,"abstract":abstract,"categories":categories})
    return pd.DataFrame(records).drop_duplicates(subset=["id","title"]).reset_index(drop=True)

def search_papers(papers, query, top_k=8):
    corpus=(papers["title"].astype(str)+" "+papers["abstract"].astype(str)).tolist()
    vectorizer=TfidfVectorizer(stop_words="english",max_features=100000,ngram_range=(1,2))
    matrix=vectorizer.fit_transform(corpus+[query])
    scores=cosine_similarity(matrix[-1],matrix[:-1]).ravel()
    order=scores.argsort()[::-1][:top_k]
    return [{"id":papers.iloc[i]["id"],"title":papers.iloc[i]["title"],
             "abstract":papers.iloc[i]["abstract"],"categories":papers.iloc[i]["categories"],
             "score":float(scores[i])} for i in order if scores[i]>0]

def summarize_abstract(text, max_sentences=3):
    sentences=re.split(r"(?<=[.!?])\s+",str(text).strip())
    return " ".join(sentences[:max_sentences]) if sentences else str(text)[:600]

def ollama_explain(question, context, model="llama3.2"):
    import requests
    prompt=("You are a research assistant. Answer only from the supplied research context. "
            "If evidence is insufficient, say so. Explain technical terms clearly.\n\n"
            f"Context:\n{context[:9000]}\n\nQuestion:\n{question}")
    try:
        response=requests.post("http://localhost:11434/api/generate",
            json={"model":model,"prompt":prompt,"stream":False},timeout=120)
        response.raise_for_status()
        return response.json().get("response","No response returned.")
    except Exception as exc:
        return ("Local Ollama is not available. Start Ollama and run `ollama pull llama3.2` "
                f"to enable generated explanations. Error: {exc}\n\nEvidence-based extractive context:\n{context[:1800]}")

def build_concept_graph(results):
    import re
    from collections import Counter
    terms=Counter()
    stop={"this","that","with","from","using","based","which","their","these","those","into","paper","method","model","approach","show"}
    for p in results:
        words=[w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b",p["title"]+" "+p["abstract"])]
        terms.update(w for w in words if w not in stop)
    top=[w for w,_ in terms.most_common(10)]
    if not top: return "graph G { label=\"No concepts available\"; }"
    nodes="; ".join(f'"{w}"' for w in top)
    edges="; ".join(f'"{a}" -- "{b}"' for a,b in zip(top,top[1:]))
    return f'graph G {{ rankdir=LR; node [shape=ellipse, style=filled, fillcolor="#e8f0fe"]; {nodes}; {edges}; }}'
