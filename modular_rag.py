"""
Modular RAG backend (FastAPI): interchangeable modules wired into pipelines, chosen by an LLM router.
Run:  uvicorn modular_rag:app --reload      then open http://localhost:8000/docs
"""
import json
import os
import re
from pathlib import Path

import chromadb
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DATA_DIR, INDEX_DIR = Path("data"), Path("index")
STOP = set("a an the and or of to in on is are was were do does did what which who how why this that it for with".split())

app = FastAPI(title="Modular RAG (Router + Modules)")
embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
chunks, by_id, bm25, col, reranker = [], {}, None, None, None


def tokenize(text):
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOP]


def llm(prompt, json_mode=False):
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key.startswith("your_"):
        raise HTTPException(500, "Set your GROQ_API_KEY in the .env file.")
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    for _ in range(3):  # the model sometimes returns an empty reply
        reply = Groq(api_key=key).chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], **extra)
        if reply.choices[0].message.content:
            return reply.choices[0].message.content
    return ""


# ---------- indexing ----------
def load_docs():
    docs = []
    for f in sorted(DATA_DIR.glob("*")):
        if f.suffix.lower() == ".pdf":
            docs.append((f.name, " ".join(p.extract_text() or "" for p in PdfReader(f).pages)))
        elif f.suffix.lower() in (".txt", ".md"):
            docs.append((f.name, f.read_text(encoding="utf-8", errors="ignore")))
    return docs


def split(text, size=800, overlap=150):
    text = re.sub(r"\s+", " ", text)
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]


def build_index():
    items = [{"id": f"{name}::{i}", "source": name, "text": c}
             for name, text in load_docs() for i, c in enumerate(split(text))]
    if not items:
        raise HTTPException(400, "No documents found in the data/ folder.")
    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(items), encoding="utf-8")

    client = chromadb.PersistentClient(str(INDEX_DIR / "chroma"))
    if "chunks" in [c.name for c in client.list_collections()]:
        client.delete_collection("chunks")
    store = client.create_collection("chunks", metadata={"hnsw:space": "cosine"})
    vectors = embedder.encode([c["text"] for c in items], normalize_embeddings=True).tolist()
    for i in range(0, len(items), 500):
        part = items[i:i + 500]
        store.add(ids=[c["id"] for c in part], documents=[c["text"] for c in part], embeddings=vectors[i:i + 500])
    load_index()
    return len(items)


def load_index():
    global chunks, by_id, bm25, col
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in chunks}
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    col = chromadb.PersistentClient(str(INDEX_DIR / "chroma")).get_collection("chunks")


# ---------- retrieval helpers ----------
def dense(query, n=20):
    vec = embedder.encode([query], normalize_embeddings=True).tolist()
    return col.query(query_embeddings=vec, n_results=min(n, len(chunks)))["ids"][0]


def keyword(query, n=20):
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
    return [chunks[i]["id"] for i in top if scores[i] > 0]


def rrf(rankings, k=60):
    score = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] = score.get(cid, 0) + 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)


# ---------- modules: each takes the shared state `s` and returns it ----------
def query_of(s):
    return s.get("query") or s["question"]


def chat_text(s, n=300):
    return "\n".join(f"{m['role']}: {m['content'][:n]}" for m in s["history"][-4:]) or "(none)"


def rewrite(s):
    s["query"] = llm("Rewrite the last question as one standalone search query. Use the chat to resolve "
                     f"words like 'it'. Return only the query.\n\nChat:\n{chat_text(s, 200)}\n\n"
                     f"Question: {s['question']}").strip()
    return s


def retrieve_dense(s):
    s["hits"] = dense(query_of(s))
    return s


def retrieve_keyword(s):
    s["hits"] = keyword(query_of(s))
    return s


def retrieve_hybrid(s):
    s["hits"] = rrf([dense(query_of(s)), keyword(query_of(s))])
    return s


def rerank(s):
    global reranker
    reranker = reranker or CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = reranker.predict([(query_of(s), by_id[i]["text"]) for i in s["hits"]])
    s["hits"] = [i for _, i in sorted(zip(scores, s["hits"]), key=lambda x: -x[0])]
    return s


def top4(s):
    s["hits"] = s["hits"][:4]
    return s


def generate(s):
    context = "\n\n".join(f"[{n}] {by_id[i]['text']}" for n, i in enumerate(s["hits"], 1))
    s["answer"] = llm("Answer using only the context below and cite sources like [1]. If the answer is not "
                      f"in the context, say you don't know.\n\nChat so far:\n{chat_text(s)}\n\n"
                      f"Context:\n{context}\n\nQuestion: {s['question']}")
    return s


def generate_direct(s):
    s["answer"] = llm(f"You are a friendly assistant for a document Q&A system. Reply briefly.\n\n{s['question']}")
    return s


MODULES = {f.__name__: f for f in [rewrite, retrieve_dense, retrieve_keyword, retrieve_hybrid,
                                   rerank, top4, generate, generate_direct]}

PIPELINES = {
    "direct": ["generate_direct"],
    "naive": ["retrieve_dense", "top4", "generate"],
    "keyword": ["retrieve_keyword", "top4", "generate"],
    "full": ["rewrite", "retrieve_hybrid", "rerank", "top4", "generate"],
}

ROUTES = {
    "direct": "greeting, thanks or small talk that needs no documents",
    "naive": "a simple conceptual question that is fully understandable on its own, without the chat",
    "keyword": "a lookup of a specific term, name or acronym",
    "full": "a complex, multi-part or comparison question, or ANY question that depends on the earlier chat (for example it uses words like it, that, they, this)",
}


def route(question, history):
    options = "\n".join(f'"{k}": {v}' for k, v in ROUTES.items())
    chat = "\n".join(f"{m['role']}: {m['content'][:150]}" for m in history[-4:]) or "(none)"
    prompt = (f"Pick the best pipeline.\n{options}\n\nChat:\n{chat}\n\nQuestion: {question}\n\n"
              'Reply as JSON: {"pipeline": "<name>", "reason": "<short reason>"}')
    try:
        out = json.loads(llm(prompt, json_mode=True))
        if out["pipeline"] in PIPELINES:
            return out["pipeline"], out.get("reason", "")
    except HTTPException:
        raise
    except Exception:
        pass
    return "full", "router fallback"


def run(question, history, pipeline):
    s = {"question": question, "history": history, "hits": [], "answer": ""}
    reason = "chosen by user"
    if pipeline == "auto":
        pipeline, reason = route(question, history)
    for name in PIPELINES[pipeline]:
        s = MODULES[name](s)
    s.update(pipeline=pipeline, reason=reason, steps=PIPELINES[pipeline])
    return s


# ---------- API ----------
class Question(BaseModel):
    question: str
    pipeline: str = "auto"  # auto | direct | naive | keyword | full
    history: list[dict] = []  # earlier turns: [{"role": "user", "content": "..."}, ...]


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(chunks), "documents": [f.name for f in DATA_DIR.glob("*")]}


@app.get("/pipelines")
def pipelines():
    return {"modules": list(MODULES), "pipelines": PIPELINES}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Save a .pdf/.txt/.md file into data/. Call /ingest afterwards to index it."""
    name = Path(file.filename).name
    if Path(name).suffix.lower() not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "Only .pdf, .txt and .md files are supported.")
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / name).write_bytes(file.file.read())
    return {"saved": name}


@app.post("/ingest")
def ingest():
    """(Re)build the dense and BM25 indexes from everything in data/."""
    return {"indexed_chunks": build_index()}


@app.post("/ask")
def ask(q: Question):
    if not chunks:
        raise HTTPException(400, "Index is empty. Call POST /ingest first.")
    if q.pipeline != "auto" and q.pipeline not in PIPELINES:
        raise HTTPException(400, "pipeline must be auto, direct, naive, keyword or full.")
    s = run(q.question, q.history, q.pipeline)
    return {"answer": s["answer"], "pipeline": s["pipeline"], "reason": s["reason"], "steps": s["steps"],
            "sources": [{"source": by_id[i]["source"], "text": by_id[i]["text"][:200]} for i in s["hits"]]}


if (INDEX_DIR / "chunks.json").exists():
    load_index()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
