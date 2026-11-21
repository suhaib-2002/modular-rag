# Modular RAG (Router + Composable Modules)

A FastAPI backend built from **interchangeable modules** wired into **pipelines**. A **router** (an LLM call) picks the right pipeline for each question, so simple questions skip expensive steps and hard ones get the full flow.

## Core idea
Every module takes a shared state dict and returns it. A pipeline is an ordered list of module names:

```python
PIPELINES = {
    "direct":  ["generate_direct"],                                              # greetings, no retrieval
    "naive":   ["retrieve_dense", "top4", "generate"],                           # simple conceptual question
    "keyword": ["retrieve_keyword", "top4", "generate"],                         # exact term or acronym
    "full":    ["rewrite", "retrieve_hybrid", "rerank", "top4", "generate"],     # complex or follow-up question
}
```
To add a capability, write one function and add its name to a pipeline.

## Modules
| Module | Role |
|---|---|
| `rewrite` | Rewrites the question into a standalone search query, using chat history |
| `retrieve_dense` | Embedding search (ChromaDB) |
| `retrieve_keyword` | BM25 keyword search |
| `retrieve_hybrid` | Dense + BM25 fused with Reciprocal Rank Fusion |
| `rerank` | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the 20 candidates |
| `top4` | Keeps the 4 best chunks |
| `generate` | Grounded answer with `[n]` citations |
| `generate_direct` | Answers without retrieval |
| Router | LLM chooses the pipeline, falls back to `full` if its reply is unusable |
| Memory | Earlier turns sent in `history` are used by the router, `rewrite` and `generate` |

## Project files
```
modular_rag.py      the whole backend (indexing, modules, router, FastAPI)
requirements.txt    dependencies
COMMANDS.txt        step-by-step run commands
.env                your Groq API key and model settings
data/               documents to search (14 sample Wikipedia files included)
```

## Quick start
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# put your own Groq key in .env  (GROQ_API_KEY=...)
uvicorn modular_rag:app --reload
```
Open http://localhost:8000/docs, run `POST /ingest`, then `POST /ask`. Full commands are in [COMMANDS.txt](COMMANDS.txt). The first `full` request downloads the reranker model (about 90 MB).

## How to use the API
Open http://localhost:8000/docs, click an endpoint, then **Try it out** and **Execute**.

| Endpoint | What to enter | Result |
|---|---|---|
| `GET /health` | nothing | Indexed chunk count and the files in `data/` |
| `GET /pipelines` | nothing | Modules and pipelines |
| `POST /upload` | **Choose File**: a `.pdf`, `.txt` or `.md` | Saved to `data/`. Not searchable until `/ingest` |
| `POST /ingest` | nothing | Rebuilds the index from `data/`. Run first, and after every upload |
| `POST /ask` | the JSON below | `answer`, `pipeline`, `steps`, `reason`, `sources` |

```json
{"question": "What does the acronym BM25 mean?", "pipeline": "auto", "history": []}
```
`pipeline` is `auto` (router decides), `direct`, `naive`, `keyword` or `full`. For a follow-up question, put the earlier turns in `history`:
```json
{"question": "How could it be used with RAG?", "pipeline": "auto",
 "history": [{"role": "user", "content": "What is a knowledge graph?"},
             {"role": "assistant", "content": "A knowledge graph is a ..."}]}
```

**Suggested test order:** `/ingest`; ask "Hello there" (goes to `direct`), "What does the acronym BM25 mean?" (`keyword`), and a comparison question (`full`); then a follow-up with `history`; then `/upload` your own file, `/ingest`, and `/ask` about it.

## Using your own key
Edit `.env` and set `GROQ_API_KEY`. Never commit a real key to a public repository.

## Modular vs Hybrid RAG
Hybrid RAG is one fixed retrieval strategy (dense + BM25 + RRF). Modular RAG treats that as one block (`retrieve_hybrid`) among several and decides per question which blocks to run.

## Tech stack
Python 3.10+, FastAPI, sentence-transformers (embeddings and reranker), ChromaDB, rank-bm25, pypdf, Groq (`openai/gpt-oss-20b`).
