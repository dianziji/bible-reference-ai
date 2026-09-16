# Bible Reference AI

A retrieval-augmented scripture assistant. Ask a question in plain language and get an answer grounded in the King James Bible, with the exact verses it was drawn from returned alongside it.

**Stack:** FastAPI · LangChain · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`, `omni-moderation-latest`) · Pinecone · Next.js 16 · Ragas · Docker · Kubernetes

## How it works

```
                 ┌──────────────────────── ingestion (one-off) ────────────────────────┐
                 │  KJV JSON (66 books, 1,189 chapters, 31,000+ verses)                 │
                 │      └─▶ scripts/load_and_embed_kjv.py                              │
                 │            batches of 100 ─▶ text-embedding-3-small (1536-d)        │
                 │            ─▶ Pinecone index, cosine similarity                     │
                 └──────────────────────────────────────────────────────────────────────┘

  browser ─▶ Next.js /api/query (server route, proxies to backend)
                 │
                 ▼
          FastAPI POST /api/query { question }
                 │
                 ├─ 1. moderation gate      OpenAI omni-moderation-latest; flagged input is rejected
                 ├─ 2. retrieve             LangChain retriever over Pinecone, k=5, score_threshold=0.5
                 ├─ 3. generate             gpt-4o-mini, prompted to answer only from the retrieved verses
                 │                          and to say "not sure" when the context is insufficient
                 └─ 4. respond              { answer, verses: [{ book, chapter, verse, text }] }
```

The retrieved verses are returned as structured data, not just quoted in prose, so the UI can render them as citations and the user can check every claim against the text.

## Evaluation

`scripts/eval_with_ragas.py` runs a question set through the pipeline and scores each answer with [Ragas](https://docs.ragas.io) on **faithfulness** (is the answer supported by the retrieved verses?) and **answer relevancy** (does it address the question?). Results are written to `backend/eval_results/ragas_eval_results.csv` and were reviewed by hand alongside the automated scores.

## Repository layout

```
backend/
  app/main.py                    FastAPI app: GET /health, POST /api/query, CORS
  app/services/rag_service.py    RAGService: retriever + prompt + generation
  app/services/openai_client.py  embeddings, chat completion, moderation helpers
  app/services/pinecone_client.py index bootstrap (1536-d, cosine) and access
  scripts/init_pinecone.py       create the index
  scripts/load_and_embed_kjv.py  batch-embed the KJV and upsert to Pinecone
  scripts/eval_with_ragas.py     Ragas evaluation run
  data/bible-json/               KJV source data (see Data below)
  eval_results/                  Ragas output
  Dockerfile
frontend/
  app/page.tsx                   chat UI
  app/api/query/route.ts         server-side proxy to the backend (backend URL never reaches the client)
  Dockerfile
k8s/
  namespace, configmap, secret.example, backend (2 replicas), frontend, redis, ingress
kind-config.yaml                 local kind cluster for testing the manifests
```

## Running locally

**Backend**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT
python -m scripts.init_pinecone
python -m scripts.load_and_embed_kjv   # one-off, embeds ~31k verses
uvicorn app.main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

**Kubernetes**

```bash
kind create cluster --config kind-config.yaml
cp k8s/secret.example.yaml k8s/secret.yaml   # fill in real values; secret.yaml is gitignored
kubectl apply -f k8s/
```

The ingress routes `/` to the frontend and `/api` to the backend. Tunables (`TOP_K`, `SCORE_THRESHOLD`, Pinecone index/env, backend URL) live in `k8s/configmap.yaml`.

## Data

Scripture text is the King James Version from [kenyonbowers/BibleJSON](https://github.com/kenyonbowers/BibleJSON) (MIT), vendored under `backend/data/bible-json/` with its license and README intact.

## Notes

- Secrets are read from environment variables only. `backend/.env` and `k8s/secret.yaml` are gitignored; the `.example` files show the expected keys.
- The prompt instructs the model to decline rather than speculate when retrieval is weak; the `score_threshold` is what decides "weak".
- Fallback messages (moderation refusal, no verses found) are bilingual English / 中文 and live in `app/services/messages.py`.
