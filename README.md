# Drive Chatbot

A chatbot that runs on your machine and answers questions from the Google Docs in one Google Drive folder, using Retrieval-Augmented Generation (RAG).

```
Google Drive folder ──(read-only API)──▶ backend: chunk ─▶ local embeddings ─▶ ChromaDB
React chat UI ─▶ POST /chat ─▶ embed question ─▶ search ChromaDB ─▶ LLM (only sees retrieved text) ─▶ answer + sources
```

- **Frontend:** React + Vite (`frontend/`)
- **Backend:** Python + FastAPI (`backend/`)
- **Embeddings:** Sentence Transformers, run locally (`all-MiniLM-L6-v2`). Free, no API key.
- **Vector DB:** ChromaDB, stored locally in `backend/data/`
- **Answer model:** any OpenAI-compatible API, chosen in `backend/.env` (default: the Modal-hosted Qwen endpoint)

## Prerequisites

- Python 3.10+ and Node.js 18+
- A Google account that can read the Drive folder
- Credentials for one LLM provider (see below)

## 1. Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU-only PyTorch (much smaller download; fine for this app)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env     # then edit .env
```

## 2. Google Cloud / OAuth setup (one time)

The app only needs **read-only** access to Drive. It exports each Google Doc as plain text through the Drive API.

1. Go to https://console.cloud.google.com/ and create a project (e.g. "drive-chatbot").
2. **APIs & Services → Library**: enable **Google Drive API**. (Enabling Google Docs API is harmless but this app does not call it.)
3. **APIs & Services → OAuth consent screen**: choose **External**, fill in the app name and your email, and under **Test users** add your own Google account. Leaving it in "Testing" is fine for personal use (note: refresh tokens for apps in Testing expire after 7 days — just re-run step 6 if that happens).
4. **APIs & Services → Credentials → Create credentials → OAuth client ID**, application type **Desktop app**.
5. Download the JSON and save it as `backend/credentials.json` (git-ignored).
6. Get the folder ID: open the folder in Drive; the URL is `https://drive.google.com/drive/folders/<FOLDER_ID>`. Put it in `backend/.env` as `GOOGLE_DRIVE_FOLDER_ID`.
7. Authorize once (opens your browser):
   ```bash
   cd backend && source .venv/bin/activate
   python -m app.scripts.google_auth
   ```
   This writes `backend/token.json` (git-ignored, read-only scope).

Only Google Docs and Word (`.docx`) files directly inside that folder are indexed (not subfolders, Sheets, PDFs, etc.). Word files are downloaded and their paragraphs and table text extracted.

## 3. Choose an AI provider (`backend/.env`)

| `LLM_PROVIDER` | Needs | Notes |
|---|---|---|
| `modal` (default) | `MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET` | Your Modal-hosted Qwen endpoint. `LLM_MODEL` defaults to `Qwen/Qwen3.8-Max-VL-Thinking`. Reasoning effort is set to high; this can be slow. |
| `openai` | `OPENAI_API_KEY`, `LLM_MODEL` (e.g. `gpt-5.4-mini`) | Paid API. Reasoning models (`o*`, `gpt-5*`) are sent without a temperature. Older models get deprecated — if you see "not available (wrong name or deprecated)", pick a current model. |
| `groq` | `LLM_API_KEY`, `LLM_MODEL` | Has a free tier with rate limits that change — check Groq's current terms. |
| `gemini` | `LLM_API_KEY`, `LLM_MODEL` | Uses Google's OpenAI-compatible endpoint. Free-tier limits change — check Google's current terms. |
| `custom` | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Any OpenAI-compatible server (this includes Ollama at `http://localhost:11434/v1`, untested here). |

Switching providers only means editing `.env` and restarting; retrieval code is unchanged. Don't assume any provider is free forever — check their pricing.

### Embeddings

- **Local (default):** leave `OPENAI_EMBEDDING_MODEL` empty. Uses `EMBEDDING_MODEL` (Sentence Transformers), free and offline.
- **OpenAI:** set `OPENAI_EMBEDDING_MODEL=text-embedding-3-small` and `OPENAI_API_KEY`. Each sync of changed docs and every question costs a small amount of API usage.

The embedding model is independent of the answer model. Each embedding model gets its own ChromaDB collection, so after changing it just click **Sync documents** and everything is re-indexed. You may need to tune `MAX_DISTANCE` per embedding model.

### Greetings and small talk

Greetings, thanks, goodbyes, "how are you", "what can you do" and short acknowledgements ("ok", "got it") get an instant friendly reply that names your project. They never search the documents or call the answer model. Any real question is answered from the documents as usual. A leading greeting followed by punctuation is removed first, so "Hola, what's the weather?" is searched as "what's the weather?". Without that punctuation the message is left as it is, so content queries like "morning news headlines" keep their first word.

Two layers detect small talk. Exact phrase rules come first. If they don't match, a semantic fallback compares short messages (no digits, up to `SEMANTIC_SMALL_TALK_MAX_WORDS` words) with example phrases, using the same embedding model as retrieval. When in doubt, the message goes to RAG.

| Setting | Default | Meaning |
|---|---|---|
| `PROJECT_NAME` | empty | Name used in replies ("Ask me anything about Acme Portal."). Empty means replies say "this project". |
| `CONVERSATIONAL_REPLIES_ENABLED` | `true` | Master switch. `false` sends every message to RAG. |
| `SEMANTIC_SMALL_TALK_ENABLED` | `true` | Turns the semantic fallback on or off. The phrase rules keep working when it is off. |
| `SEMANTIC_SMALL_TALK_THRESHOLD` | `0.65` | Minimum similarity to a small-talk example. |
| `SEMANTIC_SMALL_TALK_MARGIN` | `0.10` | How far the message must score above the closest real-question example. |
| `SEMANTIC_SMALL_TALK_MAX_WORDS` | `6` | Longer messages skip the semantic fallback. |

Restart the backend after changing these settings. The threshold and margin were calibrated for `all-MiniLM-L6-v2` and `text-embedding-3-small`. After switching to another embedding model, recalibrate:

```bash
cd backend && source .venv/bin/activate
python -m app.scripts.calibrate_small_talk
```

The script prints scores for a set of probe messages and sweeps threshold and margin values. Pick the lowest values that give zero false positives and leave room above the "tricky" probes. A miss only means the message gets today's "could not find" answer, but a false positive would swallow a real question.

With OpenAI embeddings, a short message that the rules miss costs one extra small embedding request. The example phrases are embedded once per backend process.

## 4. Frontend setup

```bash
cd frontend
npm install
cp .env.example .env     # only holds the backend URL; never put keys here
```

## 5. Run

Terminal 1:
```bash
cd backend && source .venv/bin/activate
uvicorn app.main:app --reload
```
Terminal 2:
```bash
cd frontend && npm run dev
```
Open http://localhost:5173, click **Sync documents** (the first sync downloads the ~90 MB embedding model), then ask questions. Click Sync again whenever your docs change: only new/changed docs are re-embedded and deleted docs are removed.

Useful endpoints (http://localhost:8000/docs): `GET /health`, `GET /documents?include_text=true`, `POST /sync`, `GET /sync/status`, `POST /chat`.

## Tests

```bash
cd backend && source .venv/bin/activate
python -m pytest
```
Tests use temporary folders and mocks for Google Drive and the LLM. They do not need credentials. (The first run downloads the embedding model.)

## Troubleshooting

- **"Google is not authorized yet"** → run `python -m app.scripts.google_auth` from `backend/`.
- **"Permission denied" / "folder not found"** → check `GOOGLE_DRIVE_FOLDER_ID`, that Drive API is enabled, and that your account can open the folder.
- **Google login stopped working after a week** → OAuth apps in "Testing" expire refresh tokens after 7 days; re-run the auth script.
- **"OpenAI rejected the API key"** → `OPENAI_API_KEY=` must be followed directly by the key (`sk-...`), with no placeholder text, quotes or spaces.
- **"Modal credentials are missing"** → fill `MODAL_PROXY_TOKEN_ID` / `MODAL_PROXY_TOKEN_SECRET` in `backend/.env` and restart the backend.
- **Answers always say "could not find"** → sync first; if still failing, raise `MAX_DISTANCE` (e.g. 0.9) in `.env`.
- **Frontend says it cannot reach the backend** → make sure uvicorn runs on port 8000 and `CORS_ORIGINS` includes the frontend URL.
- **Chat is slow** → the high-reasoning Qwen model can take a while; raise `LLM_TIMEOUT_SECONDS` if requests time out.

## Known limitations

- Only top-level Google Docs and `.docx` files in one folder; images/screenshots, layout and comments are not indexed (text only).
- Sync is manual; there is no background scheduler.
- Chat has no memory between questions (each question is answered independently).
- No authentication on the local API — it is intended for `localhost` use only.
- Automated tests mock Google and the LLM; the real Google Drive and Modal flows were checked manually on one sample `.docx`; the browser UI was not tested automatically.

## Secrets

`.env` files, `credentials.json`, `token.json`, and the `backend/data/` ChromaDB folder are git-ignored. Never commit them.
