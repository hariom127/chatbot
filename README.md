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

Greetings, thanks, goodbyes, "how are you", "what's up", "what can you do", short acknowledgements ("ok", "got it") and similar small talk get an instant friendly reply that names your project. Typos and chat shorthand work too ("helo ther", "thnks a lott", "hw r u doing"). These replies never call the answer model. Any real question, including one that starts with "Hi", is answered from the documents as usual, with its original text.

Each message goes through these steps. The first one that decides wins, and when in doubt the message goes to RAG.

1. **Rules.** The whole message must be known phrases ("hi there", "thanks, bye!").
2. **Small-talk knowledge base**, only for short messages (up to 60 characters and `SMALL_TALK_MAX_WORDS` words, no digits):
   - **Fuzzy matching.** Stretched letters are squeezed ("hiii"), shorthand is expanded ("u", "r", "thx") and typos are fixed toward small-talk words. The rules run again on the result, then it is compared letter by letter with the examples and with real-question anchors.
   - **Semantic matching.** The message is compared with the same examples and anchors in a separate ChromaDB collection (`small_talk_*`), using the retrieval embedding model. Synced documents and `/sync/status` never see it.
3. **RAG**, unchanged.
4. **Safety net.** If RAG finds nothing, a message that was close to small talk still gets a small-talk reply. A question that starts with a greeting gets a not-found answer that greets back. Everything else gets the usual "I don't have enough information" answer.

| Setting | Default | Meaning |
|---|---|---|
| `PROJECT_NAME` | empty | Name used in replies ("Ask me anything about Acme Portal."). Empty means replies say "this project". |
| `CONVERSATIONAL_REPLIES_ENABLED` | `true` | Master switch for all of the above. `false` sends every message to RAG, exactly as before. |
| `SMALL_TALK_KB_ENABLED` | `true` | Fuzzy and semantic matching. The rules keep working when it is off. |
| `SMALL_TALK_SEMANTIC_ENABLED` | `true` | Semantic matching only. Fuzzy matching keeps working, and nothing is embedded. |
| `SMALL_TALK_SAFETY_NET_ENABLED` | `true` | The safety net after RAG. |
| `SMALL_TALK_MAX_WORDS` | `8` | Longer messages skip the knowledge base. |
| `SMALL_TALK_FUZZY_THRESHOLD` | `0.85` | Minimum letter-by-letter similarity to a small-talk example. |
| `SMALL_TALK_FUZZY_MARGIN` | `0.15` | How far it must score above the closest real-question anchor. |
| `SMALL_TALK_SEMANTIC_THRESHOLD` | `0.75` | Minimum embedding similarity to a small-talk example. Calibrated for `all-MiniLM-L6-v2`. |
| `SMALL_TALK_SEMANTIC_MARGIN` | `0.10` | How far it must score above the closest real-question anchor. |
| `ABUSIVE_LANGUAGE_FILTER_ENABLED` | `true` | Abusive messages get a polite reply that names `PROJECT_NAME`. |

Abusive messages ("you are useless", "f*ck off") get a polite reply without searching the documents or calling any model. If the message also asks a real question ("You stupid bot, what is an invoice?"), the abuse is removed and the question is answered as usual. Insults count only when aimed at the assistant, and words in quotes or in a question about the word itself ("What does 'idiot' mean?") are left alone. The rules are in `backend/app/services/moderation.py`.

Restart the backend after changing these settings.

#### Editing the small-talk replies and examples

Everything the bot says in small talk lives in `backend/app/small_talk/small_talk_kb.json`. Each entry has example messages (8 to 15, informal spelling, no digits, no project name) and 3 to 4 replies. The `question_anchors` list holds real questions that must never be mistaken for small talk; add a few that match your own documents. Replies may use `{project}`, only right after "about", "with" or "for".

After editing, run the tests (they check the file) and restart the backend. The ChromaDB collection rebuilds itself on the next short message. If the file is invalid, the backend logs one error and keeps answering with the phrase rules only.

#### Calibrating for an embedding model

The semantic values depend on the embedding model. After changing `EMBEDDING_MODEL` or `OPENAI_EMBEDDING_MODEL`, or after a big edit to the knowledge base, run:

```bash
cd backend && source .venv/bin/activate
python -m app.scripts.calibrate_small_talk
```

It uses a temporary ChromaDB folder, prints the scores and decision for each probe message, lists false positives and misses, and sweeps threshold and margin values. Pick the lowest values with zero false positives that still send the "tricky" probes to RAG. A miss is usually caught by the safety net, but a false positive would swallow a real question.

With OpenAI embeddings, a short message that the rules and fuzzy matching miss costs one extra small embedding request, plus one batch whenever the collection is rebuilt.

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
- **Answers always say "I don't have enough information"** → sync first; if still failing, raise `MAX_DISTANCE` (e.g. 0.9) in `.env`.
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



cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
