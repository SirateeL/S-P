# FinWise AI

A local Flask finance learning platform for university students. Includes a responsive dashboard, monthly budget planner, savings calculator, four-topic learning hub, and PDF-grounded AI with page citations. No paid API or cloud account is required.

## macOS setup

Use **Python 3.11 or 3.12** (recommended for ChromaDB compatibility). Install Python from python.org or, with Homebrew, `brew install python@3.12`. Install Ollama from [ollama.com](https://ollama.com/download/mac), then open the Ollama application. Alternatively run `ollama serve` in a separate terminal if it is not already running.

In this project folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2:3b
python app.py
```

Open **http://127.0.0.1:5001**. Keep the terminal running; Ctrl+C stops Flask. The initial package/model downloads require internet; subsequent use runs locally. Model downloads may take several gigabytes. If an existing `.venv` was created with another Python version, remove that project-local environment before recreating it.

Test installation:

```sh
source .venv/bin/activate
python -m pytest -q
```

Tests use a real generated PDF and local Chroma database in temporary directories, with mocked embeddings and generation. They do not download models or require a running Ollama server.

## Use

1. Enter your take-home monthly income and category expenses in **Budget planner**, then save. The dashboard updates with a breakdown and balance. Values start empty/zero, not invented sample finances.
2. Set a target, current savings, and future date in **Savings goals**. The app estimates monthly contributions using days until the target divided by 30.4375, rounded up to whole contribution months (minimum one). It rounds the contribution up to cents. Dates more than 50 years away are rejected; expired saved goals prompt an update. Contributions exceeding the budget balance are flagged.
3. In **Finance AI expert**, upload a text-based finance PDF. Wait for indexing, then ask a question. Expand each source to read the exact retrieved text; open the PDF at the cited page. You can remove documents from the library.
4. Explore the **Learning hub** for separately labeled general introductory lessons. These lessons do not claim support from uploaded documents.

The interface uses USD for consistency. Budget and goal data are stored in this browser's localStorage, without an account. Clearing site data removes them. PDFs and embeddings are stored in `data/`; deleting that directory while the app is stopped resets the library. Browser storage and the PDF library are separate.

## How document-grounded answers work

PyMuPDF extracts text one page at a time. Text is normalized and chunked into up to 1,100 characters with approximately 180 characters of overlap, never merging pages. Each chunk records its sanitized filename, document ID and original page number. Ollama `nomic-embed-text` produces vectors stored in persistent ChromaDB using cosine distance. Questions use the same embedding model. Up to 20 semantic candidates are combined with BM25 keyword candidates using reciprocal rank fusion. Titles and contents slides are deprioritized; up to eight passages are sent to the model. A fixed cosine cutoff is not used as an evidence verdict. Existing embeddings remain unchanged.

The system prompt restricts answers to PDF evidence and prohibits personalized investment recommendations. Ollama receives a JSON schema for paragraphs, each containing explanatory text, one retrieved source ID, and an exact supporting quote. The backend validates the structure, ID, and quote against that specific retrieved passage before rendering a citation. It retries one failed generation with the validation error and previous output; two failures return a clear error with no attached answer or sources. Empty evidence returns an insufficient-evidence explanation. Displayed pages, excerpts and PDF URLs come only from persisted passage metadata. Supporting quotes appear in the source panel.

**Limits:** citation validation proves a source exists, not that every generated statement is entailed by it. Local models can misinterpret passages or ignore instructions, so check the supporting text. Hybrid retrieval ranks candidates; it does not establish answerability. Matching a quote does not prove that a paraphrase accurately interprets it. This is retrieval-augmented generation, not model fine-tuning. Questions are independent; previous chat turns are not sent as context. The learning content and AI are educational, not personalized financial or investment advice.

## Configuration and troubleshooting

Optional environment variables (the app does not automatically load `.env` files):

```sh
export OLLAMA_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3.2:3b
export EMBEDDING_MODEL=nomic-embed-text
export PORT=5001
# Optional: store runtime files elsewhere
# export FINWISE_DATA_DIR=/absolute/path/to/finwise-data
python app.py
```

- **Ollama offline / missing model:** open Ollama, run `ollama list`, and pull both models. The header reports readiness. Budget, savings and lessons work without Ollama.
- **Model too slow:** use a smaller compatible Ollama chat model and set `OLLAMA_MODEL`. Uploads embed in batches; generation requests allow up to 180 seconds each. Multiple embedding batches can make a large upload take longer.
- **Embedding model changed:** remove uploaded documents and re-upload them (or reset `data/` with Flask stopped). Embeddings from different models must not share a collection.
- **No searchable text:** scanned PDFs require OCR in another tool before upload. Encrypted documents are rejected. Maximum upload: 20 MB, 300 pages, 1,500 chunks.
- **Port in use:** `PORT=5002 python app.py`, then open the corresponding URL.
- **Dependencies fail on Apple's system Python:** create a fresh environment with Python 3.11/3.12. Do not install into the system Python.

This is a single-user local application. It binds to localhost, uses no CDN assets and has no external model calls when using the default Ollama address. It has no multiuser authentication and is not configured for public deployment. Do not expose the Flask port to a network. Uploaded files are validated by extension, signature and PDF parser, stored under random IDs, and served only through validated IDs. Secrets, PDFs, databases, runtime data and virtual environments are excluded by `.gitignore`.

## Manual acceptance checks

- Save income 1,500 and expenses totaling 1,300: remaining balance is 200. Refresh and confirm persistence. Overspend to see a negative balance.
- Set a 1,200 goal with 300 saved and a future date: progress is 25%; monthly contribution is 900 divided by contribution months, rounded up to cents. Check past dates, zero target, and already completed goals.
- Upload a two-page finance PDF and ask a question answered on page two. Expand the citation and open its page. Ask an unrelated question and check for an evidence-gap response.
- Stop Ollama and ask another question: the UI should give a useful service error while the other tools still work.
- Check navigation and layouts on a narrow phone viewport. User text, PDF excerpts and model output render as text rather than executable HTML.

## Restart and verify an existing installation

Do not delete `data/` or re-upload existing PDFs for this update.

1. In the terminal running Flask, press **Ctrl+C**.
2. Open Ollama, or run `ollama serve` in a separate terminal if it is not running.
3. In a project terminal:

```sh
cd /Users/sarita/Desktop/finwise-ai-new
source .venv/bin/activate
ollama list
```

If either model is missing, install it:

```sh
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Start Flask using the existing environment and data directory:

```sh
python app.py
```

Open http://127.0.0.1:5001 and reload the browser to load the updated source panel.
If you previously set `FINWISE_DATA_DIR`, retain that same setting. The default
is this project's `data/` directory.

From another terminal, check the running services:

```sh
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:5001/api/status
curl http://127.0.0.1:5001/api/documents
curl -X POST http://127.0.0.1:5001/api/chat -H 'Content-Type: application/json' -d '{"question":"what is valuation"}'
```

Confirm the existing 41-page PDF is listed. In Finance AI expert, ask
“what is valuation” and “Explain the DCF valuation process.” Expand sources,
compare the supporting quotes with the displayed passages, and open each PDF
link to verify its page. Ask “What is the capital of France?”; this DCF PDF
should yield an explicit insufficient-evidence explanation with no sources.
If generation fails twice, the terminal logs the validation reason. It never
substitutes a guessed source. Slow local hardware may take several minutes
across two generation attempts.

Automated checks (no running Ollama needed):

```sh
source .venv/bin/activate
python -m pytest -q
python -m unittest discover -s tests -p '*unittest.py'
osascript -l JavaScript tests/check_frontend.js "$PWD/static/app.js"
```

Latest verification: 52 pytest tests, five dependency-free unittest checks,
and frontend syntax plus five calculator checks passed. All 42 existing
indexed passages were checked against their original PDF pages. Tests use
temporary databases; existing uploads are preserved. The current sandbox could
not connect to either local service, so live Llama generation, semantic ranking
for the original question, and browser rendering remain unverified. Dependency
warnings concern urllib3/LibreSSL and PyMuPDF SWIG deprecations.
