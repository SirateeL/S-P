# FinWise AI

A local Flask finance learning platform for university students. It includes a responsive dashboard, monthly budget planner, savings calculator, four-topic learning hub, and PDF-grounded AI with page citations. It runs locally with Ollama, so no paid API or cloud account is required.

## Project location

All code, tests and full setup instructions are in [`finwise-ai-new/`](finwise-ai-new/README.md).

## Quick start (macOS)

```sh
cd finwise-ai-new
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2:3b
python app.py
```

Then open http://127.0.0.1:5001. See `finwise-ai-new/README.md` for details.
