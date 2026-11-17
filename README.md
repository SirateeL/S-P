# FinWise AI

A local Flask finance learning platform for university students. It includes a responsive dashboard, monthly budget planner, savings calculator, four-topic learning hub, and PDF-grounded AI with page citations. It runs locally with Ollama, so no paid API or cloud account is required.

## Project location

All code, tests and full setup instructions are in [finwise-ai-new/](finwise-ai-new/).

## Quick start (download and run)

1. Click the green **Code** button on this page, then **Download ZIP**. Unzip it and open the `finwise-ai-new` folder inside.
2. Install [Python 3](https://www.python.org/downloads/) (3.11 or 3.12 recommended; tested on 3.9.6) and [Ollama](https://ollama.com/download). Open the Ollama app.
3. In Terminal, go into the `finwise-ai-new` folder and run:

        python3 -m venv .venv
        source .venv/bin/activate
        python -m pip install -r requirements.txt
        ollama pull nomic-embed-text
        ollama pull llama3.2:3b
        python app.py

4. Open http://127.0.0.1:5001 in your browser. Press Ctrl+C in Terminal to stop.

No PDFs are included, so upload your own text-based finance PDF in **Finance AI expert**. Commands above are for macOS/Linux; Windows is untested. See [finwise-ai-new/README.md](finwise-ai-new/README.md) for details.
