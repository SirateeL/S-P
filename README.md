# FinWise AI

A local Flask finance learning platform for university students. It includes a responsive dashboard, monthly budget planner, savings calculator, four-topic learning hub, and PDF-grounded AI with page citations. It runs locally with Ollama, so no paid API or cloud account is required.

## Project location

All code, tests and full setup instructions are in [finwise-ai-new/](finwise-ai-new/).

## Quick start (macOS, one command)

1. Install [Python 3](https://www.python.org/downloads/) and [Ollama](https://ollama.com/download), then open the Ollama app.
2. Open Terminal, paste this whole command and press Enter. It downloads the project to your Desktop, installs it and starts the app (the first run downloads several GB of AI models):

        cd ~/Desktop && curl -L -o finwise.zip https://github.com/SirateeL/S-P/archive/refs/heads/main.zip && unzip -q -o finwise.zip && cd S-P-main/finwise-ai-new && python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt && ollama pull nomic-embed-text && ollama pull llama3.2:3b && python app.py

3. Open http://127.0.0.1:5001 in your browser. Press Ctrl+C in Terminal to stop.

To start it again later, run:

        cd ~/Desktop/S-P-main/finwise-ai-new && source .venv/bin/activate && python app.py

No PDFs are included, so upload your own text-based finance PDF in **Finance AI expert**. Windows is untested. See [finwise-ai-new/README.md](finwise-ai-new/README.md) for details.
