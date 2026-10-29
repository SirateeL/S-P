#!/usr/bin/env bash
# Builds output/: a sample report (retrieval only) and the Studio page as index.html.
# Used by both the local deploy script and GitHub Actions.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
OLLAMA_URL="http://127.0.0.1:11434"
mkdir -p output

# --- 1. a small, realistic syllabus to analyse -------------------------------
"$PYTHON" - <<'PY'
import math
import fitz

W, H, M = 595, 842, 64
doc = fitz.open()
page = doc.new_page(width=W, height=H)
y = M


def block(text, size=11, font="helv", gap=10):
    global page, y
    lines = max(1, math.ceil(fitz.get_text_length(text, fontname=font, fontsize=size) / (W - 2 * M - 6)) + 1)
    h = lines * size * 1.4 + 4
    if y + h > H - M:
        page, y = doc.new_page(width=W, height=H), M
    if page.insert_textbox(fitz.Rect(M, y, W - M, y + h), text, fontsize=size, fontname=font) < 0:
        raise SystemExit("sample syllabus layout overflow")
    y += h + gap


block("COURSE SYLLABUS: DESIGN RESEARCH METHODS", 20, "hebo", 14)
block("Instructor: Dr. Placeholder. Office hours: Tuesdays 14:00 to 16:00 in room 304, or by appointment.")
SECTIONS = [
    ("Course Description",
     ["This course introduces qualitative and quantitative methods for studying how people use designed things. "
      "Students plan, run and report a small study over the term. Prerequisite: one prior course in design or the "
      "social sciences. The course carries three credits."]),
    ("Learning Outcomes",
     ["By the end of the course, students will be able to frame a research question, choose a fitting method, and "
      "justify that choice in writing.",
      "Plan and conduct a semi-structured interview study with at least six participants."]),
    ("Grading",
     ["Participation counts for 20% of the final grade, the two short assignments for 15% each, the midterm critique "
      "for 10%, and the final project for 40%. Late work loses five points per day."]),
    ("Schedule",
     ["Week 3 covers interviews: we practise writing neutral questions and probing for detail, then run paired mock "
      "interviews in class. The midterm critique is due in week 7 and the final project is presented in week 14."]),
    ("Academic Integrity",
     ["All submitted work must be your own. Cite every source and disclose any use of generative tools in an "
      "appendix to the assignment. Attendance is expected at every studio session."]),
    ("Required Readings",
     ["Each week's readings are posted on the course site by Friday. Expect roughly forty pages per week and come "
      "prepared to discuss them."]),
]
for title, paras in SECTIONS:
    block(title, 13, "hebo", 6)
    for p in paras:
        block(p)
doc.save("sample_syllabus.pdf")
print("wrote sample_syllabus.pdf")
PY

# --- 2. analyse it (embeddings need a local Ollama; no language model is used) ---
ollama_up() { curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }
if ! ollama_up && command -v ollama >/dev/null 2>&1; then
  (nohup ollama serve >/tmp/etude-ollama.log 2>&1 &)
  for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
fi
if ollama_up; then
  curl -sf "$OLLAMA_URL/api/tags" | grep -q 'nomic-embed-text' || ollama pull nomic-embed-text || true
  if "$PYTHON" pdf_analyzer.py sample_syllabus.pdf --no-llm --out output; then
    echo "built output/report.html"
  elif [ -n "${CI:-}" ]; then
    echo "sample report failed in CI" >&2; exit 1
  else
    echo "warning: sample report failed; continuing without it" >&2
  fi
elif [ -n "${CI:-}" ]; then
  echo "Ollama is not running in CI" >&2; exit 1
else
  echo "warning: Ollama not running, skipping the sample report (start it with: ollama serve)" >&2
fi

# --- 3. the Studio page becomes the site root --------------------------------
"$PYTHON" - <<'PY'
import pdf_analyzer
page = pdf_analyzer.APP_PAGE.replace(' nonce="__NONCE__"', "")
assert "Ask the page" in page and "__NONCE__" not in page
open("output/index.html", "w", encoding="utf-8").write(page)
print("built output/index.html")
PY
touch output/.nojekyll
