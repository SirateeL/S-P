#!/usr/bin/env python3
"""pdf_analyzer.py -- offline RAG over research papers and course syllabi.

    PyMuPDF  ->  structure-aware chunks  ->  Ollama embeddings  ->  ChromaDB
             ->  top-k retrieval  ->  llama3.2 answer  ->  RESULTS.md + report.html

Every citation carries: page, paragraph, section heading, an exact quoted
snippet, and a similarity percentage. Nothing leaves localhost.

    python pdf_analyzer.py paper.pdf syllabus.pdf -q "How is grading weighted?"   # command line
    python pdf_analyzer.py --serve --open                                        # drag-and-drop in the browser
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import math
import os
import re
import secrets
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

# Keep every library quiet and local *before* importing them.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import fitz  # PyMuPDF
    import chromadb
    from chromadb.config import Settings
    import ollama
    from rich import box
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Missing dependency ({exc.name}). Run: pip install -r requirements.txt")

# --------------------------------------------------------------------------- #
# Palette & tunables
# --------------------------------------------------------------------------- #
STONE, ESPRESSO, SLATE = "#F5F4F0", "#1C1B1A", "#7E7B75"

INDEX_VERSION = "1"      # bump when parsing/chunking changes, invalidates cache
MAX_CHARS = 1000         # soft ceiling for one indexed passage
MIN_CHARS = 40           # shorter fragments (captions, table cells) are dropped
MIN_WORDS = 6
SNIPPET_CHARS = 320
EMBED_BATCH = 16

console = Console(highlight=False)


class Cancelled(Exception):
    """Raised inside a web job when the reader presses Cancel."""


class EtudeError(Exception):
    """A problem the user can act on (shown plainly, never as a traceback)."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Line:
    text: str
    size: float
    bold: bool


@dataclass
class Block:
    page: int
    lines: list
    y0: float
    y1: float
    page_h: float

    @property
    def text(self) -> str:
        return clean(join_lines([ln.text for ln in self.lines]))


@dataclass
class Paragraph:
    page: int
    number: int
    section: str
    text: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc: str
    page: int
    paragraph: int
    section: str
    part: int
    text: str


@dataclass
class Citation:
    n: int
    doc: str
    page: int
    paragraph: int
    section: str
    quote: str
    truncated: bool
    score: float
    text: str


@dataclass
class QA:
    question: str
    kind: str
    answer: str
    citations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class DocInfo:
    doc_id: str
    name: str
    title: str
    kind: str
    pages: int
    paragraphs: int
    chunks: int
    empty_pages: list
    cached: bool = False


@dataclass
class Run:
    generated: datetime
    llm: str
    embed_model: str
    generated_text: bool
    top_k: int
    docs: list
    qas: list
    skipped: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Text hygiene
# --------------------------------------------------------------------------- #
_REPLACE = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\u00ad": "", "\u2010": "-", "\u00a0": " ",
}


def clean(text: str) -> str:
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def join_lines(lines) -> str:
    """Join wrapped lines, healing end-of-line hyphenation ("meth-" + "odology")."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}" if out else ln
    return out


# --------------------------------------------------------------------------- #
# PDF ingestion (PyMuPDF)
# --------------------------------------------------------------------------- #
_BOLD_FONT = re.compile(
    r"bold|black|heavy|demi|medi\b|cmbx|semib|[-_]bd\b|[-_]b$|\.b$", re.I)


def _is_bold(span: dict) -> bool:
    return bool(span.get("flags", 0) & 16) or bool(_BOLD_FONT.search(span.get("font", "")))


def extract_blocks(pdf_path: Path):
    """Read every text block with its font size and weight. Returns (blocks, meta)."""
    blocks, empty_pages = [], []
    with fitz.open(str(pdf_path)) as doc:
        meta_title = ((doc.metadata or {}).get("title") or "").strip()
        n_pages = doc.page_count
        for pno, page in enumerate(doc, start=1):
            height = float(page.rect.height)
            seen_chars = 0
            for b in page.get_text("dict")["blocks"]:
                if b.get("type", 0) != 0:      # skip images
                    continue
                lines = []
                for ln in b.get("lines", []):
                    spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    weights, bold_chars, total = Counter(), 0, 0
                    for s in spans:
                        n = len(s["text"].strip())
                        weights[round(float(s["size"]), 1)] += n
                        total += n
                        if _is_bold(s):
                            bold_chars += n
                    seen_chars += total
                    text = "".join(s["text"] for s in ln["spans"])
                    lines.append(Line(text, weights.most_common(1)[0][0], bold_chars >= 0.6 * total))
                if lines:
                    _, y0, _, y1 = b["bbox"]
                    blocks.append(Block(pno, lines, float(y0), float(y1), height))
            if seen_chars < 20:
                empty_pages.append(pno)
    return blocks, {"title": meta_title, "pages": n_pages, "empty_pages": empty_pages}


def body_font_size(blocks) -> float:
    c = Counter()
    for b in blocks:
        for ln in b.lines:
            c[ln.size] += len(ln.text.strip())
    return c.most_common(1)[0][0] if c else 10.0


# --------------------------------------------------------------------------- #
# Structure: headings, paragraphs
# --------------------------------------------------------------------------- #
_NUMBERED = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVXLC]{1,5}\.|[A-Z]\."
    r"|(?i:week|session|module|unit|lecture|chapter|part|section)\s+\d{1,2}[:.\-\u2013]?)\s+[A-Z]")
_KNOWN = re.compile(
    r"^(?:abstract|introduction|background|related work|literature review|methods?|methodology"
    r"|materials and methods|data|experiments?|evaluation|results?|findings|analysis|discussion"
    r"|limitations|conclusions?|future work|acknowledge?ments?|references|bibliography|appendix"
    r"|appendices|course (?:description|overview|objectives|outcomes|schedule|policies)"
    r"|learning (?:outcomes|objectives)|prerequisites?|grading|assessment|assignments?|schedule"
    r"|policies|attendance|academic integrity|required (?:texts?|readings?)|readings?"
    r"|office hours|instructor|contact|overview|objectives)\b", re.I)
_CAPTION = re.compile(r"^(?:fig(?:ure)?|table|tab|eq(?:uation)?|algorithm|listing)\.?\s*\d", re.I)
# A caption has a separator after its number ("Figure 2: ..."); "Table 2 shows..." is body text.
_CAPTION_STRICT = re.compile(r"^(?:fig(?:ure)?|table|tab|algorithm|listing)\.?\s*\d+\s*[:.–—-]", re.I)
_REFS = re.compile(r"^(?:references|bibliography|works cited|literature cited)\b", re.I)
_END_SENT = re.compile(r"[.!?:;)\]\"\u201d\u2019]\s*$")


def line_is_heading(text: str, size: float, bold: bool, body: float, single_line: bool = False) -> bool:
    t = text.strip()
    words = t.split()
    if not (2 <= len(t) <= 110) or len(words) > 14:
        return False
    if t[-1] in ".,;" or _CAPTION.match(t) or not any(ch.isalpha() for ch in t):
        return False
    numbered = bool(_NUMBERED.match(t))
    stripped = re.sub(r"^\S+\s+", "", t) if numbered else t
    known = len(words) <= 6 and bool(_KNOWN.match(stripped.rstrip(":")))
    allcaps = t.upper() == t and sum(ch.isalpha() for ch in t) >= 4
    if size >= body * 1.18 and "," not in t and "@" not in t:
        return True
    if bold and (numbered or known or allcaps):
        return True
    if numbered and size >= body * 1.02:
        return True
    if known and allcaps:
        return True
    if bold and single_line and len(words) <= 8 and size >= body * 0.97:
        return True
    return False


def split_block(b: Block, body: float):
    """Peel heading lines off the top of a block -> (heading_lines, body_lines)."""
    n, k = len(b.lines), 0
    while k < n and line_is_heading(b.lines[k].text, b.lines[k].size, b.lines[k].bold,
                                    body, single_line=(n == 1)):
        k += 1
    return b.lines[:k], b.lines[k:]


def is_noise(b: Block) -> bool:
    t = b.text
    if not t:
        return True
    if re.fullmatch(r"[\W\d_]+", t):
        return True
    if re.fullmatch(r"(?:page|p\.)?\s*\d+(?:\s*(?:of|/)\s*\d+)?", t, re.I):
        return True
    if _CAPTION_STRICT.match(t):
        return True
    in_margin = b.y1 < b.page_h * 0.06 or b.y0 > b.page_h * 0.94
    return in_margin and len(t) < 120


def build_paragraphs(blocks, body: float, keep_refs: bool = False):
    """Walk blocks in reading order -> numbered paragraphs tagged with their section.

    Paragraph numbers restart on every page and count only substantive body text
    (headings, captions, running heads and page numbers are not counted).
    """
    paras, title_parts = [], []
    counters = defaultdict(int)
    section, seen_section, skipping = "Front matter", False, False
    pending, pending_skip = None, False

    def flush():
        nonlocal pending
        p, pending = pending, None
        if p is None or pending_skip:
            return
        if len(p.text) < MIN_CHARS or len(p.text.split()) < MIN_WORDS:
            return
        counters[p.page] += 1
        p.number = counters[p.page]
        paras.append(p)

    for b in blocks:
        if is_noise(b):
            continue
        head, rest = split_block(b, body)
        if head:
            flush()
            text = clean(join_lines([ln.text for ln in head])).rstrip(":").strip()
            if b.page == 1 and not seen_section and max(ln.size for ln in head) >= body * 1.5:
                title_parts.append(text)
            else:
                section, seen_section = text, True
                core = re.sub(r"^\S+\s+", "", text) if _NUMBERED.match(text) else text
                skipping = (not keep_refs) and bool(_REFS.match(core))
        if rest:
            text = clean(join_lines([ln.text for ln in rest]))
            if not text:
                continue
            if (pending is not None and not head and pending.page == b.page
                    and not _END_SENT.search(pending.text) and text[:1].islower()):
                pending.text = f"{pending.text} {text}"       # paragraph split by a column/page box
            else:
                flush()
                pending = Paragraph(b.page, 0, section, text)
                pending_skip = skipping
    flush()
    return paras, " ".join(title_parts).strip()


# --------------------------------------------------------------------------- #
# Sentences, chunks, snippets
# --------------------------------------------------------------------------- #
_SENT_END = re.compile(r"([.!?][\"\u201d')\]]*)\s+(?=[A-Z0-9\u201c\"(\[])")
_ABBR = {"al.", "e.g.", "i.e.", "fig.", "figs.", "eq.", "eqs.", "vs.", "cf.", "dr.", "prof.",
         "mr.", "mrs.", "ms.", "no.", "sec.", "approx.", "ch.", "pp.", "vol."}


def sentence_spans(text: str):
    """(start, end) offsets of each sentence; slicing text[start:end] is always exact."""
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        words = text[start:m.end(1)].split()
        tok = words[-1].lower() if words else ""
        if tok in _ABBR or re.fullmatch(r"[a-z]\.", tok):
            continue
        spans.append((start, m.end(1)))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def chunk_paragraph(p: Paragraph, doc_id: str, doc: str):
    text = p.text
    if len(text) <= MAX_CHARS:
        parts = [text]
    else:
        spans, parts, i = sentence_spans(text), [], 0
        while i < len(spans):
            j = i
            while j < len(spans) and spans[j][1] - spans[i][0] <= MAX_CHARS:
                j += 1
            j = max(j, i + 1)
            parts.append(text[spans[i][0]:spans[j - 1][1]])
            if j >= len(spans):
                break
            i = j - 1 if j - 1 > i else j          # one-sentence overlap between windows
    return [Chunk(f"{doc_id}-p{p.page:03d}-n{p.number:03d}-{k}", doc_id, doc, p.page,
                  p.number, p.section, k, part) for k, part in enumerate(parts)]


_STOP = set(
    "a an the of and or to in on for with as by is are was were be been being this that these those "
    "it its from at which who whom what when where why how do does did can could should would may "
    "might will shall not no than then there their they them we our you your i he she his her has "
    "have had into over under about between among also such more most other any each both via per "
    "using use used based".split())


def terms(s: str) -> set:
    out = set()
    for w in re.findall(r"[a-z0-9]{3,}", s.lower()):
        if w in _STOP:
            continue
        for suf in ("ing", "ed", "es", "s", "ly"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def best_snippet(text: str, question: str, limit: int = SNIPPET_CHARS):
    """Pick the sentence (or two) in `text` that best overlaps the question.

    The result is always a literal slice of `text`, so it is an exact quote.
    """
    spans, q = sentence_spans(text), terms(question)
    scores = [len(q & terms(text[a:b])) for a, b in spans]
    best = max(range(len(spans)), key=lambda i: (scores[i], -i))
    a, b = spans[best]
    if best + 1 < len(spans):
        na, nb = spans[best + 1]
        if nb - a <= limit and (b - a < 80 or q & terms(text[na:nb])):
            b = nb
    truncated = False
    if b - a > limit:
        cut = text.rfind(" ", a, a + limit)
        b = cut if cut > a + limit * 0.6 else a + limit
        truncated = True
    return text[a:b].strip(), truncated


# --------------------------------------------------------------------------- #
# Ollama helpers
# --------------------------------------------------------------------------- #
def normalise_host(raw: str) -> str:
    raw = raw.strip()
    if "://" not in raw:
        raw = "http://" + raw
    u = urlparse(raw)
    host = u.hostname or ""
    if host in ("", "0.0.0.0"):
        host = "127.0.0.1"
    if not (host == "localhost" or host.startswith("127.") or host == "::1"):
        fail(f"Refusing non-local Ollama host '{host}'. This tool is offline by design; "
             "point --host at localhost.")
    shown = "[::1]" if host == "::1" else host
    return f"{u.scheme}://{shown}:{u.port or 11434}"


def model_names(resp) -> set:
    names = set()
    for m in resp["models"]:
        name = None
        for key in ("model", "name"):
            try:
                name = m[key]
                break
            except Exception:
                continue
        if name:
            names.add(name)
    return names


def has_model(names: set, wanted: str) -> bool:
    if wanted in names:
        return True
    if ":" not in wanted:
        return any(n.split(":")[0] == wanted for n in names)
    return False


def unit(v):
    v = [float(x) for x in v]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


class Embedder:
    def __init__(self, client, model: str):
        self.client, self.model = client, model
        nomic = "nomic" in model
        self.prefix = {"doc": "search_document: " if nomic else "",
                       "query": "search_query: " if nomic else ""}

    def embed(self, texts, kind="doc", on_batch=None):
        out = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = [self.prefix[kind] + t for t in texts[i:i + EMBED_BATCH]]
            resp = self.client.embed(model=self.model, input=batch)
            out.extend(unit(v) for v in resp["embeddings"])
            if on_batch:
                on_batch(len(out), len(texts))
        if len(out) != len(texts):
            raise RuntimeError("Ollama returned an unexpected number of embeddings.")
        return out


SYSTEM_PROMPT = (
    "You are a careful research assistant running fully offline. Answer the question using ONLY "
    "the numbered excerpts provided. After each claim, cite the supporting excerpt(s) as [1], [2], "
    "etc. Never cite a number that was not provided. Do not invent quotes, page numbers or facts. "
    "If the excerpts do not contain the answer, say so plainly. Write plain prose, at most 150 words.")


def build_prompt(question: str, cites, kind: str) -> str:
    ex = "\n\n".join(
        f"[{c.n}] (page {c.page}, paragraph {c.paragraph}, section: {c.section})\n{c.text}"
        for c in cites)
    return f"Document type: {kind}\n\nExcerpts:\n{ex}\n\nQuestion: {question}\n\nAnswer:"


def generate(client, model: str, question: str, cites, kind: str) -> str:
    resp = client.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": build_prompt(question, cites, kind)}],
        options={"temperature": 0.1, "num_ctx": 4096})
    return resp["message"]["content"].strip()


_CITE_GROUP = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")
_NO_ANSWER = re.compile(r"do(?:es)? not (?:contain|answer|provide|mention)|not (?:addressed|covered|stated)", re.I)


def audit_answer(answer: str, n_cites: int):
    """Cheap guard against a small model citing excerpts that do not exist."""
    refs = {int(x) for g in _CITE_GROUP.findall(answer) for x in re.findall(r"\d+", g)}
    warnings = []
    bad = sorted(r for r in refs if not 1 <= r <= n_cites)
    if bad:
        verb = "does" if len(bad) == 1 else "do"
        warnings.append("The answer cites " + ", ".join(f"[{b}]" for b in bad)
                        + f", which {verb} not match any retrieved passage.")
    if not refs and not _NO_ANSWER.search(answer):
        warnings.append("The answer carries no citation markers; check it against the passages below.")
    return warnings


# --------------------------------------------------------------------------- #
# Index (ChromaDB) and retrieval
# --------------------------------------------------------------------------- #
def collection_name(embed_model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", embed_model).strip("_")[:40] or "default"
    return f"papers_{slug}"


def open_collection(db_path: Path, embed_model: str):
    client = chromadb.PersistentClient(path=str(db_path),
                                       settings=Settings(anonymized_telemetry=False))
    name = collection_name(embed_model)
    try:
        return client.get_or_create_collection(
            name=name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)
    except (TypeError, ValueError):                       # older chromadb releases
        return client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}, embedding_function=None)


def doc_fingerprint(path: Path, keep_refs: bool) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    h.update(f"|v{INDEX_VERSION}|{int(keep_refs)}|{MAX_CHARS}".encode())
    return h.hexdigest()[:12]


def detect_kind(paras) -> str:
    sample = " ".join(f"{p.section} {p.text}" for p in paras[:30]).lower()
    signals = (r"syllabus", r"office hours", r"grading|grade breakdown", r"prerequisite",
               r"credits?\b|credit hours", r"attendance", r"learning (?:outcomes|objectives)")
    return "syllabus" if sum(bool(re.search(s, sample)) for s in signals) >= 2 else "paper"


def pick_title(meta_title: str, detected: str, stem: str) -> str:
    junk = re.compile(r"\.(?:docx?|tex|pdf|indd)$|^microsoft word|^untitled", re.I)
    if len(meta_title) >= 6 and not junk.search(meta_title):
        return meta_title
    return detected if len(detected) >= 6 else stem


def ingest(pdf: Path, col, emb: Embedder, kind: str = "auto", keep_refs: bool = False,
           reindex: bool = False, on_progress=None) -> DocInfo:
    """Parse one PDF and (unless cached) embed it into `col`.

    `on_progress(done, total)` is called while embedding, so the terminal and the
    web interface can each show their own progress.
    """
    doc_id = doc_fingerprint(pdf, keep_refs)
    blocks, meta = extract_blocks(pdf)
    paras, detected = build_paragraphs(blocks, body_font_size(blocks), keep_refs)
    chunks = [c for p in paras for c in chunk_paragraph(p, doc_id, pdf.name)]
    kind = kind if kind != "auto" else detect_kind(paras)
    info = DocInfo(doc_id, pdf.name, pick_title(meta["title"], detected, pdf.stem), kind,
                   meta["pages"], len(paras), len(chunks), meta["empty_pages"])
    if not chunks:
        return info
    existing = col.get(where={"doc_id": doc_id}, limit=1)["ids"]
    if len(existing) and not reindex:
        info.cached = True
        return info
    if len(existing):
        col.delete(where={"doc_id": doc_id})
    if on_progress:
        on_progress(0, len(chunks))
    vecs = emb.embed([f"{c.section}\n{c.text}" for c in chunks], "doc", on_batch=on_progress)
    for i in range(0, len(chunks), 64):
        part = chunks[i:i + 64]
        col.add(ids=[c.id for c in part], embeddings=vecs[i:i + 64],
                documents=[c.text for c in part],
                metadatas=[{"doc_id": c.doc_id, "doc": c.doc, "page": c.page,
                            "paragraph": c.paragraph, "section": c.section, "part": c.part}
                           for c in part])
    return info


def retrieve(col, emb: Embedder, question: str, k: int, doc_ids, min_score: float):
    qv = emb.embed([question], "query")[0]
    total = col.count()
    if total == 0:
        return []
    kwargs = {"query_embeddings": [qv], "n_results": max(1, min(k, total)),
              "include": ["documents", "metadatas", "embeddings"]}
    if len(doc_ids) == 1:
        kwargs["where"] = {"doc_id": doc_ids[0]}
    elif doc_ids:
        kwargs["where"] = {"doc_id": {"$in": list(doc_ids)}}
    res = col.query(**kwargs)
    rows = []
    for text, meta, ev in zip(res["documents"][0], res["metadatas"][0], res["embeddings"][0]):
        # Both vectors are unit length, so the dot product is the cosine similarity,
        # whatever distance metric the collection was configured with.
        sim = sum(a * float(b) for a, b in zip(qv, ev))
        rows.append((round(max(0.0, min(1.0, sim)) * 100, 1), text, meta))
    rows.sort(key=lambda r: -r[0])
    cites = []
    for score, text, meta in rows:
        if score < min_score:
            continue
        quote, truncated = best_snippet(text, question)
        cites.append(Citation(len(cites) + 1, meta["doc"], int(meta["page"]), int(meta["paragraph"]),
                              meta["section"], quote, truncated, score, text))
    return cites


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def bar(score: float, width: int = 10) -> str:
    filled = max(0, min(width, round(score / 100 * width)))
    return "\u25ae" * filled + "\u25af" * (width - filled)


def md_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def render_markdown(run: Run) -> str:
    multi = len(run.docs) > 1
    L = []
    L.append("# ANALYSIS")
    L.append("")
    L.append(f"`N\u00ba {run.generated:%y%m%d-%H%M}` \u00b7 {run.generated:%d %B %Y} \u00b7 "
             f"{run.llm if run.generated_text else 'retrieval only'} \u00b7 {run.embed_model}")
    L.append("")
    L.append("\u2014\u2014\u2014")
    L.append("")
    L.append("## Documents")
    L.append("")
    L.append("| N\u00ba | Title | File | Kind | Pages | \u00b6 | Passages |")
    L.append("|---:|---|---|---|---:|---:|---:|")
    for i, d in enumerate(run.docs, 1):
        L.append(f"| {i:02d} | {md_cell(d.title)} | `{md_cell(d.name)}` | {d.kind} | {d.pages} "
                 f"| {d.paragraphs} | {d.chunks} |")
    for d in run.docs:
        if d.empty_pages:
            pages = ", ".join(map(str, d.empty_pages[:12]))
            L.append("")
            L.append(f"> \u2020 `{md_cell(d.name)}`: no extractable text on page(s) {pages}. "
                     "Scanned pages need OCR first.")
    for name in run.skipped:
        L.append("")
        L.append(f"> \u2020 `{md_cell(name)}`: skipped, no extractable text (scanned PDF?). Run OCR first.")
    for i, qa in enumerate(run.qas, 1):
        L.append("")
        L.append("---")
        L.append("")
        L.append(f"## {i:02d} \u2014 {qa.question}")
        L.append("")
        L.append(re.sub(r"(?m)^(\s*)#", r"\1\\#", qa.answer))
        for w in qa.warnings:
            L.append("")
            L.append(f"*\u2020 {w}*")
        if qa.citations:
            L.append("")
            L.append("### Evidence")
            L.append("")
            head = "| Ref |" + (" Document |" if multi else "") + " Page | \u00b6 | Section | Match |"
            sep = "|:---:|" + ("---|" if multi else "") + "---:|---:|---|---|"
            L.append(head)
            L.append(sep)
            for c in qa.citations:
                doc = f" `{md_cell(c.doc)}` |" if multi else ""
                L.append(f"| **[{c.n}]** |{doc} {c.page} | {c.paragraph} | \u00a7 {md_cell(c.section)} "
                         f"| `{bar(c.score)}` {c.score:.1f}% |")
            for c in qa.citations:
                L.append("")
                tail = " \u2026" if c.truncated else ""
                L.append(f"> **[{c.n}]** \u201c{c.quote}\u201d{tail}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*Produced offline by `pdf_analyzer.py`. Match is cosine similarity between the question "
             "and the passage embedding, multiplied by 100: a ranking signal, not a probability of "
             "correctness. Quotes are literal passages of the extracted text (line-wrap hyphens "
             "and ligatures normalised).*")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Design system (shared by report.html and the web interface)
# --------------------------------------------------------------------------- #
_GRAIN = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'>"
          "<filter id='g' x='0' y='0' width='100%25' height='100%25'>"
          "<feTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='2' stitchTiles='stitch'/>"
          "<feColorMatrix values='0 0 0 0 .098 0 0 0 0 .094 0 0 0 0 .09 0 0 0 .085 0'/></filter>"
          "<rect width='100%25' height='100%25' filter='url(%23g)'/></svg>\")")

TOKENS = r"""
:root{
--linen:#F5F3EE;--parchment:#EFECE6;--ink:#191817;--slate:#6F6A61;--rule:#C4BEB4;--ink-soft:#4A4743;
--serif:"Didot","Bodoni MT","Cinzel","Georgia",serif;
--sans:-apple-system,BlinkMacSystemFont,"Helvetica Neue","Arial",sans-serif;
--s1:4px;--s2:8px;--s3:12px;--s4:16px;--s5:24px;--s6:32px;--s7:48px;--s8:64px;--s9:96px;--s10:128px;
--gutter:24px;--ease:cubic-bezier(.2,.7,.2,1);--grain:__GRAIN__}
*{box-sizing:border-box;margin:0;padding:0}
html{background:var(--linen);color:var(--ink);scroll-behavior:smooth;-webkit-text-size-adjust:100%;
-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{font-family:var(--serif);font-size:17px;line-height:1.7;font-kerning:normal;-webkit-font-smoothing:antialiased;
text-rendering:optimizeLegibility;background:var(--linen) var(--grain)}
a{color:inherit}
::selection{background:var(--ink);color:var(--linen)}
:focus-visible{outline:1px solid var(--ink);outline-offset:3px}
h1,h2{text-wrap:balance;font-weight:400}
.label,dt,th,summary{font-family:var(--sans);font-size:10.5px;line-height:1.4;letter-spacing:.12em;
text-transform:uppercase;font-weight:500;color:var(--slate)}
.status{display:inline-flex;align-items:center;gap:var(--s2);font-family:var(--sans);font-size:10px;line-height:1.3;
letter-spacing:.12em;text-transform:uppercase;font-weight:500;color:var(--ink-soft);padding:3px 10px 3px 9px;
border:1px solid var(--rule);border-radius:999px;white-space:nowrap}
.status i{width:5px;height:5px;border-radius:50%;background:var(--slate);flex:none}
.status.ok i{background:var(--ink)}
.status.warn{border-style:dashed;color:var(--slate)}
.status.warn i{background:transparent;box-shadow:inset 0 0 0 1px currentColor}
/* corner framing notches: hairline brackets inset from the edge; charcoal on hover/focus/target */
.notched{position:relative}
.notched::before,.notched::after{content:"";position:absolute;inset:8px;pointer-events:none;
background:
linear-gradient(var(--c),var(--c)) top left/10px 1px no-repeat,linear-gradient(var(--c),var(--c)) top left/1px 10px no-repeat,
linear-gradient(var(--c),var(--c)) top right/10px 1px no-repeat,linear-gradient(var(--c),var(--c)) top right/1px 10px no-repeat,
linear-gradient(var(--c),var(--c)) bottom left/10px 1px no-repeat,linear-gradient(var(--c),var(--c)) bottom left/1px 10px no-repeat,
linear-gradient(var(--c),var(--c)) bottom right/10px 1px no-repeat,linear-gradient(var(--c),var(--c)) bottom right/1px 10px no-repeat}
.notched::before{--c:var(--rule)}
.notched::after{--c:var(--ink);opacity:0;transition:opacity .3s var(--ease)}
.notched:hover::after,.notched:focus-within::after,.notched:target::after,.notched.on::after{opacity:1}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important;scroll-behavior:auto!important}}
""".replace("__GRAIN__", _GRAIN)

LAYOUT = r"""
.page{max-width:1280px;margin:0 auto;padding:0 var(--s7) var(--s9)}
.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));column-gap:var(--gutter)}

/* ---- masthead ---- */
.topbar{position:relative;padding:var(--s5) 0 var(--s4);align-items:center}
.topbar::after{content:"";position:absolute;left:0;right:0;bottom:0;height:1px;background:var(--rule)}
.topbar::before{content:"";position:absolute;left:0;bottom:-1px;width:64px;height:3px;background:var(--ink);z-index:1}
.topbar .wm{grid-column:1/4;color:var(--ink);font-weight:700;letter-spacing:.34em}
.topbar .mid{grid-column:4/10}
.topbar .end{grid-column:10/13;justify-self:end;display:flex;align-items:center;gap:var(--s4)}
.hero{padding:var(--s9) 0 0;align-items:end}
h1{grid-column:1/10;font-size:clamp(3.6rem,11.5vw,9.8rem);line-height:.86;letter-spacing:-.03em;margin-left:-.04em}
.hero .stamp{grid-column:10/13;justify-self:end;text-align:right;padding-bottom:var(--s4)}
.hero .stamp b{display:block;color:var(--ink);font-weight:600;margin-bottom:2px}
.sub{grid-column:4/12;margin-top:var(--s6);font-size:1.22rem;line-height:1.55;color:var(--slate);font-style:italic;
text-wrap:pretty}
@media(max-width:960px){
.page{padding:0 var(--s5) var(--s8)}
.topbar .wm{grid-column:1/7}.topbar .mid{display:none}.topbar .end{grid-column:7/13}
.hero{padding-top:var(--s8)}h1{grid-column:1/13}
.hero .stamp{grid-column:1/13;justify-self:start;text-align:left;padding:var(--s5) 0 0}
.sub{grid-column:1/13;margin-top:var(--s5)}}
@media print{.page{padding:0}}
"""

CSS = TOKENS + LAYOUT + r"""
.meta{position:relative;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin-top:var(--s8);
border-top:1px solid var(--ink);border-bottom:1px solid var(--rule)}
.meta::before,.meta::after{content:"+";position:absolute;top:-10px;font-family:var(--sans);font-size:15px;
line-height:1;color:var(--ink);font-weight:300}
.meta::before{left:-4px}.meta::after{right:-4px}
.meta>div{padding:var(--s5) var(--s5) var(--s5) 0}
.meta>div+div{padding-left:var(--s5);border-left:1px solid var(--rule)}
.meta dd{margin-top:var(--s2);font-size:1.2rem;line-height:1.35;overflow-wrap:anywhere}

/* ---- register ---- */
.regwrap{overflow-x:auto;margin-top:var(--s5)}
table.reg{width:100%;border-collapse:collapse}
table.reg th{text-align:left;padding:var(--s4) var(--s4) var(--s3) 0;border-bottom:1px solid var(--rule);white-space:nowrap}
table.reg th.n{text-align:right}
table.reg td{padding:var(--s4) var(--s4) var(--s4) 0;border-bottom:1px solid var(--rule);vertical-align:middle;
font-size:1.04rem;line-height:1.4}
table.reg td.n{text-align:right;font-family:var(--sans);font-size:12.5px;font-variant-numeric:tabular-nums}
table.reg td.f,table.reg td.k{font-family:var(--sans);font-size:12.5px;color:var(--slate)}
table.reg tr:last-child td{border-bottom:1px solid var(--ink)}
.warn-line{margin-top:var(--s4);font-family:var(--sans);font-size:12px;color:var(--slate)}

/* ---- frame ---- */
.frame{margin-top:var(--s9)}
.rail{grid-column:1/4;position:sticky;top:var(--s6);align-self:start}
.rail ol{list-style:none;margin-top:var(--s4);border-top:1px solid var(--ink)}
.rail li a{display:grid;grid-template-columns:2.6rem 1fr;padding:var(--s3) 0;border-bottom:1px solid var(--rule);
text-decoration:none;font-size:.92rem;line-height:1.45;transition:padding-left .3s var(--ease),color .2s}
.rail li a:hover{padding-left:var(--s2);color:var(--slate)}
.rail .rt{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
main{grid-column:4/13}

/* ---- question & synthesis ---- */
.plate{position:relative;border-top:1px solid var(--ink);padding:var(--s6) 0 var(--s9);scroll-margin-top:var(--s5)}
.plate::before{content:"";position:absolute;top:-2px;left:0;width:48px;height:3px;background:var(--ink)}
.plate-head{display:grid;grid-template-columns:5.4rem minmax(0,1fr);column-gap:var(--gutter);align-items:baseline}
.qn{font-family:var(--sans);font-size:13px;font-weight:700;letter-spacing:.1em;color:var(--ink);white-space:nowrap}
.qn i{font-style:normal;color:var(--slate);font-weight:400;margin-left:.4em}
h2{font-size:clamp(1.6rem,3vw,2.2rem);line-height:1.2;letter-spacing:-.012em}
.synth{margin:var(--s7) 0 0 calc(5.4rem + var(--gutter));padding:var(--s1) 0 var(--s1) var(--s6);
border-left:1px solid var(--ink);max-width:68ch}
.synth-head{display:flex;align-items:center;flex-wrap:wrap;gap:var(--s3) var(--s4);margin-bottom:var(--s5)}
.answer{font-size:1.4rem;line-height:1.7;letter-spacing:-.003em;text-wrap:pretty}
.answer p+p,.answer ul{margin-top:.9em}
.answer ul{padding-left:1.1em}
.answer li::marker{color:var(--slate)}
a.ref,span.ref{font-family:var(--sans);font-size:10px;font-weight:700;letter-spacing:.02em;line-height:1.35;
display:inline-block;position:relative;top:-.5em;min-width:1.9em;padding:1px 6px;margin:0 2px;text-align:center;
border:1px solid var(--ink);border-radius:999px;text-decoration:none;transition:background .2s,color .2s}
a.ref:hover{background:var(--ink);color:var(--linen)}
span.ref.bad{border:1px dashed var(--slate);color:var(--slate)}
.note{margin:var(--s5) 0 0 calc(5.4rem + var(--gutter));padding-left:var(--s6);font-family:var(--sans);font-size:12px;
line-height:1.6;color:var(--slate);max-width:72ch}

/* ---- evidence ---- */
.evidence{margin-top:var(--s8)}
.evidence>.label{display:flex;align-items:center;gap:var(--s4);margin-bottom:var(--s5)}
.evidence>.label::after{content:"";flex:1;height:1px;background:var(--rule)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:var(--s5)}
.cite{display:flex;flex-direction:column;background:var(--parchment);border:1px solid var(--rule);
border-top-color:var(--ink);padding:var(--s6) var(--s6) var(--s5);scroll-margin-top:var(--s5);
transition:transform .35s var(--ease),box-shadow .35s var(--ease),background-color .3s}
.cite:hover{transform:translateY(-3px);box-shadow:0 18px 30px -22px rgba(25,24,23,.45);background:#F1EEE8}
.cite:target{background:var(--linen);box-shadow:0 0 0 1px var(--ink)}
.cite-top{display:flex;align-items:flex-start;gap:var(--s3)}
.idx{font-family:var(--sans);font-size:12px;font-weight:700;letter-spacing:.06em;padding-top:5px}
.tag{display:inline-block;max-width:100%;font-family:var(--sans);font-size:10.5px;line-height:1.4;letter-spacing:.1em;
text-transform:uppercase;font-weight:500;padding:4px 12px;border:1px solid var(--ink);border-radius:14px;
overflow-wrap:anywhere}
.src{margin-top:var(--s3);font-family:var(--sans);font-size:11.5px;color:var(--slate);overflow-wrap:anywhere}
.anchors{display:flex;gap:var(--s6);margin-top:var(--s5);padding:var(--s4) 0;border-top:1px solid var(--rule);
border-bottom:1px solid var(--rule)}
.anchors dd{margin-top:var(--s1);font-family:var(--sans);font-size:1.5rem;line-height:1.1;font-weight:700;
letter-spacing:-.02em;font-variant-numeric:tabular-nums}
blockquote{position:relative;margin-top:var(--s5);padding-top:var(--s6);font-size:1.14rem;line-height:1.68;
font-style:italic;overflow-wrap:anywhere;text-wrap:pretty}
blockquote::before{content:"\201C";position:absolute;top:-6px;left:-2px;font-family:var(--serif);font-style:normal;
font-size:3.6rem;line-height:1;color:var(--slate)}
.foot{margin-top:auto;padding-top:var(--s5)}
.match{display:grid;grid-template-columns:auto 1fr auto;align-items:center;column-gap:var(--s3)}
.track{position:relative;height:1px;background:var(--rule)}
.track i{position:absolute;left:0;top:-1px;height:3px;background:var(--ink);transition:width .6s var(--ease)}
.pct{font-family:var(--sans);font-size:12.5px;font-weight:700;font-variant-numeric:tabular-nums}
details{margin-top:var(--s4)}
summary{cursor:pointer;list-style:none;display:inline-block;transition:color .2s}
summary::-webkit-details-marker{display:none}
summary::before{content:"+ "}details[open] summary::before{content:"\2013 "}
summary:hover{color:var(--ink)}
details p{margin-top:var(--s3);padding-top:var(--s3);border-top:1px solid var(--rule);font-size:.97rem;line-height:1.65;
color:var(--ink-soft);overflow-wrap:anywhere}

/* ---- colophon ---- */
footer{margin-top:var(--s5);padding-top:var(--s6);border-top:1px solid var(--ink)}
footer p{max-width:72ch;font-family:var(--sans);font-size:12px;line-height:1.75;color:var(--slate)}
footer strong{color:var(--ink);font-weight:600}
.sw{display:flex;flex-wrap:wrap;gap:var(--s2) var(--s5);margin-top:var(--s5)}
.sw span{display:flex;align-items:center;gap:var(--s2)}
.sw i{width:12px;height:12px;border:1px solid var(--rule);display:inline-block}

@media(max-width:960px){
.meta{grid-template-columns:repeat(2,minmax(0,1fr));margin-top:var(--s7)}
.meta>div:nth-child(3){padding-left:0;border-left:0}
.meta>div:nth-child(n+3){border-top:1px solid var(--rule)}
.frame{margin-top:var(--s8)}.rail{display:none}main{grid-column:1/13}
.plate{padding-bottom:var(--s8)}
.plate-head{grid-template-columns:1fr;row-gap:var(--s3)}
.synth,.note{margin-left:0}.synth{padding-left:var(--s5)}.note{padding-left:var(--s5)}}
@media(max-width:640px){
body{font-size:16px}.answer{font-size:1.22rem}
table.reg th:nth-child(3),table.reg td:nth-child(3),table.reg th:nth-child(4),table.reg td:nth-child(4),
table.reg th:nth-child(5),table.reg td:nth-child(5){display:none}
.cite{padding:var(--s5) var(--s5) var(--s4)}.cards{gap:var(--s4)}}
@media print{
html,body{background:#fff}.rail{display:none}main{grid-column:1/13}.page{padding:0}
.cite,.synth{break-inside:avoid}.cite:hover{transform:none;box-shadow:none}.plate{padding-bottom:var(--s7)}
.notched::before,.notched::after{display:none}}
"""


def answer_html(answer: str, qid: str, n_cites: int) -> str:
    text = html.escape(answer.strip())

    def link(m):
        out = []
        for n in (int(x) for x in re.findall(r"\d+", m.group(1))):
            if 1 <= n <= n_cites:
                out.append(f'<a class="ref" href="#{qid}c{n}">{n}</a>')
            else:
                out.append(f'<span class="ref bad" title="No such passage">{n}</span>')
        return "".join(out)

    text = _CITE_GROUP.sub(link, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    out = []
    for b in blocks:
        lines = b.strip().split("\n")
        if all(re.match(r"\s*[-*•]\s+", ln) for ln in lines):
            items = "".join("<li>" + re.sub(r"^\s*[-*•]\s+", "", ln) + "</li>" for ln in lines)
            out.append("<ul>" + items + "</ul>")
        else:
            out.append("<p>" + "<br>".join(lines) + "</p>")
    return "".join(out)


def cite_html(qid: str, c: Citation, multi: bool) -> str:
    e = html.escape
    tail = " …" if c.truncated else ""
    src = f'<p class="src">{e(c.doc)}</p>' if multi else ""
    return (
        f'<article class="cite notched" id="{qid}c{c.n}">'
        f'<header><div class="cite-top"><span class="idx">[{c.n}]</span>'
        f'<span class="tag" title="Section heading">{e(c.section)}</span></div>{src}</header>'
        f'<dl class="anchors"><div><dt>Page</dt><dd>{c.page}</dd></div>'
        f"<div><dt>Paragraph</dt><dd>¶ {c.paragraph}</dd></div></dl>"
        f"<blockquote>{e(c.quote)}{tail}</blockquote>"
        f'<div class="foot"><div class="match"><span class="label">Match</span>'
        f'<span class="track"><i style="width:{c.score:.1f}%"></i></span>'
        f'<span class="pct">{c.score:.1f}%</span></div>'
        f'<details><summary>Full passage</summary><p>{e(c.text)}</p></details></div></article>')


def qa_status(qa: QA, generated: bool):
    """(css class, label) for the discreet status pill beside each synthesis."""
    if not qa.citations:
        return "warn", "No evidence"
    if qa.warnings:
        return "warn", "Check citations"
    if not generated:
        return "", "Retrieval only"
    return "ok", "Citations valid"


def render_html(run: Run) -> str:
    e = html.escape
    multi = len(run.docs) > 1
    titles = " · ".join(d.title for d in run.docs)
    engine = f"{run.llm} · {run.embed_model}" if run.generated_text else f"Retrieval only · {run.embed_model}"
    passages = sum(d.chunks for d in run.docs)
    paragraphs = sum(d.paragraphs for d in run.docs)
    synth_label = f"Synthesis · {run.llm}" if run.generated_text else "Retrieved evidence"

    rail = "".join(
        f'<li><a href="#q{i}"><span class="label">{i:02d}</span><span class="rt">{e(qa.question)}</span></a></li>'
        for i, qa in enumerate(run.qas, 1))
    rows = "".join(
        f'<tr><td class="n">{i:02d}</td><td>{e(d.title)}</td><td class="f">{e(d.name)}</td>'
        f'<td class="k">{e(d.kind)}</td>'
        f'<td><span class="status{"" if d.cached else " ok"}"><i></i>{"Cached" if d.cached else "Indexed"}</span></td>'
        f'<td class="n">{d.pages}</td><td class="n">{d.paragraphs}</td><td class="n">{d.chunks}</td></tr>'
        for i, d in enumerate(run.docs, 1))
    warn = "".join(
        f'<p class="warn-line">† {e(d.name)}: no extractable text on page(s) '
        f'{e(", ".join(map(str, d.empty_pages[:12])))}. Scanned pages need OCR first.</p>'
        for d in run.docs if d.empty_pages)
    warn += "".join(
        f'<p class="warn-line">† {e(name)}: skipped, no extractable text (scanned PDF?). '
        "Run OCR first, e.g. ocrmypdf.</p>" for name in run.skipped)

    plates = []
    for i, qa in enumerate(run.qas, 1):
        qid = f"q{i}"
        notes = "".join(f'<p class="note">† {e(w)}</p>' for w in qa.warnings)
        n = len(qa.citations)
        plural = "" if n == 1 else "s"
        cards = "".join(cite_html(qid, c, multi) for c in qa.citations)
        evidence = (f'<div class="evidence"><p class="label">Evidence // {n} passage{plural}</p>'
                    f'<div class="cards">{cards}</div></div>' if n else "")
        cls, text = qa_status(qa, run.generated_text)
        pill = f'<span class="status {cls}"><i></i>{text}</span>'
        plates.append(
            f'<section class="plate" id="{qid}"><header class="plate-head">'
            f'<span class="qn">{i:02d}<i>//</i></span><h2>{e(qa.question)}</h2></header>'
            f'<div class="synth"><div class="synth-head"><p class="label">{e(synth_label)}</p>{pill}</div>'
            f'<div class="answer">{answer_html(qa.answer, qid, n)}</div></div>'
            f"{notes}{evidence}</section>")

    when = run.generated.strftime("%d %b %Y") + " · " + run.generated.strftime("%H:%M")
    meta = [("Engine", engine), ("Generated", when),
            ("Documents", f"{len(run.docs):02d}"), ("Index", f"{paragraphs} ¶ · {passages} passages")]
    meta_html = "".join(f"<div><dt>{k}</dt><dd>{e(v)}</dd></div>" for k, v in meta)
    stamp = f"<b>Nº {run.generated:%y%m%d-%H%M}</b>{run.generated:%d %B %Y}"
    swatches = [("#F5F3EE", "Linen"), ("#EFECE6", "Parchment"), ("#191817", "Charcoal"),
                ("#6F6A61", "Slate"), ("#C4BEB4", "Hairline")]
    sw = "".join(f'<span><i style="background:{h}"></i>{name} {h[1:]}</span>' for h, name in swatches)

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; "
        "img-src data:\">"
        '<link rel="icon" href="data:,">'
        "<title>Étude — Analysis</title><style>" + CSS + "</style></head><body><div class=\"page\">"
        '<div class="topbar grid"><span class="label wm">Étude</span>'
        '<span class="label mid">Local · Offline · Reading report</span>'
        f'<span class="end"><span class="status ok"><i></i>Offline</span><span class="label">{run.generated:%Y}</span></span></div>'
        '<header class="hero grid"><h1>Analysis</h1>'
        f'<p class="label stamp">{stamp}</p>'
        f'<p class="sub">{e(titles)}</p></header>'
        f'<dl class="meta">{meta_html}</dl>'
        '<div class="regwrap"><table class="reg"><thead><tr><th class="n">Nº</th><th>Title</th><th>File</th>'
        '<th>Kind</th><th>Status</th><th class="n">Pages</th><th class="n">¶</th><th class="n">Passages</th>'
        f"</tr></thead><tbody>{rows}</tbody></table></div>{warn}"
        f'<div class="frame grid"><nav class="rail"><p class="label">Questions</p><ol>{rail}</ol></nav>'
        f"<main>{''.join(plates)}</main></div>"
        "<footer><p>Produced offline by pdf_analyzer.py. <strong>Match</strong> is the cosine similarity "
        "between the question and the passage embedding, times 100: a ranking signal, not a probability "
        "of correctness. <strong>Citations valid</strong> means every [n] marker in the synthesis points at a "
        "retrieved passage; it does not verify the claim. Quotes are literal passages of the extracted text, "
        "with line-wrap hyphens and ligatures normalised. A small local model can still misread its sources; "
        "the quotes and page references are retrieved, not generated, and are the ground truth.</p>"
        f'<div class="sw label">{sw}</div></footer>'
        "</div></body></html>")


# Alias so callers can use either name.
generate_html_report = render_html


# --------------------------------------------------------------------------- #
# Terminal output
# --------------------------------------------------------------------------- #
def fail(msg: str):
    console.print(f"[bold]error[/bold]  {escape(msg)}")
    sys.exit(1)


def show_qa(i: int, qa: QA):
    console.print()
    console.print(Panel(escape(qa.answer), title=f"{i:02d}  {escape(qa.question)}", title_align="left",
                        box=box.HORIZONTALS, border_style=SLATE, padding=(1, 2)))
    for w in qa.warnings:
        console.print(f"[{SLATE}]\u2020 {escape(w)}[/]")
    if not qa.citations:
        return
    t = Table(box=box.SIMPLE_HEAD, header_style=SLATE, pad_edge=False)
    for col, just in (("Ref", "left"), ("Page", "right"), ("\u00b6", "right"),
                      ("Section", "left"), ("Match", "right"), ("Quote", "left")):
        t.add_column(col, justify=just, overflow="ellipsis", no_wrap=(col != "Quote"),
                     max_width=28 if col == "Section" else None)
    for c in qa.citations:
        quote = c.quote if len(c.quote) <= 90 else c.quote[:87].rstrip() + "..."
        t.add_row(f"[{c.n}]".replace("[", "\\["), str(c.page), str(c.paragraph), escape(c.section),
                  f"{c.score:.1f}%", escape(f"\u201c{quote}\u201d"))
    console.print(t)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
DEFAULT_QUESTIONS = {
    "paper": [
        "What research question or problem does this paper address, and what is its main contribution?",
        "What methodology, data and evaluation approach does the paper use?",
        "What are the key findings, and what limitations do the authors acknowledge?"],
    "syllabus": [
        "What are the course's learning outcomes and objectives?",
        "How is the final grade calculated, and what are the assessment weights?",
        "What are the key deadlines, policies and required readings?"],
}


@dataclass
class Ctx:
    col: object
    emb: Embedder
    client: object
    model: str
    k: int
    min_score: float
    no_llm: bool
    quiet: bool = False


def answer_question(question: str, kind: str, doc_ids, ctx: Ctx) -> QA:
    cites = retrieve(ctx.col, ctx.emb, question, ctx.k, doc_ids, ctx.min_score)
    if not cites:
        return QA(question, kind, "No passage cleared the similarity threshold.")
    if ctx.no_llm:
        return QA(question, kind, "Retrieval only: generation was skipped (--no-llm).", cites)
    try:
        spinner = (contextlib.nullcontext() if ctx.quiet
                   else console.status(f"[{SLATE}]Reading locally with {ctx.model}..."))
        with spinner:
            answer = generate(ctx.client, ctx.model, question, cites, kind)
        return QA(question, kind, answer, cites, audit_answer(answer, len(cites)))
    except Exception as exc:  # keep the evidence even if generation fails
        return QA(question, kind, "Generation failed; the retrieved evidence is listed below.",
                  cites, [f"Ollama error: {exc}"])


def plan_questions(args, docs):
    if args.question:
        kinds = {d.kind for d in docs}
        kind = next(iter(kinds)) if len(kinds) == 1 else "document"
        return [(q, kind, [d.doc_id for d in docs]) for q in args.question]
    if args.interactive:
        return []
    plan = []
    for kind in dict.fromkeys(d.kind for d in docs):
        ids = [d.doc_id for d in docs if d.kind == kind]
        plan += [(q, kind, ids) for q in DEFAULT_QUESTIONS[kind]]
    return plan


def ollama_status(client, model: str, embed_model: str) -> dict:
    """Never raises: describes what the local Ollama can do right now."""
    try:
        names = model_names(client.list())
    except Exception as exc:
        return {"online": False, "error": str(exc), "chat_ok": False, "embed_ok": False, "fallback": False}
    chat_ok, embed_ok = has_model(names, model), has_model(names, embed_model)
    return {"online": True, "error": "", "chat_ok": chat_ok, "embed_ok": embed_ok,
            "fallback": chat_ok and not embed_ok}


def resolve_models(client, model: str, embed_model: str, no_llm: bool, notify=None, host: str = "") -> str:
    """Return the embedding model to use, or raise EtudeError with the fix."""
    st = ollama_status(client, model, embed_model)
    if not st["online"]:
        where = f" at {host}" if host else ""
        raise EtudeError(f"Cannot reach Ollama{where}. Start it with `ollama serve`. ({st['error']})")
    if not no_llm and not st["chat_ok"]:
        raise EtudeError(f"Model '{model}' is not installed. Run: ollama pull {model}")
    if st["embed_ok"]:
        return embed_model
    if st["chat_ok"]:
        if notify:
            notify(f"'{embed_model}' is not installed; falling back to '{model}' for embeddings "
                   f"(weaker retrieval). For better results: ollama pull {embed_model}")
        return model
    raise EtudeError(f"Embedding model '{embed_model}' is not installed. Run: ollama pull {embed_model}")


def preflight(client, host: str, args) -> str:
    try:
        return resolve_models(client, args.model, args.embed_model, args.no_llm, host=host,
                              notify=lambda msg: console.print(f"[{SLATE}]{escape(msg)}[/]"))
    except EtudeError as exc:
        fail(str(exc))


# --------------------------------------------------------------------------- #
# Local web interface ("try it out"): standard library only, loopback only
# --------------------------------------------------------------------------- #
SAFE_ID = re.compile(r"^[0-9a-f]{32}$")
MAX_FILES_PER_RUN = 8
MAX_QUESTIONS = 12
KEEP_JOBS = 12

APP_CSS = r"""
.topbar .mid{grid-column:4/7}.topbar .end{grid-column:7/13;flex-wrap:wrap;justify-content:flex-end;gap:var(--s2)}
.studio{margin-top:var(--s9);align-items:start}
.steps{grid-column:1/6;padding-right:var(--s5)}
.steps ol{list-style:none;border-top:1px solid var(--ink)}
.steps li{display:grid;grid-template-columns:5.4rem minmax(0,1fr);column-gap:var(--gutter);padding:var(--s5) 0;
border-bottom:1px solid var(--rule)}
.qn{font-family:var(--sans);font-size:13px;font-weight:700;letter-spacing:.1em;color:var(--ink);white-space:nowrap;padding-top:.5em}
.qn i{font-style:normal;color:var(--slate);font-weight:400;margin-left:.4em}
.steps h3{font-weight:400;font-size:1.75rem;line-height:1.15;letter-spacing:-.01em}
.steps li p{margin-top:var(--s2);font-size:1rem;line-height:1.65;color:var(--ink-soft)}
.privacy{margin-top:var(--s7);padding:var(--s5) var(--s6);border:1px solid var(--rule)}
.privacy p:not(.label){margin-top:var(--s3);font-family:var(--sans);font-size:12.5px;line-height:1.75;color:var(--ink-soft)}
.panel{grid-column:6/13;background:var(--parchment);border:1px solid var(--rule);border-top-color:var(--ink);
padding:var(--s7)}
.drop{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:var(--s3);text-align:center;
padding:var(--s9) var(--s5);background:var(--linen);border:1px solid var(--rule);cursor:pointer;
transition:background-color .3s var(--ease),border-color .3s}
.drop:hover,.drop.on{background-color:#F8F6F2;border-color:var(--ink)}
.drop-title{font-size:1.9rem;line-height:1.15;letter-spacing:-.01em;transition:font-style .2s}
.drop.on .drop-title{font-style:italic}
.plus{position:relative;width:28px;height:28px;margin-bottom:var(--s2);transition:transform .6s var(--ease)}
.plus::before,.plus::after{content:"";position:absolute;background:var(--ink)}
.plus::before{left:0;right:0;top:50%;height:1px}.plus::after{top:0;bottom:0;left:50%;width:1px}
.drop:hover .plus,.drop.on .plus{transform:rotate(90deg)}
.files{list-style:none;margin-top:var(--s5)}
.file{position:relative;display:grid;grid-template-columns:2rem minmax(0,1fr) auto auto;align-items:center;
column-gap:var(--s4);padding:var(--s4) 0;border-bottom:1px solid var(--rule)}
.fname{display:block;font-size:1.1rem;line-height:1.35;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fmeta{display:block;font-family:var(--sans);font-size:11.5px;color:var(--slate)}
.file.error .fmeta{color:var(--ink)}
.rm{font-family:var(--sans);background:none;border:0;font-size:20px;line-height:1;color:var(--slate);cursor:pointer;
padding:var(--s1) var(--s2);transition:color .2s,transform .4s var(--ease)}
.rm:hover{color:var(--ink);transform:rotate(90deg)}
.fbar{position:absolute;left:0;right:0;bottom:-1px;height:1px}
.fbar i{display:block;height:3px;background:var(--ink);margin-top:-1px;transition:width .2s}
.file.ready .fbar,.file.error .fbar{display:none}
.field{margin-top:var(--s7)}
.field-head{display:flex;justify-content:space-between;gap:var(--s4);margin-bottom:var(--s3)}
textarea{display:block;width:100%;min-height:132px;resize:vertical;font-family:var(--serif);font-size:1.12rem;
line-height:1.65;color:var(--ink);background:var(--linen);border:1px solid var(--rule);padding:var(--s4) var(--s5);
border-radius:0;transition:border-color .2s,box-shadow .2s}
textarea::placeholder{color:var(--slate);font-style:italic}
textarea:focus{outline:none;border-color:var(--ink);box-shadow:0 0 0 3px rgba(25,24,23,.06)}
.chips{display:flex;flex-wrap:wrap;gap:var(--s2);margin-top:var(--s3)}
.chip{font-family:var(--sans);font-size:11.5px;line-height:1.3;color:var(--ink-soft);background:transparent;
border:1px solid var(--rule);border-radius:999px;padding:5px 12px;cursor:pointer;
transition:background-color .2s,color .2s,border-color .2s}
.chip:hover{background:var(--ink);color:var(--linen);border-color:var(--ink)}
.opts{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:var(--s5);margin-top:var(--s6);
padding-top:var(--s5);border-top:1px solid var(--rule)}
fieldset{border:0;min-width:0}
.seg{display:inline-flex;margin-top:var(--s2);border:1px solid var(--ink);position:relative}
.seg label{cursor:pointer}
.seg input{position:absolute;opacity:0;pointer-events:none}
.seg span{display:block;min-width:46px;text-align:center;padding:6px 14px;font-family:var(--sans);font-size:12px;
font-weight:600;font-variant-numeric:tabular-nums;transition:background-color .2s,color .2s}
.seg label+label span{border-left:1px solid var(--ink)}
.seg input:checked+span{background:var(--ink);color:var(--linen)}
.seg input:focus-visible+span{outline:1px solid var(--ink);outline-offset:2px}
.check{position:relative;display:inline-flex;align-items:center;gap:var(--s3);cursor:pointer;font-family:var(--sans);
font-size:12.5px;color:var(--ink-soft)}
.check input{position:absolute;opacity:0}
.check .box{width:16px;height:16px;border:1px solid var(--ink);display:inline-block;position:relative;flex:none}
.check input:checked+.box::after{content:"";position:absolute;inset:3px;background:var(--ink)}
.check input:focus-visible+.box{outline:1px solid var(--ink);outline-offset:2px}
.check em{color:var(--slate);font-style:normal}
.actions{display:flex;align-items:center;flex-wrap:wrap;gap:var(--s4) var(--s5);margin-top:var(--s6)}
.go{display:inline-flex;align-items:center;gap:var(--s4);font-family:var(--sans);font-size:11px;font-weight:600;
letter-spacing:.18em;text-transform:uppercase;color:var(--linen);background:var(--ink);border:1px solid var(--ink);
padding:var(--s4) var(--s6);cursor:pointer;transition:background-color .3s var(--ease),color .3s var(--ease)}
.go i{font-style:normal;transition:transform .3s var(--ease)}
.go:hover:not(:disabled){background:transparent;color:var(--ink)}
.go:hover:not(:disabled) i{transform:translateX(5px)}
.go:disabled{opacity:.35;cursor:not-allowed}
.hint{font-family:var(--sans);font-size:12px;line-height:1.55;color:var(--slate);flex:1;min-width:14rem}
.run,.viewer{margin-top:var(--s9);border-top:1px solid var(--ink);padding-top:var(--s6);position:relative}
.run::before,.viewer::before{content:"";position:absolute;top:-2px;left:0;width:48px;height:3px;background:var(--ink)}
.run-head,.viewer-head{display:flex;align-items:center;justify-content:space-between;gap:var(--s4);flex-wrap:wrap}
.stages{list-style:none;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--gutter);margin-top:var(--s6)}
.stages li{padding-top:var(--s3);border-top:1px solid var(--rule);color:var(--slate);
transition:color .3s,border-color .3s}
.stages li b{display:block;margin-top:var(--s1);font-weight:400;font-size:1.35rem;line-height:1.2;color:inherit}
.stages li.done{color:var(--ink);border-top-color:var(--ink)}
.stages li.active{color:var(--ink);border-top:3px solid var(--ink);margin-top:-2px}
.stages li.active b{font-style:italic}
.bar{height:1px;background:var(--rule);margin-top:var(--s6);position:relative}
.bar i{position:absolute;left:0;top:-1px;height:3px;background:var(--ink);width:0;transition:width .5s var(--ease)}
.run-msg{margin-top:var(--s4);font-family:var(--sans);font-size:12.5px;line-height:1.6;color:var(--ink-soft);
overflow-wrap:anywhere}
.links{display:flex;flex-wrap:wrap;gap:var(--s2) var(--s5);align-items:center}
.links a,.links button{font-family:var(--sans);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;
font-weight:500;color:var(--ink);text-decoration:none;background:none;border:0;border-bottom:1px solid var(--ink);
padding:0 0 2px;cursor:pointer;transition:color .2s,border-color .2s}
.links a:hover,.links button:hover{color:var(--slate);border-color:var(--slate)}
.warn-line{margin-top:var(--s4);font-family:var(--sans);font-size:12px;color:var(--slate)}
iframe{display:block;width:100%;height:min(86vh,1200px);margin-top:var(--s5);border:1px solid var(--rule);
background:var(--linen)}
.preview{margin-bottom:var(--s6);padding:var(--s5) var(--s6);background:var(--linen);border:1px solid var(--rule)}
.preview p:not(.label){margin-top:var(--s3);font-size:1rem;line-height:1.65;color:var(--ink-soft)}
.cmd{display:flex;align-items:center;justify-content:space-between;gap:var(--s4);margin-top:var(--s4);padding:var(--s3) var(--s4);border:1px solid var(--ink);background:var(--parchment)}
.cmd code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:13px;overflow-wrap:anywhere}
.drop.off{opacity:.5;cursor:not-allowed;pointer-events:none}
.preview .more{margin-top:var(--s4)}
.preview .more a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--ink);padding-bottom:2px;transition:color .2s,border-color .2s}
.preview .more a:hover{color:var(--slate);border-color:var(--slate)}
textarea:disabled{opacity:.6;cursor:not-allowed}
/* ---- working-surface polish: compact hero, sticky action bar, run controls, history, idle ghost ---- */
.hero{padding-top:var(--s7)}
h1{font-size:clamp(3rem,6.4vw,5.8rem);line-height:.9}
.sub{margin-top:var(--s4);font-size:1.12rem}
.studio{margin-top:var(--s7)}
.panel{padding:var(--s6) var(--s7) var(--s7)}
.drop{padding:var(--s7) var(--s5)}
.field{margin-top:var(--s6)}
textarea{min-height:96px}
.opts{margin-top:var(--s5);padding-top:var(--s4)}
.actions{position:sticky;bottom:0;z-index:3;margin:var(--s5) calc(-1*var(--s7)) 0;padding:var(--s4) var(--s7);
background:var(--parchment);border-top:1px solid var(--rule)}
.static .actions{position:static}
html{scroll-padding-bottom:96px}
.hint .chip{margin-left:var(--s3)}
.run-head .links{gap:var(--s4)}
.run-head .links button:disabled{opacity:.45;cursor:default}
.runq{margin-top:var(--s5);font-size:1.3rem;line-height:1.45;font-style:italic;max-width:56ch;text-wrap:pretty}
.runq:empty{display:none}
.runtime{margin-top:var(--s3);font-variant-numeric:tabular-nums}
.recent{margin-top:var(--s8)}
.recent ol{list-style:none;margin-top:var(--s4);border-top:1px solid var(--ink)}
.recent li{display:grid;grid-template-columns:8rem minmax(0,1fr) auto;column-gap:var(--gutter);align-items:baseline;
padding:var(--s4) 0;border-bottom:1px solid var(--rule)}
.recent .when{font-family:var(--sans);font-size:11.5px;color:var(--slate);font-variant-numeric:tabular-nums}
.recent .what{font-size:1.05rem;line-height:1.4;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.recent .what small{display:block;font-family:var(--sans);font-size:11.5px;color:var(--slate)}
.ghost{margin-top:var(--s8);padding:var(--s6);border:1px dashed var(--rule)}
.gl{height:9px;margin-top:var(--s3);background:var(--rule);opacity:.5}
.gl.w6{width:60%}.gl.w8{width:80%}.gl.w9{width:90%}.gl.w10{width:100%}
.gcards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:var(--gutter);margin-top:var(--s5)}
.gcard{padding:var(--s4);border:1px solid var(--rule)}
.gcard .status{opacity:.6}
[hidden]{display:none!important}
footer{margin-top:var(--s9);padding-top:var(--s6);border-top:1px solid var(--ink);position:relative}
footer::before{content:"";position:absolute;top:-2px;left:0;width:48px;height:3px;background:var(--ink)}
footer p{max-width:72ch;font-family:var(--sans);font-size:12px;line-height:1.75;color:var(--slate)}
@media(max-width:960px){
.topbar .mid{display:none}.topbar .end{grid-column:1/13;justify-content:flex-start;margin-top:var(--s3)}
.topbar .wm{grid-column:1/13}
.studio{margin-top:var(--s8)}.steps,.panel{grid-column:1/13}.steps{padding:0 0 var(--s7)}
.panel{padding:var(--s5);order:-1}.drop{padding:var(--s6) var(--s4)}
.hero{padding-top:var(--s6)}.studio{margin-top:var(--s6)}
.actions{margin:var(--s5) calc(-1*var(--s5)) 0;padding:var(--s3) var(--s5)}
.recent li{grid-template-columns:minmax(0,1fr) auto}.recent .when{grid-column:1/3}
.gcards{grid-template-columns:1fr}
.stages{grid-template-columns:repeat(2,minmax(0,1fr));row-gap:var(--s5)}
.file{column-gap:var(--s3)}.steps li{grid-template-columns:4.2rem minmax(0,1fr)}
.run,.viewer{margin-top:var(--s8)}iframe{height:80vh}}
"""

APP_BODY = r"""
<div class="page">
<div class="topbar grid"><span class="label wm">&Eacute;tude</span>
<span class="label mid">Local &middot; Offline &middot; Reading room</span>
<span class="end" id="pills"></span></div>
<header class="hero grid"><h1>Ask the page</h1>
<p class="label stamp"><b>Try it out</b>Drop &middot; Ask &middot; Read</p>
<p class="sub">Drop in a paper or a syllabus and ask it anything. Every answer comes back with the page, paragraph and section it was found on.</p></header>

<div class="studio grid">
<aside class="steps">
<ol>
<li><span class="qn">01<i>//</i></span><div><h3>Drop</h3><p>PDF papers or syllabi, up to <span id="maxmb">100</span> MB each. Text-based PDFs only; scans need OCR first.</p></div></li>
<li><span class="qn">02<i>//</i></span><div><h3>Ask</h3><p>One question per line, or leave it blank for sensible defaults.</p></div></li>
<li><span class="qn">03<i>//</i></span><div><h3>Read</h3><p>A typeset report: page, paragraph, section heading, the exact quote and a match score.</p></div></li>
</ol>
<div class="privacy notched"><p class="label">Private by construction</p>
<p>This page is served from 127.0.0.1 and talks only to your local Ollama. No accounts, no credits, no external requests. Uploaded files are deleted as soon as each run finishes.</p></div>
</aside>

<section class="panel notched" aria-label="Upload and ask">
<div class="preview notched" id="preview" hidden>
<p class="label">Static preview // not connected</p>
<p>This copy is published on GitHub Pages, which can&rsquo;t run Python or Ollama, so uploads are switched off here. To try it for real, run &Eacute;tude on your own computer:</p>
<div class="cmd"><code id="cmd">python pdf_analyzer.py --serve --open</code><button type="button" id="copycmd" class="chip">Copy</button></div>
<p class="label more"><a href="report.html">View the sample report &rarr;</a></p>
</div>
<label class="drop notched" id="drop" for="pick" tabindex="0">
<span class="plus" aria-hidden="true"></span>
<span class="drop-title">Drop PDFs here</span>
<span class="label">or click to browse</span>
</label>
<input id="pick" type="file" accept="application/pdf,.pdf" multiple hidden>
<ul class="files" id="files" aria-live="polite"></ul>

<div class="field">
<div class="field-head"><label class="label" for="qs">Questions // optional</label><span class="label" id="qcount"></span></div>
<textarea id="qs" rows="4" placeholder="One question per line. Leave blank for default questions suited to papers and syllabi."></textarea>
<div class="chips" id="chips"></div>
</div>

<div class="opts">
<fieldset><legend class="label">Passages per question</legend>
<div class="seg" role="radiogroup">
<label><input type="radio" name="k" value="3"><span>3</span></label>
<label><input type="radio" name="k" value="5" checked><span>5</span></label>
<label><input type="radio" name="k" value="8"><span>8</span></label>
</div></fieldset>
<label class="check"><input type="checkbox" id="nollm"><span class="box"></span><span>Retrieval only <em>(skip the language model)</em></span></label>
</div>

<div class="actions">
<button class="go" id="go" type="button" disabled><span>Analyse</span><i>&rarr;</i></button>
<p class="hint" id="hint"></p>
</div>
</section>
</div>

<section class="run" id="run" hidden aria-live="polite">
<div class="run-head"><p class="label">Reading // in progress</p>
<div class="links"><span class="status" id="runstatus"><i></i>Working</span><button type="button" id="cancel" disabled>Cancel</button></div></div>
<ol class="stages" id="stages"></ol>
<div class="bar"><i id="barfill"></i></div>
<p class="runq" id="runq"></p>
<p class="run-msg" id="runmsg"></p>
<p class="label runtime" id="runtime"></p>
</section>

<section class="viewer" id="viewer" hidden>
<div class="viewer-head"><p class="label" id="vlabel">Report // ready</p>
<div class="links"><a id="open" target="_blank" rel="noopener">Open full report &#8599;</a>
<a id="dlhtml" download>report.html</a><a id="dlmd" download>RESULTS.md</a>
<button type="button" id="again">Analyse another</button></div></div>
<div class="warn-line" id="skipped" hidden></div>
<iframe id="frame" title="&Eacute;tude report" sandbox="allow-same-origin"></iframe>
</section>

<section class="ghost notched" id="ghost" aria-hidden="true">
<p class="label">Your report // will appear here</p>
<div class="gl w9"></div><div class="gl w10"></div><div class="gl w6"></div>
<div class="gcards">
<div class="gcard"><span class="status"><i></i>Page 3</span><div class="gl w10"></div><div class="gl w8"></div></div>
<div class="gcard"><span class="status"><i></i>Page 7</span><div class="gl w10"></div><div class="gl w6"></div></div>
</div></section>

<section class="recent" id="recent" hidden>
<p class="label">Recent readings // kept on this computer</p>
<ol id="recentlist"></ol>
</section>

<footer><p>Produced offline by pdf_analyzer.py. Nothing on this page loads from the network; the report you generate is a single self-contained file you can keep, print or share.</p></footer>
</div>
"""

APP_JS = r"""
(() => {
"use strict";
const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
const pad = (n) => String(n).padStart(2, "0");
const fmtSize = (b) => b < 1048576 ? Math.max(1, Math.round(b / 1024)) + " KB" : (b / 1048576).toFixed(1) + " MB";
const SUGGEST = ["What is the main contribution?", "Which methods and data are used?",
  "What limitations do the authors acknowledge?", "How is the final grade calculated?",
  "What are the key deadlines and policies?"];
const STAGES = [["reading", "Read"], ["embedding", "Embed"], ["answering", "Answer"], ["composing", "Typeset"]];
const fmtT = (ms) => { const t = Math.max(0, Math.round(ms / 1000)); return Math.floor(t / 60) + ":" + pad(t % 60); };
const S = { files: [], status: null, busy: false, maxMB: 100, t0: 0, pct: 0, job: null, ticker: null,
  static: !["127.0.0.1", "localhost"].includes(location.hostname) };

const drop = $("#drop"), pick = $("#pick"), list = $("#files"), qs = $("#qs"), go = $("#go"), hint = $("#hint"),
  nollm = $("#nollm"), pills = $("#pills"), runBox = $("#run"), viewer = $("#viewer"), ghost = $("#ghost"),
  cancelBtn = $("#cancel");
async function copyText(text, btn, label) {
  try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; }
  catch (_) { btn.textContent = "Select and copy"; }
  setTimeout(() => { btn.textContent = label; }, 1800);
}

/* ---------- status pills ---------- */
function pill(text, kind) { const s = el("span", "status " + kind); s.append(el("i")); s.append(document.createTextNode(text)); return s; }
function renderStatus() {
  pills.replaceChildren();
  const s = S.status;
  if (S.static) { pills.append(pill("Static preview", "")); return; }
  if (!s) { pills.append(pill("Server unreachable", "warn")); return; }
  pills.append(pill(s.online ? "Ollama online" : "Ollama offline", s.online ? "ok" : "warn"));
  if (s.online) {
    pills.append(pill(s.model + (s.chat_ok ? " ready" : " missing"), s.chat_ok ? "ok" : "warn"));
    pills.append(s.embed_ok ? pill(s.embed_model + " ready", "ok")
      : s.fallback ? pill("embeddings via " + s.model, "") : pill(s.embed_model + " missing", "warn"));
  }
  $("#maxmb").textContent = s.max_upload_mb;
}
async function loadStatus() {
  if (S.static) { S.status = null; renderStatus(); renderGo(); return; }
  try { const r = await fetch("/api/status", { cache: "no-store" }); S.status = await r.json(); S.maxMB = S.status.max_upload_mb || S.maxMB; }
  catch (_) { S.status = null; }
  renderStatus(); renderGo();
}

/* ---------- files ---------- */
function addFiles(items) {
  if (S.static) return;
  for (const f of items) {
    const isPdf = f.type === "application/pdf" || /\.pdf$/i.test(f.name);
    const e = { file: f, name: f.name, size: f.size, state: "uploading", pct: 0, id: null, pages: null, error: "" };
    if (!isPdf) { e.state = "error"; e.error = "Only PDF files are supported"; }
    else if (f.size > S.maxMB * 1048576) { e.state = "error"; e.error = "Larger than " + S.maxMB + " MB"; }
    else if (S.files.filter((x) => x.state !== "error").length >= 8) { e.state = "error"; e.error = "At most 8 files per run"; }
    S.files.push(e);
    if (e.state === "uploading") upload(e);
  }
  render();
}
function upload(e) {
  const x = new XMLHttpRequest(); e.xhr = x;
  x.open("POST", "/api/upload");
  x.setRequestHeader("X-Filename", encodeURIComponent(e.name));
  x.setRequestHeader("Content-Type", "application/pdf");
  x.upload.onprogress = (ev) => { if (ev.lengthComputable) { e.pct = ev.loaded / ev.total; renderFiles(); } };
  x.onload = () => {
    let j = {}; try { j = JSON.parse(x.responseText); } catch (_) {}
    if (x.status === 200) { e.state = "ready"; e.id = j.id; e.pages = j.pages; e.name = j.name || e.name; e.pct = 1; }
    else { e.state = "error"; e.error = j.error || "Upload failed"; }
    render();
  };
  x.onerror = () => { if (e.state !== "removed") { e.state = "error"; e.error = "Upload failed"; render(); } };
  x.send(e.file);
}
function removeFile(e) {
  if (e.state === "uploading" && e.xhr) { e.state = "removed"; e.xhr.abort(); }
  if (e.state === "ready" && e.id) fetch("/api/upload/" + e.id, { method: "DELETE" }).catch(() => {});
  S.files = S.files.filter((x) => x !== e); render();
}
function renderFiles() {
  list.replaceChildren();
  S.files.forEach((f, i) => {
    const li = el("li", "file " + f.state);
    li.append(el("span", "label", pad(i + 1)));
    const main = el("div"); main.append(el("span", "fname", f.name));
    main.append(el("span", "fmeta", f.state === "error" ? f.error : [f.pages ? f.pages + (f.pages === 1 ? " page" : " pages") : null, fmtSize(f.size)].filter(Boolean).join(" · ")));
    li.append(main);
    li.append(f.state === "uploading" ? pill(Math.round(f.pct * 100) + "%", "") : f.state === "ready" ? pill("Ready", "ok") : pill("Error", "warn"));
    const rm = el("button", "rm", "×"); rm.type = "button"; rm.setAttribute("aria-label", "Remove " + f.name); rm.onclick = () => removeFile(f); li.append(rm);
    const bar = el("span", "fbar"); const fill = el("i"); fill.style.width = Math.round(f.pct * 100) + "%"; bar.append(fill); li.append(bar);
    list.append(li);
  });
}

/* ---------- go button ---------- */
function questions() { return qs.value.split("\n").map((l) => l.trim()).filter(Boolean); }
function blocker() {
  const s = S.status;
  if (S.static) return "Preview only. To analyse a PDF, run it on your computer (see above).";
  if (S.busy) return "Working…";
  if (!s) return "Cannot reach the local server.";
  if (!s.online) return "Ollama is not running. Start it with: ollama serve";
  if (!nollm.checked && !s.chat_ok) return "Model missing. Run: ollama pull " + s.model;
  if (!s.embed_ok && !s.fallback) return "Embedding model missing. Run: ollama pull " + s.embed_model;
  if (S.files.some((f) => f.state === "uploading")) return "Uploading…";
  if (!S.files.some((f) => f.state === "ready")) return "Add at least one PDF to begin.";
  return "";
}
function fixCmd() {
  const s = S.status;
  if (S.static || S.busy || !s) return "";
  if (!s.online) return "ollama serve";
  if (!nollm.checked && !s.chat_ok) return "ollama pull " + s.model;
  if (!s.embed_ok && !s.fallback) return "ollama pull " + s.embed_model;
  return "";
}
function renderGo() {
  const b = blocker();
  go.disabled = !!b || S.busy;
  const n = questions().length;
  $("#qcount").textContent = n ? n + (n === 1 ? " question" : " questions") : "defaults";
  const s = S.status;
  if (!b && s && !s.embed_ok && s.fallback) hint.textContent = "Ready. Tip: ollama pull " + s.embed_model + " for sharper retrieval.";
  else hint.textContent = b || "Ready. Runs entirely on this machine." + (S.files.length ? " Press Ctrl or Cmd + Enter in the questions box." : "");
  const cmd = fixCmd();
  if (cmd) { const c = el("button", "chip", "Copy command"); c.type = "button"; c.onclick = () => copyText(cmd, c, "Copy command"); hint.append(c); }
}
function render() { renderFiles(); renderGo(); }

/* ---------- analysis ---------- */
function buildStages() {
  const ol = $("#stages"); ol.replaceChildren();
  STAGES.forEach(([key, name], i) => { const li = el("li"); li.dataset.key = key; li.append(el("span", "label", pad(i + 1) + " //")); li.append(el("b", "", name)); ol.append(li); });
}
function renderProgress(j) {
  const idx = STAGES.findIndex(([k]) => k === j.stage);
  document.querySelectorAll("#stages li").forEach((li, i) => {
    li.className = j.state === "done" || i < idx ? "done" : i === idx ? "active" : "";
  });
  $("#barfill").style.width = Math.max(2, Math.min(100, j.pct)) + "%";
  S.pct = j.pct;
  $("#runq").textContent = j.stage === "answering" && j.question ? "\u201c" + j.question + "\u201d" : "";
  $("#runmsg").textContent = j.cancelling ? "Stopping after the current step\u2026" : (j.message || "");
}
function tick() {
  const el_ = Date.now() - S.t0;
  let t = fmtT(el_) + " elapsed";
  if (S.pct >= 10 && el_ > 6000) {
    const left = el_ * (100 - S.pct) / S.pct / 1000;
    t += left < 60 ? " \u00b7 under a minute left (rough)" : " \u00b7 roughly " + Math.round(left / 60) + " min left";
  }
  $("#runtime").textContent = t;
}
function startTick() { stopTick(); S.t0 = Date.now(); S.pct = 0; tick(); S.ticker = setInterval(tick, 1000); }
function stopTick() { if (S.ticker) clearInterval(S.ticker); S.ticker = null; }
async function analyse() {
  if (S.busy) return;
  const ready = S.files.filter((f) => f.state === "ready");
  S.busy = true; renderGo();
  viewer.hidden = true; ghost.hidden = true; runBox.hidden = false; runBox.classList.remove("error");
  cancelBtn.disabled = true; cancelBtn.textContent = "Cancel"; $("#runq").textContent = ""; startTick();
  buildStages(); $("#barfill").style.width = "2%"; $("#runmsg").textContent = "Starting…";
  const st = $("#runstatus"); st.className = "status"; st.replaceChildren(el("i"), document.createTextNode("Working"));
  runBox.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const body = { files: ready.map((f) => f.id), questions: questions(), no_llm: nollm.checked,
      top_k: parseInt(document.querySelector("input[name=k]:checked").value, 10) };
    const r = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Request failed");
    S.files = S.files.filter((f) => f.state !== "ready");   // the server deletes them after the run
    S.job = j.job; cancelBtn.disabled = false;
    render();
    await poll(j.job);
  } catch (err) { fail(String(err.message || err)); }
  stopTick(); cancelBtn.disabled = true; S.job = null;
  S.busy = false; renderGo(); loadRecent();
}
function cancelled() {
  const st = $("#runstatus"); st.className = "status warn"; st.replaceChildren(el("i"), document.createTextNode("Cancelled"));
  $("#runq").textContent = "";
  $("#runmsg").textContent = "Cancelled. Nothing was saved, and your PDFs were deleted. Add them again to re-run.";
  ghost.hidden = false;
}
function fail(msg) {
  runBox.classList.add("error");
  const st = $("#runstatus"); st.className = "status warn"; st.replaceChildren(el("i"), document.createTextNode("Stopped"));
  $("#runmsg").textContent = msg;
}
async function poll(id) {
  for (;;) {
    const r = await fetch("/api/job/" + id, { cache: "no-store" });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Lost track of the job");
    renderProgress(j);
    if (j.state === "done") { showReport(j); return; }
    if (j.state === "error") { fail(j.error || "Analysis failed"); return; }
    if (j.state === "cancelled") { cancelled(); return; }
    await new Promise((res) => setTimeout(res, 600));
  }
}
function showReport(j) {
  const base = "/report/" + j.id + "/";
  $("#frame").src = base + "report.html";
  $("#open").href = base + "report.html";
  $("#dlhtml").href = base + "report.html?download=1";
  $("#dlmd").href = base + "RESULTS.md?download=1";
  const sk = $("#skipped");
  if (j.skipped && j.skipped.length) { sk.hidden = false; sk.textContent = "† Skipped (no extractable text, scanned?): " + j.skipped.join(", "); } else sk.hidden = true;
  const st = $("#runstatus"); st.className = "status ok"; st.replaceChildren(el("i"), document.createTextNode("Done"));
  if (S.ticker) $("#runtime").textContent = "Finished in " + fmtT(Date.now() - S.t0);
  $("#vlabel").textContent = "Report // ready"; ghost.hidden = true;
  viewer.hidden = false;
  setTimeout(() => viewer.scrollIntoView({ behavior: "smooth", block: "start" }), 250);
}

/* ---------- recent readings ---------- */
async function loadRecent() {
  if (S.static) return;
  let list_ = [];
  try { const r = await fetch("/api/reports", { cache: "no-store" }); list_ = (await r.json()).reports || []; } catch (_) {}
  const box = $("#recent"), ol = $("#recentlist");
  ol.replaceChildren(); box.hidden = !list_.length;
  list_.forEach((r) => {
    const d = new Date(r.created * 1000), li = el("li");
    li.append(el("span", "when", d.toLocaleDateString(undefined, { day: "numeric", month: "short" }) + ", " +
      d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })));
    const w = el("span", "what", r.files.join(", ") || "Report");
    w.append(el("small", "", r.questions + (r.questions === 1 ? " question" : " questions")));
    li.append(w);
    const links = el("span", "links");
    const view = el("button", "", "View"); view.type = "button"; view.onclick = () => showReport({ id: r.id });
    const del = el("button", "", "Remove"); del.type = "button";
    del.onclick = async () => { try { await fetch("/api/report/" + r.id, { method: "DELETE" }); } catch (_) {} loadRecent(); };
    links.append(view, del); li.append(links); ol.append(li);
  });
}

/* ---------- wiring ---------- */
cancelBtn.addEventListener("click", async () => {
  if (!S.job) return;
  cancelBtn.disabled = true; cancelBtn.textContent = "Stopping\u2026";
  try { await fetch("/api/job/" + S.job + "/cancel", { method: "POST" }); } catch (_) {}
});
qs.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !go.disabled) { e.preventDefault(); analyse(); } });
SUGGEST.forEach((t) => { const b = el("button", "chip", t); b.type = "button"; b.onclick = () => {
  if (!questions().includes(t)) qs.value = (qs.value.trim() ? qs.value.trim() + "\n" : "") + t; renderGo(); qs.focus(); }; $("#chips").append(b); });
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("on"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("on"); }));
drop.addEventListener("drop", (e) => { if (e.dataTransfer && e.dataTransfer.files.length) addFiles(Array.from(e.dataTransfer.files)); });
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick.click(); } });
pick.addEventListener("change", () => { addFiles(Array.from(pick.files)); pick.value = ""; });
// a missed drop must not navigate the tab to the PDF
["dragover", "drop"].forEach((ev) => window.addEventListener(ev, (e) => e.preventDefault()));
qs.addEventListener("input", renderGo); nollm.addEventListener("change", renderGo);
go.addEventListener("click", analyse);
$("#again").addEventListener("click", () => { viewer.hidden = true; runBox.hidden = true; ghost.hidden = false; window.scrollTo({ top: 0, behavior: "smooth" }); });
if (S.static) {
  document.body.classList.add("static"); ghost.hidden = true;
  // show the sample report that CI built, right on the page
  fetch("report.html", { method: "HEAD" }).then((r) => {
    if (!r.ok) { ghost.hidden = false; return; }
    $("#frame").src = "report.html"; $("#open").href = "report.html"; $("#dlhtml").href = "report.html";
    $("#dlmd").hidden = true; $("#again").hidden = true;
    $("#vlabel").textContent = "Sample report // built from a demo syllabus";
    viewer.hidden = false;
  }).catch(() => { ghost.hidden = false; });
  $("#preview").hidden = false; drop.classList.add("off"); drop.removeAttribute("tabindex");
  pick.disabled = true; qs.disabled = true; nollm.disabled = true;
  $("#chips").hidden = true;
  document.querySelector(".privacy p:not(.label)").textContent = "Run on your own computer, this page is served from 127.0.0.1 and talks only to your local Ollama. No accounts, no credits, no external requests. Uploaded files are deleted as soon as each run finishes.";
  $("#copycmd").addEventListener("click", () => copyText($("#cmd").textContent, $("#copycmd"), "Copy"));
}
render(); loadStatus(); loadRecent(); setInterval(loadStatus, 15000);
})();
"""

APP_PAGE = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    '<meta name="color-scheme" content="light"><link rel="icon" href="data:,">'
    "<title>Étude — Ask the page</title><style>" + TOKENS + LAYOUT + APP_CSS + "</style></head><body>"
    + APP_BODY + '<script nonce="__NONCE__">' + APP_JS + "</script></body></html>")


def safe_pdf_name(raw: str) -> str:
    name = Path(raw.replace("\\", "/")).name
    stem = re.sub(r"[^\w.\- ()\[\]]+", "_", Path(name).stem).strip(" ._") or "document"
    return stem[:100] + ".pdf"


def pdf_page_count(path: Path) -> int:
    try:
        with fitz.open(str(path)) as doc:
            if getattr(doc, "needs_pass", False):
                raise ValueError("Password-protected PDFs are not supported")
            return int(doc.page_count)
    except ValueError:
        raise
    except Exception:
        raise ValueError("Could not read this PDF (corrupt or unsupported)") from None


class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex
        self.state, self.stage, self.pct = "queued", "reading", 0.0
        self.message, self.error, self.skipped = "Waiting for the reader", "", []
        self.question, self.index, self.total = "", 0, 0
        self.files: list = []
        self.created = time.time()
        self.cancel = threading.Event()

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()

    def set(self, stage=None, message=None, pct=None):
        if stage is not None:
            self.stage = stage
        if message is not None:
            self.message = message
        if pct is not None:
            self.pct = round(max(0.0, min(100.0, float(pct))), 1)

    def public(self) -> dict:
        d = {"id": self.id, "state": self.state, "stage": self.stage, "pct": self.pct,
             "message": self.message, "error": self.error, "skipped": list(self.skipped),
             "question": self.question, "index": self.index, "total": self.total,
             "cancelling": self.cancel.is_set() and self.state in ("queued", "running")}
        return d


class WebState:
    def __init__(self, args, workdir: Path):
        self.args = args
        self.client = ollama.Client(host=args.host)
        self.status_client = ollama.Client(host=args.host, timeout=3)
        self.workdir = workdir
        self.uploads = workdir / "uploads"
        self.reports = workdir / "reports"
        self.max_mb = int(args.max_upload_mb)
        self.max_bytes = self.max_mb * 1024 * 1024
        self.jobs: dict = {}
        self.run_lock = threading.Lock()
        self.hosts: set = set()
        self.origins: set = set()
        self.port = 0

    def bind(self, port: int):
        self.port = port
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.origins = {f"http://{h}" for h in self.hosts}

    def status(self) -> dict:
        a = self.args
        st = ollama_status(self.status_client, a.model, a.embed_model)
        return {"online": st["online"], "model": a.model, "embed_model": a.embed_model,
                "chat_ok": st["chat_ok"], "embed_ok": st["embed_ok"], "fallback": st["fallback"],
                "max_upload_mb": self.max_mb}

    def new_job(self) -> Job:
        job = Job()
        self.jobs[job.id] = job
        for old in sorted(self.jobs.values(), key=lambda j: j.created)[:-KEEP_JOBS]:
            self.jobs.pop(old.id, None)
            shutil.rmtree(self.reports / old.id, ignore_errors=True)
        self.prune_reports()
        return job

    def prune_reports(self):
        """Keep only the newest KEEP_JOBS reports on disk (survives server restarts)."""
        dirs = [d for d in self.reports.iterdir() if d.is_dir() and SAFE_ID.match(d.name)] \
            if self.reports.is_dir() else []
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        for d in dirs[KEEP_JOBS:]:
            shutil.rmtree(d, ignore_errors=True)

    def list_reports(self) -> list:
        out = []
        if not self.reports.is_dir():
            return out
        for d in self.reports.iterdir():
            meta = d / "meta.json"
            if not (d.is_dir() and SAFE_ID.match(d.name) and meta.is_file() and (d / "report.html").is_file()):
                continue
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                out.append({"id": d.name, "created": float(m.get("created", 0)),
                            "files": [str(x) for x in m.get("files", [])][:8],
                            "questions": int(m.get("questions", 0))})
            except (ValueError, OSError, TypeError):
                continue
        out.sort(key=lambda r: r["created"], reverse=True)
        return out[:KEEP_JOBS]

    def upload_path(self, uid: str):
        d = self.uploads / uid
        found = sorted(d.glob("*.pdf")) if d.is_dir() else []
        return found[0] if found else None

    def run_job(self, job: Job, ids, paths, questions, top_k: int, no_llm: bool):
        job.files = [p.name for p in paths]
        with self.run_lock:
            job.state = "running"
            job.set("reading", "Opening the documents", 1)
            try:
                job.check()
                run = self._analyse(job, paths, questions, top_k, no_llm)
                job.check()
                out = self.reports / job.id
                out.mkdir(parents=True, exist_ok=True)
                (out / "RESULTS.md").write_text(render_markdown(run), encoding="utf-8")
                (out / "report.html").write_text(render_html(run), encoding="utf-8")
                (out / "meta.json").write_text(json.dumps(
                    {"created": job.created, "files": [d.name for d in run.docs] or job.files,
                     "questions": len(run.qas)}), encoding="utf-8")
                job.skipped = list(run.skipped)
                job.set("composing", "Report ready", 100)
                job.state = "done"
            except Cancelled:
                job.state, job.message = "cancelled", "Cancelled. Nothing was saved."
            except EtudeError as exc:
                job.error, job.state = str(exc), "error"
            except Exception as exc:  # surface anything unexpected to the page, never to a stack trace
                job.error, job.state = f"{type(exc).__name__}: {exc}", "error"
            finally:
                for uid in ids:
                    shutil.rmtree(self.uploads / uid, ignore_errors=True)

    def _analyse(self, job: Job, paths, questions, top_k: int, no_llm: bool) -> Run:
        a = argparse.Namespace(**vars(self.args))
        a.question, a.interactive, a.top_k, a.no_llm = list(questions), False, top_k, no_llm
        embed_model = resolve_models(self.client, a.model, a.embed_model, no_llm, host=a.host)
        col = open_collection(a.db, embed_model)
        emb = Embedder(self.client, embed_model)
        n = len(paths)
        docs, skipped = [], []
        for i, pdf in enumerate(paths):
            job.check()
            lo, hi = i / n * 55, (i + 1) / n * 55
            job.set("reading", f"Reading {pdf.name}", lo + 1)

            def on_progress(done, total, name=pdf.name, lo=lo, hi=hi):
                job.check()      # raised before anything is written to the index
                job.set("embedding", f"Embedding {name} · {done}/{total} passages",
                        lo + (hi - lo) * done / max(total, 1))

            info = ingest(pdf, col, emb, a.kind, a.keep_references, a.reindex, on_progress)
            if info.chunks:
                docs.append(info)
            else:
                skipped.append(pdf.name)
        if not docs:
            raise EtudeError("No extractable text was found. Scanned PDFs need OCR first (for example with ocrmypdf).")
        ctx = Ctx(col, emb, self.client, a.model, a.top_k, a.min_score, a.no_llm, quiet=True)
        plan = plan_questions(a, docs)
        qas = []
        job.total = len(plan)
        for i, (q, kind, ids) in enumerate(plan):
            job.check()
            job.index, job.question = i + 1, q
            job.set("answering", f"Question {i + 1} of {len(plan)}", 55 + 40 * i / len(plan))
            qas.append(answer_question(q, kind, ids, ctx))
        job.set("composing", "Typesetting the report", 97)
        return Run(datetime.now(), a.model, embed_model, not no_llm, top_k, docs, qas, skipped)


def make_handler(state: WebState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Etude"
        sys_version = ""

        def log_message(self, *args):  # keep the terminal quiet
            pass

        # -- plumbing ---------------------------------------------------------
        def _send(self, status: int, body: bytes, ctype: str, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, status: int = 200):
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def _guard(self, mutating: bool) -> bool:
            """Reject DNS-rebinding (bad Host) and cross-site writes (bad Origin)."""
            if (self.headers.get("Host") or "").lower() not in state.hosts:
                self._json({"error": "Forbidden host"}, 403)
                return False
            if mutating:
                origin = self.headers.get("Origin")
                if (origin and origin.lower() not in state.origins) or \
                        self.headers.get("Sec-Fetch-Site") == "cross-site":
                    self._json({"error": "Forbidden origin"}, 403)
                    return False
            return True

        def _body(self, limit: int):
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._json({"error": "Content-Length required"}, 411)
                return None
            if n < 0 or n > limit:
                self.close_connection = True
                self._json({"error": "Request too large"}, 413)
                return None
            return self.rfile.read(n)

        # -- GET --------------------------------------------------------------
        def do_GET(self):
            if not self._guard(False):
                return
            path = urlparse(self.path)
            route = path.path
            if route == "/":
                nonce = secrets.token_urlsafe(16)
                csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; img-src data:; "
                       "connect-src 'self'; frame-src 'self'; base-uri 'none'; form-action 'none'; "
                       "frame-ancestors 'none'")
                return self._send(200, APP_PAGE.replace("__NONCE__", nonce).encode("utf-8"),
                                  "text/html; charset=utf-8", {"Content-Security-Policy": csp})
            if route == "/api/status":
                return self._json(state.status())
            if route == "/api/reports":
                return self._json({"reports": state.list_reports()})
            m = re.fullmatch(r"/api/job/([0-9a-f]{32})", route)
            if m:
                job = state.jobs.get(m.group(1))
                return self._json(job.public() if job else {"error": "Unknown job"}, 200 if job else 404)
            m = re.fullmatch(r"/report/([0-9a-f]{32})/(report\.html|RESULTS\.md)", route)
            if m:
                f = state.reports / m.group(1) / m.group(2)
                if not f.is_file():
                    return self._json({"error": "Not found"}, 404)
                html_ = m.group(2).endswith(".html")
                extra = {"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
                                                    "frame-ancestors 'self'",
                         "X-Frame-Options": "SAMEORIGIN"}
                if "download=1" in (path.query or ""):
                    extra["Content-Disposition"] = f'attachment; filename="etude-{m.group(2)}"'
                ctype = "text/html; charset=utf-8" if html_ else "text/plain; charset=utf-8"
                return self._send(200, f.read_bytes(), ctype, extra)
            if route == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            return self._json({"error": "Not found"}, 404)

        do_HEAD = do_GET

        # -- POST / DELETE ----------------------------------------------------
        def do_POST(self):
            if not self._guard(True):
                return
            route = urlparse(self.path).path
            if route == "/api/upload":
                return self._upload()
            if route == "/api/analyze":
                return self._analyze()
            m = re.fullmatch(r"/api/job/([0-9a-f]{32})/cancel", route)
            if m:
                job = state.jobs.get(m.group(1))
                if not job:
                    return self._json({"error": "Unknown job"}, 404)
                job.cancel.set()
                return self._json({"ok": True})
            return self._json({"error": "Not found"}, 404)

        def do_DELETE(self):
            if not self._guard(True):
                return
            route = urlparse(self.path).path
            m = re.fullmatch(r"/api/report/([0-9a-f]{32})", route)
            if m:
                shutil.rmtree(state.reports / m.group(1), ignore_errors=True)
                state.jobs.pop(m.group(1), None)
                return self._json({"ok": True})
            m = re.fullmatch(r"/api/upload/([0-9a-f]{32})", route)
            if not m:
                return self._json({"error": "Not found"}, 404)
            shutil.rmtree(state.uploads / m.group(1), ignore_errors=True)
            return self._json({"ok": True})

        def _upload(self):
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._json({"error": "Content-Length required"}, 411)
            if length <= 0:
                return self._json({"error": "The file is empty"}, 400)
            if length > state.max_bytes:
                self.close_connection = True
                return self._json({"error": f"File is larger than {state.max_mb} MB"}, 413)
            name = safe_pdf_name(unquote(self.headers.get("X-Filename", "document.pdf")))
            uid = uuid.uuid4().hex
            d = state.uploads / uid
            d.mkdir(parents=True, exist_ok=True)
            dest = d / name
            try:
                remaining, first = length, True
                with open(dest, "wb") as fh:
                    while remaining:
                        chunk = self.rfile.read(min(1 << 20, remaining))
                        if not chunk:
                            raise ValueError("Upload interrupted")
                        if first:
                            if b"%PDF-" not in chunk[:1024]:
                                raise ValueError("That does not look like a PDF file")
                            first = False
                        fh.write(chunk)
                        remaining -= len(chunk)
                pages = pdf_page_count(dest)
            except Exception as exc:
                shutil.rmtree(d, ignore_errors=True)
                self.close_connection = True
                return self._json({"error": str(exc)}, 400)
            return self._json({"id": uid, "name": name, "size": length, "pages": pages})

        def _analyze(self):
            raw = self._body(64 * 1024)
            if raw is None:
                return
            try:
                data = json.loads(raw.decode("utf-8"))
                assert isinstance(data, dict)
            except Exception:
                return self._json({"error": "Invalid request"}, 400)
            ids = data.get("files")
            if (not isinstance(ids, list) or not 1 <= len(ids) <= MAX_FILES_PER_RUN
                    or not all(isinstance(i, str) and SAFE_ID.match(i) for i in ids) or len(set(ids)) != len(ids)):
                return self._json({"error": "Choose between 1 and 8 uploaded files"}, 400)
            paths = [state.upload_path(i) for i in ids]
            if not all(paths):
                return self._json({"error": "An uploaded file is no longer available. Add it again."}, 400)
            qs = data.get("questions", [])
            if not isinstance(qs, list):
                return self._json({"error": "Invalid questions"}, 400)
            questions = [q.strip()[:500] for q in qs if isinstance(q, str) and q.strip()][:MAX_QUESTIONS]
            try:
                top_k = max(1, min(12, int(data.get("top_k", 5))))
            except (TypeError, ValueError):
                top_k = 5
            no_llm = bool(data.get("no_llm", False))
            job = state.new_job()
            threading.Thread(target=state.run_job, args=(job, ids, paths, questions, top_k, no_llm),
                             daemon=True).start()
            return self._json({"job": job.id})

    return Handler


def start_server(args):
    """Bind the loopback server (does not block). Returns (httpd, state)."""
    args.host = normalise_host(args.host)
    workdir = Path(args.workdir)
    shutil.rmtree(workdir / "uploads", ignore_errors=True)        # stale uploads from a crashed run
    (workdir / "uploads").mkdir(parents=True, exist_ok=True)
    (workdir / "reports").mkdir(parents=True, exist_ok=True)
    state = WebState(args, workdir)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    except OSError as exc:
        raise EtudeError(f"Cannot listen on 127.0.0.1:{args.port} ({exc}). Try --port {args.port + 1}.") from None
    httpd.daemon_threads = True
    state.bind(httpd.server_address[1])
    return httpd, state


def serve(args):
    httpd, state = start_server(args)
    url = f"http://127.0.0.1:{state.port}/"
    console.print(f"\n[bold]ÉTUDE[/bold]  [{SLATE}]reading room[/]\n")
    console.print(f"  [{SLATE}]open[/]    {url}")
    console.print(f"  [{SLATE}]model[/]   {escape(args.model)} · {escape(args.embed_model)}")
    console.print(f"  [{SLATE}]scope[/]   127.0.0.1 only. Uploads are deleted after each run; "
                  f"reports are kept in {escape(str(state.reports))}")
    console.print(f"  [{SLATE}]stop[/]    Ctrl+C\n")
    if args.open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        console.print(f"\n[{SLATE}]Closed.[/]")
    finally:
        httpd.server_close()
        shutil.rmtree(state.uploads, ignore_errors=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="pdf_analyzer", description="Offline RAG over research papers and course syllabi.")
    ap.add_argument("pdfs", nargs="*", type=Path, metavar="PDF",
                    help="PDF files to analyse; omit (or use --serve) to open the web interface")
    ap.add_argument("-q", "--question", action="append", default=[], metavar="TEXT",
                    help="question to ask (repeatable); omit for kind-specific defaults")
    ap.add_argument("-i", "--interactive", action="store_true", help="keep asking questions in a prompt")
    ap.add_argument("-k", "--top-k", type=int, default=5, help="passages retrieved per question")
    ap.add_argument("--min-score", type=float, default=0.0, help="drop passages below this match %%")
    ap.add_argument("--kind", choices=["auto", "paper", "syllabus"], default="auto")
    ap.add_argument("--model", default="llama3.2", help="Ollama chat model")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--db", type=Path, default=Path(".rag_db"), help="ChromaDB directory")
    ap.add_argument("--out", type=Path, default=Path("output"), help="where reports are written")
    ap.add_argument("--no-llm", action="store_true", help="retrieval and citations only")
    ap.add_argument("--reindex", action="store_true", help="re-embed PDFs even if already indexed")
    ap.add_argument("--keep-references", action="store_true", help="index the bibliography too")
    web = ap.add_argument_group("web interface")
    web.add_argument("--serve", action="store_true", help="open the drag-and-drop interface (default when no PDFs are given)")
    web.add_argument("--port", type=int, default=8501, help="port on 127.0.0.1 (0 picks a free one)")
    web.add_argument("--open", action="store_true", help="open the interface in your browser")
    web.add_argument("--workdir", type=Path, default=Path(".etude_web"), help="where uploads and web reports live")
    web.add_argument("--max-upload-mb", type=int, default=100, help="largest accepted PDF")
    args = ap.parse_args(argv)
    args.top_k = max(1, min(args.top_k, 12))
    return args


def main(argv=None):
    try:
        _main(argv)
    except EtudeError as exc:
        fail(str(exc))


def _main(argv=None):
    args = parse_args(argv)
    if args.serve or not args.pdfs:
        if args.pdfs:
            fail("--serve takes no PDF arguments; drop them into the page instead.")
        serve(args)
        return
    for p in args.pdfs:
        if not p.is_file() or p.suffix.lower() != ".pdf":
            fail(f"Not a PDF file: {p}")

    host = normalise_host(args.host)
    client = ollama.Client(host=host)
    embed_model = preflight(client, host, args)
    col = open_collection(args.db, embed_model)
    emb = Embedder(client, embed_model)

    docs, skipped = [], []
    progress = Progress(SpinnerColumn(style=SLATE), TextColumn("{task.description}"),
                        BarColumn(bar_width=28, style=SLATE, complete_style="bold", finished_style="bold"),
                        TextColumn("{task.completed}/{task.total}"), console=console, transient=True)
    with progress:
        for pdf in args.pdfs:
            task = progress.add_task(f"Reading {pdf.name}", total=None)

            def on_progress(done, total, task=task, name=pdf.name):
                progress.update(task, description=f"Embedding {name}", completed=done, total=total)

            try:
                info = ingest(pdf, col, emb, args.kind, args.keep_references, args.reindex, on_progress)
            except Exception as exc:
                fail(f"Could not index {pdf.name}: {exc}")
            progress.remove_task(task)
            if info.chunks == 0:
                skipped.append(pdf.name)
                console.print(f"[{SLATE}]\u2020 {escape(pdf.name)}: no extractable text "
                              "(scanned PDF?). Run OCR first, e.g. ocrmypdf.[/]")
                continue
            docs.append(info)
    if not docs:
        fail("Nothing to analyse.")
    for d in docs:
        state = "cached" if d.cached else "indexed"
        console.print(f"[{SLATE}]{state:<8}[/] {escape(d.title)}  [{SLATE}]{d.pages}p \u00b7 {d.paragraphs}\u00b6 "
                      f"\u00b7 {d.chunks} passages \u00b7 {d.kind}[/]")

    ctx = Ctx(col, emb, client, args.model, args.top_k, args.min_score, args.no_llm)
    qas = []
    for q, kind, ids in plan_questions(args, docs):
        qa = answer_question(q, kind, ids, ctx)
        qas.append(qa)
        show_qa(len(qas), qa)

    if args.interactive:
        kinds = {d.kind for d in docs}
        kind = next(iter(kinds)) if len(kinds) == 1 else "document"
        ids = [d.doc_id for d in docs]
        console.print(f"\n[{SLATE}]Ask anything. Empty line to finish and write the reports.[/]")
        while True:
            try:
                q = console.input("[bold]\u203a[/bold] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            qa = answer_question(q, kind, ids, ctx)
            qas.append(qa)
            show_qa(len(qas), qa)

    if not qas:
        console.print("No questions were asked; no report written.")
        return

    run = Run(datetime.now(), args.model, embed_model, not args.no_llm, args.top_k, docs, qas, skipped)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "RESULTS.md").write_text(render_markdown(run), encoding="utf-8")
    (args.out / "report.html").write_text(render_html(run), encoding="utf-8")
    console.print(f"\n[{SLATE}]written[/]  {escape(str(args.out / 'RESULTS.md'))}\n"
                  f"[{SLATE}]written[/]  {escape(str(args.out / 'report.html'))}")


if __name__ == "__main__":
    main()
