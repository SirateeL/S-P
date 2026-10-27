#!/usr/bin/env python3
"""pdf_analyzer.py -- offline RAG over research papers and course syllabi.

    PyMuPDF  ->  structure-aware chunks  ->  Ollama embeddings  ->  ChromaDB
             ->  top-k retrieval  ->  llama3.2 answer  ->  RESULTS.md + report.html

Every citation carries: page, paragraph, section heading, an exact quoted
snippet, and a similarity percentage. Nothing leaves localhost.

    python pdf_analyzer.py paper.pdf syllabus.pdf -q "How is grading weighted?"
"""
from __future__ import annotations

import argparse
import hashlib
import html
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

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


def ingest(pdf: Path, col, emb: Embedder, args, progress) -> DocInfo:
    doc_id = doc_fingerprint(pdf, args.keep_references)
    blocks, meta = extract_blocks(pdf)
    paras, detected = build_paragraphs(blocks, body_font_size(blocks), args.keep_references)
    chunks = [c for p in paras for c in chunk_paragraph(p, doc_id, pdf.name)]
    kind = args.kind if args.kind != "auto" else detect_kind(paras)
    info = DocInfo(doc_id, pdf.name, pick_title(meta["title"], detected, pdf.stem), kind,
                   meta["pages"], len(paras), len(chunks), meta["empty_pages"])
    if not chunks:
        return info
    existing = col.get(where={"doc_id": doc_id}, limit=1)["ids"]
    if len(existing) and not args.reindex:
        info.cached = True
        return info
    if len(existing):
        col.delete(where={"doc_id": doc_id})
    task = progress.add_task(f"Embedding {pdf.name}", total=len(chunks))
    vecs = emb.embed([f"{c.section}\n{c.text}" for c in chunks], "doc",
                     on_batch=lambda done, total: progress.update(task, completed=done))
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


CSS = r"""
:root{
--linen:#F5F3EE;--parchment:#EFECE6;--ink:#191817;--taupe:#8A857B;--rule:#D3CEC6;
--serif:"Didot","Bodoni MT","Cinzel","Georgia",serif;
--sans:-apple-system,BlinkMacSystemFont,"Helvetica Neue","Arial",sans-serif;
--gutter:24px}
*{box-sizing:border-box;margin:0;padding:0}
html{background:var(--linen);color:var(--ink);scroll-behavior:smooth;
-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{font-family:var(--serif);font-size:17px;line-height:1.7;-webkit-font-smoothing:antialiased;
text-rendering:optimizeLegibility;background:var(--linen)}
a{color:inherit}
.label,dt,th,summary{font-family:var(--sans);font-size:10.5px;line-height:1.4;letter-spacing:.12em;
text-transform:uppercase;font-weight:500;color:var(--taupe)}
.page{max-width:1240px;margin:0 auto;padding:0 40px 96px}
.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));column-gap:var(--gutter)}

/* ---- masthead ---- */
.topbar{padding:22px 0 18px;border-bottom:1px solid var(--rule);align-items:baseline}
.topbar .wm{grid-column:1/4;color:var(--ink);font-weight:700;letter-spacing:.32em}
.topbar .mid{grid-column:4/10}
.topbar .end{grid-column:10/13;text-align:right}
.hero{padding:88px 0 0;align-items:end}
h1{grid-column:1/10;font-weight:400;font-size:clamp(3.6rem,11.5vw,9.6rem);line-height:.86;
letter-spacing:-.025em;margin-left:-.04em}
.hero .stamp{grid-column:10/13;text-align:right;padding-bottom:.9rem}
.hero .stamp b{display:block;color:var(--ink);font-weight:600}
.sub{grid-column:4/12;margin-top:30px;font-size:1.2rem;line-height:1.5;color:var(--taupe);font-style:italic}
.meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin-top:64px;
border-top:1px solid var(--ink);border-bottom:1px solid var(--rule)}
.meta>div{padding:16px 20px 18px 0}
.meta>div+div{padding-left:20px;border-left:1px solid var(--rule)}
.meta dd{margin-top:6px;font-size:1.18rem;line-height:1.35;overflow-wrap:anywhere}

/* ---- register ---- */
.regwrap{overflow-x:auto;margin-top:8px}
table.reg{width:100%;border-collapse:collapse}
table.reg th{text-align:left;padding:22px 16px 12px 0;border-bottom:1px solid var(--rule);white-space:nowrap}
table.reg td{padding:13px 16px 13px 0;border-bottom:1px solid var(--rule);vertical-align:baseline;font-size:1.02rem;line-height:1.4}
table.reg th.n{text-align:right}
table.reg td.n{text-align:right;font-family:var(--sans);font-size:12.5px;font-variant-numeric:tabular-nums}
table.reg td.f,table.reg td.k{font-family:var(--sans);font-size:12.5px;color:var(--taupe)}
table.reg tr:last-child td{border-bottom:1px solid var(--ink)}
.warn{margin-top:14px;font-family:var(--sans);font-size:12px;color:var(--taupe)}

/* ---- body frame ---- */
.frame{margin-top:88px}
.rail{grid-column:1/4;position:sticky;top:28px;align-self:start}
.rail ol{list-style:none;margin-top:14px;border-top:1px solid var(--ink)}
.rail li a{display:grid;grid-template-columns:2.4rem 1fr;padding:11px 0;border-bottom:1px solid var(--rule);
text-decoration:none;font-size:.9rem;line-height:1.4;transition:color .15s}
.rail li a:hover{color:var(--taupe)}
.rail .rt{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
main{grid-column:4/13}

/* ---- question & synthesis ---- */
.plate{padding-bottom:88px;scroll-margin-top:24px}
.plate+.plate{border-top:1px solid var(--ink);padding-top:30px}
.plate:first-child{border-top:1px solid var(--ink);padding-top:30px}
.plate-head{display:grid;grid-template-columns:5.2rem minmax(0,1fr);column-gap:var(--gutter);align-items:baseline}
.qn{font-family:var(--sans);font-size:13px;font-weight:700;letter-spacing:.1em;color:var(--ink);white-space:nowrap}
.qn i{font-style:normal;color:var(--taupe);font-weight:400;margin-left:.35em}
h2{font-weight:400;font-size:clamp(1.55rem,3vw,2.15rem);line-height:1.22;letter-spacing:-.01em}
.synth{margin:40px 0 0 calc(5.2rem + var(--gutter));padding:2px 0 2px 28px;border-left:1px solid var(--ink);max-width:68ch}
.synth .label{margin-bottom:16px}
.answer{font-size:1.38rem;line-height:1.7;letter-spacing:-.003em}
.answer p+p,.answer ul{margin-top:.85em}
.answer ul{padding-left:1.1em}
.answer li::marker{color:var(--taupe)}
a.ref,span.ref{font-family:var(--sans);font-size:10px;font-weight:700;letter-spacing:.02em;line-height:1.35;
display:inline-block;position:relative;top:-.5em;min-width:1.9em;padding:1px 6px;margin:0 2px;
text-align:center;border:1px solid var(--ink);border-radius:999px;text-decoration:none}
a.ref:hover{background:var(--ink);color:var(--linen)}
span.ref.bad{border:1px dashed var(--taupe);color:var(--taupe)}
.note{margin:18px 0 0 calc(5.2rem + var(--gutter));padding-left:28px;font-family:var(--sans);font-size:12px;
line-height:1.55;color:var(--taupe);max-width:72ch}

/* ---- evidence cards ---- */
.evidence{margin-top:56px}
.evidence>.label{display:flex;align-items:center;gap:16px;margin-bottom:20px}
.evidence>.label::after{content:"";flex:1;height:1px;background:var(--rule)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.cite{display:flex;flex-direction:column;background:var(--parchment);border:1px solid var(--rule);
border-top:1px solid var(--ink);padding:20px 22px 18px;scroll-margin-top:24px;transition:background .15s,border-color .15s}
.cite:hover{border-color:var(--taupe);border-top-color:var(--ink)}
.cite:target{background:var(--linen);outline:1px solid var(--ink);outline-offset:3px}
.cite-top{display:flex;align-items:flex-start;gap:12px}
.idx{font-family:var(--sans);font-size:12px;font-weight:700;letter-spacing:.06em;padding-top:4px}
.pill{display:inline-block;max-width:100%;font-family:var(--sans);font-size:10.5px;line-height:1.35;
letter-spacing:.1em;text-transform:uppercase;font-weight:500;padding:4px 11px;border:1px solid var(--ink);
border-radius:14px;overflow-wrap:anywhere}
.src{margin-top:10px;font-family:var(--sans);font-size:11.5px;color:var(--taupe);overflow-wrap:anywhere}
.anchors{display:flex;gap:28px;margin:18px 0 0;padding:14px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.anchors dd{margin-top:2px;font-family:var(--sans);font-size:1.45rem;line-height:1.1;font-weight:700;
letter-spacing:-.02em;font-variant-numeric:tabular-nums}
blockquote{position:relative;margin-top:22px;padding-top:30px;font-size:1.13rem;line-height:1.65;font-style:italic;
overflow-wrap:anywhere}
blockquote::before{content:"\201C";position:absolute;top:-8px;left:-2px;font-family:var(--serif);font-style:normal;
font-size:3.4rem;line-height:1;color:var(--taupe)}
.foot{margin-top:auto;padding-top:20px}
.match{display:grid;grid-template-columns:auto 1fr auto;align-items:center;column-gap:12px}
.track{position:relative;height:1px;background:var(--rule)}
.track i{position:absolute;left:0;top:-1px;height:3px;background:var(--ink)}
.pct{font-family:var(--sans);font-size:12.5px;font-weight:700;font-variant-numeric:tabular-nums}
details{margin-top:14px}
summary{cursor:pointer;list-style:none;display:inline-block}
summary::-webkit-details-marker{display:none}
summary::before{content:"+ "}details[open] summary::before{content:"\2013 "}
summary:hover{color:var(--ink)}
details p{margin-top:10px;padding-top:12px;border-top:1px solid var(--rule);font-size:.96rem;line-height:1.6;
color:#4a4743;overflow-wrap:anywhere}

/* ---- colophon ---- */
footer{margin-top:16px;padding-top:28px;border-top:1px solid var(--ink)}
footer p{max-width:72ch;font-family:var(--sans);font-size:12px;line-height:1.7;color:var(--taupe)}
footer strong{color:var(--ink);font-weight:600}
.sw{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:22px}
.sw span{display:flex;align-items:center;gap:8px}
.sw i{width:12px;height:12px;border:1px solid var(--rule);display:inline-block}

@media(max-width:960px){
.page{padding:0 24px 72px}
.topbar .wm{grid-column:1/7}.topbar .mid{display:none}.topbar .end{grid-column:7/13}
.hero{padding-top:56px}h1{grid-column:1/13}.hero .stamp{grid-column:1/13;text-align:left;padding:18px 0 0}
.sub{grid-column:1/13;margin-top:20px}
.meta{grid-template-columns:repeat(2,minmax(0,1fr));margin-top:44px}
.meta>div:nth-child(3){padding-left:0;border-left:0}
.meta>div:nth-child(n+3){border-top:1px solid var(--rule)}
.frame{margin-top:56px}.rail{display:none}main{grid-column:1/13}
.plate-head{grid-template-columns:1fr;row-gap:12px}
.synth,.note{margin-left:0}.synth{padding-left:20px}.note{padding-left:20px}}
@media(max-width:640px){
body{font-size:16px}.answer{font-size:1.22rem}
table.reg th:nth-child(3),table.reg td:nth-child(3),table.reg th:nth-child(4),table.reg td:nth-child(4){display:none}
.cite{padding:18px 18px 16px}}
@media print{
html,body{background:#fff}.rail{display:none}main{grid-column:1/13}.page{padding:0}
.cite,.synth{break-inside:avoid}.plate{padding-bottom:40px}}
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
        f'<article class="cite" id="{qid}c{c.n}">'
        f'<header><div class="cite-top"><span class="idx">[{c.n}]</span>'
        f'<span class="pill" title="Section heading">{e(c.section)}</span></div>{src}</header>'
        f'<dl class="anchors"><div><dt>Page</dt><dd>{c.page}</dd></div>'
        f"<div><dt>Paragraph</dt><dd>¶ {c.paragraph}</dd></div></dl>"
        f"<blockquote>{e(c.quote)}{tail}</blockquote>"
        f'<div class="foot"><div class="match"><span class="label">Match</span>'
        f'<span class="track"><i style="width:{c.score:.1f}%"></i></span>'
        f'<span class="pct">{c.score:.1f}%</span></div>'
        f'<details><summary>Full passage</summary><p>{e(c.text)}</p></details></div></article>')


def render_html(run: Run) -> str:
    e = html.escape
    multi = len(run.docs) > 1
    titles = " · ".join(d.title for d in run.docs)
    engine = f"{run.llm} · {run.embed_model}" if run.generated_text else f"Retrieval only · {run.embed_model}"
    passages = sum(d.chunks for d in run.docs)
    paragraphs = sum(d.paragraphs for d in run.docs)
    synth_label = f"Synthesis · {run.llm}" if run.generated_text else "Retrieval only"

    rail = "".join(
        f'<li><a href="#q{i}"><span class="label">{i:02d}</span><span class="rt">{e(qa.question)}</span></a></li>'
        for i, qa in enumerate(run.qas, 1))
    rows = "".join(
        f'<tr><td class="n">{i:02d}</td><td>{e(d.title)}</td><td class="f">{e(d.name)}</td>'
        f'<td class="k">{e(d.kind)}</td><td class="n">{d.pages}</td><td class="n">{d.paragraphs}</td>'
        f'<td class="n">{d.chunks}</td></tr>'
        for i, d in enumerate(run.docs, 1))
    warn = "".join(
        f'<p class="warn">† {e(d.name)}: no extractable text on page(s) '
        f'{e(", ".join(map(str, d.empty_pages[:12])))}. Scanned pages need OCR first.</p>'
        for d in run.docs if d.empty_pages)

    plates = []
    for i, qa in enumerate(run.qas, 1):
        qid = f"q{i}"
        notes = "".join(f'<p class="note">† {e(w)}</p>' for w in qa.warnings)
        n = len(qa.citations)
        plural = "" if n == 1 else "s"
        cards = "".join(cite_html(qid, c, multi) for c in qa.citations)
        evidence = (f'<div class="evidence"><p class="label">Evidence · {n} passage{plural}</p>'
                    f'<div class="cards">{cards}</div></div>' if n else "")
        plates.append(
            f'<section class="plate" id="{qid}"><header class="plate-head">'
            f'<span class="qn">{i:02d}<i>//</i></span><h2>{e(qa.question)}</h2></header>'
            f'<div class="synth"><p class="label">{e(synth_label)}</p>'
            f'<div class="answer">{answer_html(qa.answer, qid, n)}</div></div>'
            f"{notes}{evidence}</section>")

    when = run.generated.strftime("%d %b %Y") + " · " + run.generated.strftime("%H:%M")
    meta = [("Engine", engine), ("Generated", when),
            ("Documents", f"{len(run.docs):02d}"), ("Index", f"{paragraphs} ¶ · {passages} passages")]
    meta_html = "".join(f"<div><dt>{k}</dt><dd>{e(v)}</dd></div>" for k, v in meta)
    stamp = f"<b>Nº {run.generated:%y%m%d-%H%M}</b>{run.generated:%d %B %Y}"
    swatches = [("#F5F3EE", "Linen"), ("#EFECE6", "Parchment"), ("#191817", "Charcoal"),
                ("#8A857B", "Taupe"), ("#D3CEC6", "Hairline")]
    sw = "".join(f'<span><i style="background:{h}"></i>{n_} {h[1:]}</span>' for h, n_ in swatches)

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>Étude — Analysis</title><style>" + CSS + "</style></head><body><div class=\"page\">"
        '<div class="topbar grid"><span class="label wm">Étude</span>'
        '<span class="label mid">Local · Offline · Reading report</span>'
        f'<span class="label end">{run.generated:%Y}</span></div>'
        '<header class="hero grid"><h1>Analysis</h1>'
        f'<p class="label stamp">{stamp}</p>'
        f'<p class="sub">{e(titles)}</p></header>'
        f'<dl class="meta">{meta_html}</dl>'
        '<div class="regwrap"><table class="reg"><thead><tr><th class="n">Nº</th><th>Title</th><th>File</th>'
        '<th>Kind</th><th class="n">Pages</th><th class="n">¶</th><th class="n">Passages</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>{warn}"
        f'<div class="frame grid"><nav class="rail"><p class="label">Questions</p><ol>{rail}</ol></nav>'
        f"<main>{''.join(plates)}</main></div>"
        "<footer><p>Produced offline by pdf_analyzer.py. <strong>Match</strong> is the cosine similarity "
        "between the question and the passage embedding, times 100: a ranking signal, not a probability "
        "of correctness. Quotes are literal passages of the extracted text, with line-wrap hyphens and "
        "ligatures normalised. A small local model can still misread its sources; the quotes and page "
        "references are retrieved, not generated, and are the ground truth.</p>"
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


def answer_question(question: str, kind: str, doc_ids, ctx: Ctx) -> QA:
    cites = retrieve(ctx.col, ctx.emb, question, ctx.k, doc_ids, ctx.min_score)
    if not cites:
        return QA(question, kind, "No passage cleared the similarity threshold.")
    if ctx.no_llm:
        return QA(question, kind, "Retrieval only: generation was skipped (--no-llm).", cites)
    try:
        with console.status(f"[{SLATE}]Reading locally with {ctx.model}..."):
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


def preflight(client, host: str, args) -> str:
    try:
        names = model_names(client.list())
    except Exception as exc:
        fail(f"Cannot reach Ollama at {host}. Start it with `ollama serve`. ({exc})")
    if not args.no_llm and not has_model(names, args.model):
        fail(f"Model '{args.model}' is not installed. Run: ollama pull {args.model}")
    embed_model = args.embed_model
    if not has_model(names, embed_model):
        if has_model(names, args.model):
            console.print(f"[{SLATE}]'{escape(embed_model)}' is not installed; falling back to "
                          f"'{escape(args.model)}' for embeddings (weaker retrieval). "
                          f"For better results: ollama pull {escape(embed_model)}[/]")
            embed_model = args.model
        else:
            fail(f"Embedding model '{embed_model}' is not installed. Run: ollama pull {embed_model}")
    return embed_model


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="pdf_analyzer", description="Offline RAG over research papers and course syllabi.")
    ap.add_argument("pdfs", nargs="+", type=Path, metavar="PDF")
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
    args = ap.parse_args(argv)
    args.top_k = max(1, min(args.top_k, 12))
    return args


def main(argv=None):
    args = parse_args(argv)
    for p in args.pdfs:
        if not p.is_file() or p.suffix.lower() != ".pdf":
            fail(f"Not a PDF file: {p}")

    host = normalise_host(args.host)
    client = ollama.Client(host=host)
    embed_model = preflight(client, host, args)
    col = open_collection(args.db, embed_model)
    emb = Embedder(client, embed_model)

    docs = []
    progress = Progress(SpinnerColumn(style=SLATE), TextColumn("{task.description}"),
                        BarColumn(bar_width=28, style=SLATE, complete_style="bold", finished_style="bold"),
                        TextColumn("{task.completed}/{task.total}"), console=console, transient=True)
    with progress:
        for pdf in args.pdfs:
            try:
                info = ingest(pdf, col, emb, args, progress)
            except Exception as exc:
                fail(f"Could not index {pdf.name}: {exc}")
            if info.chunks == 0:
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

    run = Run(datetime.now(), args.model, embed_model, not args.no_llm, args.top_k, docs, qas)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "RESULTS.md").write_text(render_markdown(run), encoding="utf-8")
    (args.out / "report.html").write_text(render_html(run), encoding="utf-8")
    console.print(f"\n[{SLATE}]written[/]  {escape(str(args.out / 'RESULTS.md'))}\n"
                  f"[{SLATE}]written[/]  {escape(str(args.out / 'report.html'))}")


if __name__ == "__main__":
    main()
