"""Turn source files (.pdf, .md, .txt) into per-page text.

PDFs are parsed with PyMuPDF (pymupdf4llm -> markdown with headings and tables) by default,
or with pypdf (plain text, like the baseline) when RAG_PDF_PARSER=pypdf.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}

# Some handbooks were exported with broken ligature glyphs: "tt"/"ti" come out as digits
# ("A0ributes", "A2ributes", "Descrip8on"). Repair only word-internal cases, and skip course
# codes such as "CS2xx" (two or more capitals followed by a digit).
_LIGATURE_WORD = re.compile(r"\b(?![A-Z]{2,}\d)([A-Za-z]+)([028])([a-z]{2,})\b")
_LIGATURE_MAP = {"0": "tt", "2": "tt", "8": "ti"}
# Same problem, rendered as punctuation inside a word: "Founda'onal", "Descrip;on", "Objec;ves"
_LIGATURE_APOS = re.compile(r"(?<=[a-z])(?:['’](?=on)|;(?=[a-z]))")
# Table-of-contents leader dots ("Mission ........ 2")
_LEADER_DOTS = re.compile(r"\.{4,}")


@dataclass
class PageText:
    page: int  # 1-based; 1 for single-page sources (.md/.txt)
    text: str


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # "ﬁ" -> "fi", etc.
    text = _LIGATURE_WORD.sub(lambda m: m.group(1) + _LIGATURE_MAP[m.group(2)] + m.group(3), text)
    text = _LIGATURE_APOS.sub("ti", text)
    return _LEADER_DOTS.sub(" … ", text)


def _parse_pdf_pymupdf(path: Path) -> list[PageText]:
    import contextlib
    import logging
    import io

    import pymupdf
    import pymupdf4llm

    # pymupdf4llm prints "=== Document parser messages ===" / "RapidOCR is missing" to stdout for
    # every PDF (OCR is irrelevant here: the handbook is text-based). Route its messages to the
    # `pymupdf` logger (silent unless logging is configured) and swallow the stray print().
    pymupdf.set_messages(pylogging=True, pylogging_level=logging.INFO)
    with contextlib.redirect_stdout(io.StringIO()):
        pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    return [PageText(page=p["metadata"]["page_number"], text=p["text"]) for p in pages]


def _parse_pdf_pypdf(path: Path) -> list[PageText]:
    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed-xref warnings are noise
    reader = PdfReader(str(path))
    return [PageText(page=i + 1, text=page.extract_text() or "") for i, page in enumerate(reader.pages)]


def parse_file(path: Path, pdf_parser: str = "pymupdf") -> list[PageText]:
    """Return non-empty, normalized pages of a supported file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = _parse_pdf_pymupdf(path) if pdf_parser == "pymupdf" else _parse_pdf_pypdf(path)
    elif suffix in {".md", ".txt"}:
        pages = [PageText(page=1, text=path.read_text(encoding="utf-8"))]
    else:
        raise ValueError(f"Unsupported file type: {path.name} (supported: {sorted(SUPPORTED_SUFFIXES)})")

    out = []
    for p in pages:
        text = normalize_text(p.text).strip()
        if text:  # image-only pages yield no text (no OCR) and are skipped
            out.append(PageText(page=p.page, text=text))
    return out
