"""Split parsed pages into chunks that carry source, page and heading metadata.

Pages are first cut at markdown headings so every chunk knows the section it belongs to
(the heading trail carries over across page breaks). Each section is then split with
RecursiveCharacterTextSplitter at the configured size/overlap, as in the baseline.
Each chunk's text starts with a breadcrumb line ("[Section > Heading]") so that the
embedding sees which section a fragment comes from, e.g. which course a description is about.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parsing import PageText

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Lines rendered as headings that are really body text (e.g. "### _Prerequisites: CS201_")
_FAKE_HEADING = re.compile(r"^prerequisite", re.IGNORECASE)


def clean_heading(raw: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw)  # <u>...</u>
    text = re.sub(r"[*_`]+", "", text)  # markdown emphasis
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class _Section:
    page: int
    trail: list[str]  # heading trail, outermost first
    lines: list[str] = field(default_factory=list)


def _sections(pages: list[PageText]) -> list[_Section]:
    trail: dict[int, str] = {}  # heading level -> text
    sections: list[_Section] = []

    def current_trail() -> list[str]:
        return [trail[k] for k in sorted(trail)]

    for page in pages:
        sec = _Section(page=page.page, trail=current_trail())
        sections.append(sec)
        for line in page.text.splitlines():
            m = _HEADING.match(line)
            heading = clean_heading(m.group(2)) if m else ""
            if m and heading and not _FAKE_HEADING.match(heading):
                level = len(m.group(1))
                for deeper in [k for k in trail if k >= level]:
                    del trail[deeper]
                trail[level] = heading
                sec = _Section(page=page.page, trail=current_trail())
                sections.append(sec)
            else:
                sec.lines.append(clean_heading(m.group(2)) if m else line)
    return [s for s in sections if "\n".join(s.lines).strip()]


def chunk_id(file_hash: str, page: int, index: int) -> str:
    return hashlib.sha1(f"{file_hash}:{page}:{index}".encode()).hexdigest()


def chunk_pages(
    pages: list[PageText],
    *,
    source: str,
    file_hash: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: list[Document] = []
    per_page_index: dict[int, int] = {}
    for sec in _sections(pages):
        breadcrumb = " > ".join(sec.trail)
        body = "\n".join(sec.lines).strip()
        for piece in splitter.split_text(body):
            idx = per_page_index.get(sec.page, 0)
            per_page_index[sec.page] = idx + 1
            text = f"[{breadcrumb}]\n{piece}" if breadcrumb else piece
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "id": chunk_id(file_hash, sec.page, idx),
                        "source": source,
                        "file_hash": file_hash,
                        "page": sec.page,
                        "chunk_index": idx,
                        "section": sec.trail[0] if sec.trail else "",
                        "heading": sec.trail[-1] if sec.trail else "",
                        "char_len": len(text),
                        "edited": False,
                    },
                )
            )
    return docs
