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

# Bump when chunking logic changes so existing indexes are rebuilt on the next ingest.
CHUNKER_VERSION = "2"

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Lines rendered as headings that are really body text (e.g. "### _Prerequisites: CS201_")
_FAKE_HEADING = re.compile(r"^prerequisite", re.IGNORECASE)
# Course codes: CS311, MATH 202, CS3xx
_COURSE_CODE = r"[A-Z]{2,5}\s?(?:\d{3}[A-Za-z]?|\d[xX]{2})"
# Course titles the PDF renders as bold text rather than a heading:
#   "**CS307: Machine Learning** _Prerequisites: CS201_"
_BOLD_COURSE = re.compile(rf"^\*\*\s*({_COURSE_CODE}\s*:[^*]+?)\s*\*\*(.*)$")
_IS_COURSE = re.compile(rf"^{_COURSE_CODE}\s*:")
# "CS212: Programming Languages Prerequisites: CS201" -> title + body line
_INLINE_PREREQ = re.compile(r"\s+(Prerequisites?\s*:.*)$", re.IGNORECASE)


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

    def course_level() -> int:
        """Bold course titles sit at the same level as the previous course heading,
        or one below the deepest current heading."""
        deepest = max(trail) if trail else 0
        if deepest and _IS_COURSE.match(trail[deepest]):
            return deepest
        return min(deepest + 1, 6)

    for page in pages:
        sec = _Section(page=page.page, trail=current_trail())
        sections.append(sec)
        for line in page.text.splitlines():
            m = _HEADING.match(line)
            bold = None if m else _BOLD_COURSE.match(line.strip())
            if m:
                heading, level, rest = clean_heading(m.group(2)), len(m.group(1)), ""
            elif bold:
                heading, level, rest = clean_heading(bold.group(1)), course_level(), bold.group(2)
            else:
                sec.lines.append(line)
                continue

            if not heading or _FAKE_HEADING.match(heading):
                sec.lines.append(heading)
                continue
            prereq = _INLINE_PREREQ.search(heading)
            if prereq:
                heading, rest = heading[: prereq.start()].strip(), f"{prereq.group(1)} {rest}"
            for deeper in [k for k in trail if k >= level]:
                del trail[deeper]
            trail[level] = heading
            sec = _Section(page=page.page, trail=current_trail())
            sections.append(sec)
            if clean_heading(rest):
                sec.lines.append(clean_heading(rest))
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
