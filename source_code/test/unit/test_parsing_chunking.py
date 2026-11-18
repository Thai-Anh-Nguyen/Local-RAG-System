from pathlib import Path

import pytest

from rag.chunking import chunk_id, chunk_pages, clean_heading
from rag.embeddings import DOC_PREFIX, QUERY_PREFIX, PrefixedEmbeddings
from rag.parsing import PageText, normalize_text, parse_file

HANDBOOK = Path(__file__).resolve().parents[2] / "data/handbooks/CS_MajorHandbook.pdf"


# ---- normalization -------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Graduate A0ributes", "Graduate Attributes"),
        ("Graduate A2ributes", "Graduate Attributes"),
        ("Program Descrip8on", "Program Description"),
        ("Founda'onal Courses", "Foundational Courses"),
        ("Program Educa;onal Objec;ves", "Program Educational Objectives"),
        ("ramiﬁcations", "ramifications"),
        ("Mission ........ 2", "Mission  …  2"),
    ],
)
def test_normalize_repairs_broken_ligatures(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize("text", ["CS2xx and CS3xx", "MATH103", "CS 401/402", "the student's plan", "GA10"])
def test_normalize_leaves_course_codes_and_normal_text_alone(text):
    assert normalize_text(text) == text


def test_clean_heading_strips_markdown():
    assert clean_heading("**CS311: Advanced Deep Learning** ") == "CS311: Advanced Deep Learning"
    assert clean_heading("<u>Sample Minor Plan</u>") == "Sample Minor Plan"


# ---- chunking ------------------------------------------------------------------------
PAGES = [
    PageText(1, "# 1. Intro\nWelcome text.\n## Mission\nOur mission."),
    PageText(2, "Mission continues on the next page.\n# 2. Courses\n### **CS101: Basics**\n"
                "### _Prerequisites: None_\nLearn things."),
]


def _chunks(pages=PAGES, size=1000, overlap=200):
    return chunk_pages(pages, source="hb.pdf", file_hash="abc", chunk_size=size, chunk_overlap=overlap)


def test_heading_trail_carries_across_pages():
    docs = _chunks()
    carried = next(d for d in docs if "continues" in d.page_content)
    assert carried.metadata["page"] == 2
    assert carried.metadata["section"] == "1. Intro"
    assert carried.metadata["heading"] == "Mission"
    assert carried.page_content.startswith("[1. Intro > Mission]\n")


def test_prerequisite_lines_are_body_not_headings():
    docs = _chunks()
    course = next(d for d in docs if "Learn things" in d.page_content)
    assert course.metadata["heading"] == "CS101: Basics"
    assert "Prerequisites: None" in course.page_content


def test_bold_course_titles_start_a_new_section():
    pages = [PageText(1, "### **CS306: Computer Security**\nSecurity text.\n\n"
                         "**CS307: Machine Learning** _Prerequisites: CS201_\nML text.")]
    docs = _chunks(pages)
    ml = next(d for d in docs if "ML text" in d.page_content)
    sec = next(d for d in docs if "Security text" in d.page_content)
    assert ml.metadata["heading"] == "CS307: Machine Learning"
    assert "Prerequisites: CS201" in ml.page_content
    assert sec.metadata["heading"] == "CS306: Computer Security"
    assert "ML text" not in sec.page_content


def test_inline_prerequisite_is_moved_out_of_heading():
    pages = [PageText(1, "### **CS212: Programming Languages** _Prerequisites: CS201_\nPL text.")]
    doc = _chunks(pages)[0]
    assert doc.metadata["heading"] == "CS212: Programming Languages"
    assert "Prerequisites: CS201" in doc.page_content


def test_chunk_metadata_and_ids_are_stable():
    a, b = _chunks(), _chunks()
    assert [d.metadata["id"] for d in a] == [d.metadata["id"] for d in b]
    assert len({d.metadata["id"] for d in a}) == len(a)
    first = a[0].metadata
    assert first["id"] == chunk_id("abc", first["page"], first["chunk_index"])
    for d in a:
        assert {"source", "file_hash", "page", "chunk_index", "section", "heading", "char_len"} <= d.metadata.keys()
        assert d.metadata["char_len"] == len(d.page_content)


def test_long_sections_are_split_with_overlap():
    long = PageText(1, "# Big\n" + " ".join(f"word{i}" for i in range(600)))
    docs = _chunks([long], size=300, overlap=50)
    assert len(docs) > 5
    assert all(len(d.page_content) <= 300 + len("[Big]\n") for d in docs)


# ---- real handbook -------------------------------------------------------------------
@pytest.mark.parametrize("parser", ["pymupdf", "pypdf"])
def test_handbook_parses_with_both_parsers(parser):
    pages = parse_file(HANDBOOK, parser)
    assert len(pages) >= 15
    assert all(p.text.strip() for p in pages)
    full = " ".join(p.text for p in pages)
    assert "Computer Science" in full


def test_handbook_chunks_have_course_headings():
    pages = parse_file(HANDBOOK, "pymupdf")
    docs = chunk_pages(pages, source=HANDBOOK.name, file_hash="h", chunk_size=1000, chunk_overlap=200)
    headings = {d.metadata["heading"] for d in docs}
    assert "CS311: Advanced Deep Learning" in headings
    assert not any("8on" in h or "0ributes" in h for h in headings)


def test_unsupported_file_type_is_rejected(tmp_path):
    f = tmp_path / "x.docx"
    f.write_text("hi")
    with pytest.raises(ValueError):
        parse_file(f)


# ---- embedding prefixes --------------------------------------------------------------
class _Recorder:
    def __init__(self):
        self.seen = []

    def embed_documents(self, texts):
        self.seen += texts
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        self.seen.append(text)
        return [0.0]


def test_nomic_prefixes_applied():
    rec = _Recorder()
    emb = PrefixedEmbeddings(rec, use_prefixes=True)
    emb.embed_documents(["chunk"])
    emb.embed_query("question")
    assert rec.seen == [DOC_PREFIX + "chunk", QUERY_PREFIX + "question"]


def test_prefixes_can_be_disabled_for_baseline():
    rec = _Recorder()
    PrefixedEmbeddings(rec, use_prefixes=False).embed_query("q")
    assert rec.seen == ["q"]
