from conftest import HashEmbeddings
from langchain_core.documents import Document

from rag.retrieval import retrieve
from rag.store import KnowledgeStore


def test_store_survives_collection_reset_by_another_process(store):
    """`rag.ingest --reset` (another KnowledgeStore on the same server) deletes and re-creates the
    collection; the API's store must reconnect instead of failing with 'collection does not exist'."""
    other = KnowledgeStore(store.client, HashEmbeddings(), store.collection_name)
    other.reset()
    other.add_chunks([Document(
        page_content="[Parking]\nStudents may park in lot B.",
        metadata={"id": "p1", "source": "notes.md", "file_hash": "h", "page": 1, "chunk_index": 0,
                  "section": "", "heading": "Parking", "char_len": 28, "edited": False},
    )])

    version_before = store.version
    assert store.count() == 1  # stale handle -> NotFoundError -> reconnect -> retry
    assert store.version > version_before  # caches keyed on version (BM25) are invalidated
    assert [d["source"] for d in store.list_documents()] == ["notes.md"]
    assert retrieve(store, "where can students park", 1, "hybrid")[0].chunk.metadata["source"] == "notes.md"
