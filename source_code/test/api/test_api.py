from conftest import FakeLLM, read_sse

from rag.generation import NOT_FOUND, build_messages
from rag.retrieval import Turn, retrieval_query, retrieve


# ---- chat --------------------------------------------------------------------------------
def _chat(client, **body):
    with client.stream("POST", "/api/chat", json=body) as r:
        assert r.status_code == 200
        return read_sse(r)


def test_chat_streams_sources_then_tokens_then_done(client, llm):
    events = _chat(client, question="How many credits are the computer science major requirements?")
    names = [e for e, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert set(names[1:-1]) == {"token"}
    sources = events[0][1]["sources"]
    assert sources[0]["heading"] == "Major Requirements (56 credits)"
    assert sources[0]["n"] == 1 and "page" in sources[0] and "score" in sources[0]
    answer = "".join(d["text"] for e, d in events if e == "token")
    assert "56 credits [1]" in answer
    assert events[-1][1]["model"] == "llama3.1:8b"
    assert len(llm.calls) == 1


def test_chat_refuses_off_topic_without_calling_llm(client, llm):
    events = _chat(client, question="Who won the football world cup?")
    assert [e for e, _ in events] == ["refusal"]
    assert events[0][1]["top_score"] < events[0][1]["threshold"]
    assert llm.calls == []


def test_chat_rejects_model_outside_allowlist(client):
    r = client.post("/api/chat", json={"question": "credits?", "model": "gpt-4"})
    assert r.status_code == 400


def test_chat_accepts_allowlisted_model(client):
    events = _chat(client, question="computer science major credits", model="qwen2.5:0.5b")
    assert events[-1] == ("done", {"model": "qwen2.5:0.5b", "elapsed_ms": events[-1][1]["elapsed_ms"]})


def test_chat_reports_llm_failure_as_error_event(client, settings, store):
    from api.routes_chat import llm_factory_dep

    client.app.dependency_overrides[llm_factory_dep] = lambda: (lambda _s, _m: FakeLLM(fail=True))
    events = _chat(client, question="computer science major credits")
    assert events[0][0] == "sources"
    assert events[-1][0] == "error" and "Ollama is down" in events[-1][1]["message"]


def test_followup_uses_previous_question_for_retrieval(client):
    history = [
        {"role": "user", "content": "Tell me about CS211 Operating Systems"},
        {"role": "assistant", "content": "It covers processes and threads [1]."},
    ]
    events = _chat(client, question="what does it cover", history=history)
    assert events[0][0] == "sources"
    assert events[0][1]["sources"][0]["heading"] == "CS211: Operating Systems"


# ---- prompt / retrieval units ----------------------------------------------------------
def test_retrieval_query_appends_previous_user_question():
    h = [Turn("user", "What is CS311?"), Turn("assistant", "A course [1].")]
    assert retrieval_query("prerequisites?", h) == "What is CS311?\nprerequisites?"
    assert retrieval_query("prerequisites?", []) == "prerequisites?"


def test_prompt_is_grounded_numbered_and_trims_history(store):
    hits = retrieve(store, "capstone GPA", 2)
    history = [Turn("user", f"q{i}") if i % 2 == 0 else Turn("assistant", f"a{i}") for i in range(10)]
    msgs = build_messages("capstone GPA?", hits, history, history_turns=3)
    assert NOT_FOUND in msgs[0].content and "ONLY" in msgs[0].content
    assert len(msgs) == 1 + 6 + 1  # system + 3 Q/A pairs + current
    last = msgs[-1].content
    assert "[1] (hb.pdf, p.9 — Capstone)" in last and last.rstrip().endswith("QUESTION: capstone GPA?")


def test_hybrid_finds_exact_course_code(store):
    hits = retrieve(store, "prerequisites for CS311", 3, mode="hybrid")
    assert hits[0].chunk.metadata["heading"] == "CS311: Advanced Deep Learning"
    assert hits[0].bm25_rank == 1
    assert all(0.0 <= h.score <= 1.0001 for h in hits)


def test_dense_mode_reports_dense_ranks_only(store):
    hits = retrieve(store, "operating systems", 2, mode="dense")
    assert [h.dense_rank for h in hits] == [1, 2] and all(h.bm25_rank is None for h in hits)


# ---- knowledge-base routes ---------------------------------------------------------------
def test_health_degraded_when_ollama_unreachable(client):
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is False
    assert body["chroma"]["reachable"] is True and body["chroma"]["chunks"] == 4


def test_retrieve_returns_scores_and_refusal_flag(client):
    on = client.post("/api/retrieve", json={"question": "capstone GPA", "k": 2}).json()
    assert on["hits"][0]["heading"] == "Capstone" and on["would_refuse"] is False
    off = client.post("/api/retrieve", json={"question": "sourdough bread recipe"}).json()
    assert off["would_refuse"] is True


def test_chunks_listing_filters_and_paginates(client):
    all_ = client.get("/api/chunks").json()
    assert all_["total"] == 4
    assert [c["page"] for c in all_["items"]] == sorted(c["page"] for c in all_["items"])
    assert client.get("/api/chunks", params={"page": 18}).json()["items"][0]["heading"].startswith("CS311")
    assert client.get("/api/chunks", params={"contains": "GPA"}).json()["total"] == 1
    page2 = client.get("/api/chunks", params={"limit": 3, "offset": 3}).json()
    assert page2["total"] == 4 and len(page2["items"]) == 1


def test_get_single_chunk_and_404(client):
    assert client.get("/api/chunks/c0").json()["id"] == "c0"
    assert client.get("/api/chunks/nope").status_code == 404


def test_upload_list_and_delete_markdown(client, settings):
    md = b"# Club Rules\n## Meetings\nThe robotics club meets every Friday at 5pm in room 101.\n"
    r = client.post("/api/documents", files={"file": ("club.md", md, "text/markdown")})
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "indexed" and r.json()["chunks"] >= 1
    assert (settings.data_path / "uploads" / "club.md").exists()

    again = client.post("/api/documents", files={"file": ("club.md", md, "text/markdown")})
    assert again.json()["status"] == "unchanged"

    docs = {d["source"]: d for d in client.get("/api/documents").json()["documents"]}
    assert docs["club.md"]["chunks"] >= 1 and docs["hb.pdf"]["chunks"] == 4

    hit = client.post("/api/retrieve", json={"question": "when does the robotics club meet"}).json()["hits"][0]
    assert hit["source"] == "club.md" and hit["heading"] == "Meetings"

    assert client.delete("/api/documents/club.md").json()["deleted_chunks"] >= 1
    assert client.delete("/api/documents/club.md").status_code == 404
    assert not (settings.data_path / "uploads" / "club.md").exists()


def test_upload_rejects_unsupported_type(client):
    r = client.post("/api/documents", files={"file": ("x.docx", b"hi", "application/octet-stream")})
    assert r.status_code == 400
