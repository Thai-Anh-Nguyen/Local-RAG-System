"""Prompt construction and streaming answer generation.

Answer policy (vs. the baseline `run_txt.py`):
- grounded: answer only from the numbered excerpts; no fallback to the model's general knowledge
- cited: every claim carries [n] markers that map to the sources shown in the UI
- low temperature (0.1) for factual, reproducible answers
- refusal happens *before* this module when retrieval relevance is below the threshold
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .config import Settings
from .retrieval import Hit, Turn

NOT_FOUND = "I couldn't find this in the handbook."

SYSTEM_PROMPT = f"""You are a helpful assistant for students reading a university major handbook \
(the B.S. Computer Science handbook at Fulbright University Vietnam, plus any documents the user uploaded).

Rules:
1. Answer ONLY with information from the numbered EXCERPTS in the user's message. Do not use outside knowledge.
2. After every sentence that uses an excerpt, cite it with its number in square brackets, e.g. [1] or [2][3].
3. If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}" — do not guess.
4. If the excerpts only partly answer the question, give the part they support and say what is missing.
5. Be concise: 2-5 sentences, or a short bulleted list for lists of courses or requirements.
6. Reply in the same language as the question.
7. Earlier conversation turns are only for understanding what the question refers to (e.g. "it", "that course")."""


def format_excerpts(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        m = h.chunk.metadata
        where = f"{m.get('source', '?')}, p.{m.get('page', '?')}"
        if m.get("heading"):
            where += f" — {m['heading']}"
        blocks.append(f"[{h.n}] ({where})\n{h.chunk.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit], history: list[Turn], history_turns: int) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
    # history_turns = question/answer pairs -> 2 messages each
    for t in history[-2 * history_turns:] if history_turns > 0 else []:
        messages.append(HumanMessage(t.content) if t.role == "user" else AIMessage(t.content))
    messages.append(HumanMessage(f"EXCERPTS:\n\n{format_excerpts(hits)}\n\n---\nQUESTION: {question}"))
    return messages


class ChatModel(Protocol):
    def astream(self, messages: list[BaseMessage]) -> AsyncIterator: ...


def make_llm(settings: Settings, model: str | None = None) -> ChatModel:
    return ChatOllama(
        model=model or settings.llm_model,
        base_url=settings.ollama_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
    )


async def stream_answer(llm: ChatModel, messages: list[BaseMessage]) -> AsyncIterator[str]:
    async for chunk in llm.astream(messages):
        text = chunk.content if isinstance(chunk.content, str) else ""
        if text:
            yield text
