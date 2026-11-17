from typing import Literal

from pydantic import BaseModel, Field


class TurnIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[TurnIn] = Field(default_factory=list)
    model: str | None = None


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)
