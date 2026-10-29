import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import chat as chat_service
from app.services.embeddings import EmbeddingError
from app.services.llm import LLMError

logger = logging.getLogger(__name__)
router = APIRouter()


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=10000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Optional earlier turns, oldest first, so follow-up questions keep their context
    history: list[HistoryMessage] = Field(default_factory=list, max_length=50)


class Source(BaseModel):
    title: str
    url: str


class Snippet(Source):
    text: str


class ChatResponse(BaseModel):
    answer: str
    found: bool
    sources: list[Source]
    snippets: list[Snippet]


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank.")
    try:
        history = [m.model_dump() for m in req.history if m.content.strip()]
        return chat_service.answer_question(question, history)
    except (LLMError, EmbeddingError) as exc:
        logger.warning("Chat failed (LLM): %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail="Something went wrong while answering. Check the backend logs.")
