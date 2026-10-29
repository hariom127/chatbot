import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import chat as chat_service
from app.services.embeddings import EmbeddingError
from app.services.llm import LLMError

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


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
        return chat_service.answer_question(question)
    except (LLMError, EmbeddingError) as exc:
        logger.warning("Chat failed (LLM): %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail="Something went wrong while answering. Check the backend logs.")
