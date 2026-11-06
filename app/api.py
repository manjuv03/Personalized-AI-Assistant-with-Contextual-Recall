"""FastAPI app. Run with:  uvicorn app.api:app --reload"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .retry import ModelBusyError
from .service import Assistant

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@lru_cache(maxsize=1)
def get_assistant() -> Assistant:
    return Assistant()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the embedding model and vector store at startup, so the first message isn't slow."""
    try:
        get_assistant().memory.embedder.embed_query("warm up")
    except Exception:  # noqa: BLE001 - the app should still start; errors show on first use
        log.exception("Warm-up failed")
    yield


app = FastAPI(title="Personalised AI Assistant with Contextual Recall", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(ModelBusyError)
def model_busy(_: Request, exc: ModelBusyError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
def unexpected(_: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"[:300]})


class ChatRequest(BaseModel):
    user_id: str = Field(default="demo", min_length=1, max_length=64)
    thread_id: str = Field(default="main", min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4000)
    use_memory: bool = True


class ChatResponse(BaseModel):
    reply: str
    recalled: list[dict]
    latency_ms: float
    log_id: int


class MemoryIn(BaseModel):
    text: str = Field(min_length=3, max_length=500)
    category: str = "other"
    importance: float = Field(default=0.6, ge=0, le=1)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, background: BackgroundTasks, assistant: Assistant = Depends(get_assistant)):
    result = assistant.chat(req.user_id, req.thread_id, req.message, use_memory=req.use_memory)
    if req.use_memory:
        # Saving new facts happens after the reply is sent.
        background.add_task(assistant.remember_safely, req.user_id, req.message, result.log_id)
    return ChatResponse(
        reply=result.reply, recalled=result.recalled, latency_ms=round(result.latency_ms, 1), log_id=result.log_id
    )


@app.get("/users/{user_id}/memories")
def list_memories(user_id: str, assistant: Assistant = Depends(get_assistant)) -> list[dict]:
    return [
        {"id": m.id, "text": m.text, "category": m.category, "importance": m.importance, "updated_at": m.updated_at}
        for m in assistant.memory.list(user_id)
    ]


@app.post("/users/{user_id}/memories")
def add_memory(user_id: str, body: MemoryIn, assistant: Assistant = Depends(get_assistant)) -> dict:
    rec = assistant.memory.add(user_id, body.text, body.category, body.importance)
    return {"id": rec.id, "text": rec.text, "category": rec.category}


@app.delete("/users/{user_id}/memories/{memory_id}")
def delete_memory(user_id: str, memory_id: str, assistant: Assistant = Depends(get_assistant)) -> dict:
    if not assistant.memory.delete(user_id, memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": memory_id}


@app.get("/users/{user_id}/memory-map")
def memory_map(user_id: str, assistant: Assistant = Depends(get_assistant)) -> list[dict]:
    return assistant.memory.projection(user_id)


@app.get("/users/{user_id}/threads/{thread_id}/summary")
def thread_summary(user_id: str, thread_id: str, assistant: Assistant = Depends(get_assistant)) -> dict:
    return {"summary": assistant.summary(user_id, thread_id)}


@app.get("/users/{user_id}/logs")
def logs(user_id: str, limit: int = 50, assistant: Assistant = Depends(get_assistant)) -> dict:
    return {"stats": assistant.log.stats(user_id), "items": assistant.log.recent(user_id, min(limit, 200))}
