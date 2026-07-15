from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .core import SecondBrain
from .improvement import run_improvement_cycle
from .prompt_store import load_prompt


STATIC_DIR = Path(__file__).with_name("static")
settings = Settings.from_env()
brain = SecondBrain(settings)

app = FastAPI(
    title="Second Brain",
    version="0.2.0",
    description="A memory-backed AI with measured, gated self-improvement.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)


class FactRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10_000)
    importance: float = Field(default=1.0, gt=0, le=10)


class FeedbackRequest(BaseModel):
    interaction_id: int = Field(gt=0)
    score: float = Field(ge=0, le=1)
    note: str = Field(default="", max_length=2_000)


class ImproveRequest(BaseModel):
    auto_promote: bool = False
    minimum_gain: float = Field(default=0.02, ge=0, le=1)
    maximum_case_regression: float = Field(default=0.10, ge=0, le=1)


def _optional_token(
    expected: str,
    provided: str | None,
    *,
    missing_configuration_message: str | None = None,
) -> None:
    if not expected:
        if missing_configuration_message:
            raise HTTPException(status_code=503, detail=missing_configuration_message)
        return
    if provided is None or not secrets.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="Invalid or missing access token.")


def require_access(
    x_access_token: Annotated[str | None, Header()] = None,
) -> None:
    _optional_token(os.getenv("SECOND_BRAIN_ACCESS_TOKEN", "").strip(), x_access_token)


def require_admin(
    x_admin_token: Annotated[str | None, Header()] = None,
) -> None:
    _optional_token(
        os.getenv("SECOND_BRAIN_ADMIN_TOKEN", "").strip(),
        x_admin_token,
        missing_configuration_message=(
            "Self-improvement is disabled until SECOND_BRAIN_ADMIN_TOKEN is configured."
        ),
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "second-brain"}


@app.get("/api/status")
def status(_: None = Depends(require_access)) -> dict[str, object]:
    prompt = load_prompt(settings.active_prompt_path)
    return {
        "model": settings.model,
        "prompt_version": prompt.version,
        "prompt_name": prompt.name,
        "memory": brain.memory.stats(),
        "improvement_enabled": bool(os.getenv("SECOND_BRAIN_ADMIN_TOKEN", "").strip()),
        "access_protected": bool(os.getenv("SECOND_BRAIN_ACCESS_TOKEN", "").strip()),
    }


@app.get("/api/history")
def history(
    limit: int = 30,
    _: None = Depends(require_access),
) -> dict[str, object]:
    limit = max(1, min(limit, 100))
    return {"items": brain.memory.recent_interactions(limit)}


@app.post("/api/chat")
def chat(request: ChatRequest, _: None = Depends(require_access)) -> dict[str, object]:
    try:
        answer = brain.answer(request.message)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model request failed: {exc}") from exc
    return {
        "answer": answer.text,
        "interaction_id": answer.interaction_id,
        "prompt_version": answer.prompt_version,
        "memories_used": answer.memories_used,
    }


@app.post("/api/facts")
def add_fact(request: FactRequest, _: None = Depends(require_access)) -> dict[str, object]:
    try:
        fact_id = brain.memory.add_fact(
            request.content,
            importance=request.importance,
            source="web",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"fact_id": fact_id}


@app.post("/api/feedback")
def feedback(
    request: FeedbackRequest,
    _: None = Depends(require_access),
) -> dict[str, object]:
    try:
        brain.memory.record_feedback(
            request.interaction_id,
            request.score,
            request.note,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/api/improve")
def improve(
    request: ImproveRequest,
    _: None = Depends(require_admin),
) -> dict[str, object]:
    try:
        result = run_improvement_cycle(
            settings,
            auto_promote=request.auto_promote,
            minimum_gain=request.minimum_gain,
            maximum_case_regression=request.maximum_case_regression,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Improvement cycle failed: {exc}") from exc

    return {
        "baseline_score": result.decision.baseline_score,
        "candidate_score": result.decision.candidate_score,
        "accepted": result.decision.accepted,
        "promoted": result.promoted,
        "reasons": list(result.decision.reasons),
        "candidate_path": str(result.candidate_path),
    }


def run() -> None:
    import uvicorn

    uvicorn.run(
        "second_brain.web:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
