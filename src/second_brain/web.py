from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .core import SecondBrain
from .improvement import run_improvement_cycle
from .prompt_store import load_prompt
from .training_orchestrator import TrainingOrchestrator, TrainingOrchestratorError
from .zero_runtime import ZeroRuntime, ZeroUnavailableError


STATIC_DIR = Path(__file__).with_name("static")
settings = Settings.from_env()
brain = SecondBrain(settings)
zero_runtime = ZeroRuntime()
training_orchestrator = TrainingOrchestrator(activation_path=zero_runtime.checkpoint_path)

app = FastAPI(
    title="Second Brain",
    version="0.4.0",
    description="A memory-backed AI and an autonomous scratch-Transformer laboratory.",
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


class ZeroGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8_192)
    max_new_tokens: int = Field(default=160, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.05, le=3.0)
    top_k: int = Field(default=50, ge=1, le=256)
    seed: int = Field(default=1337, ge=0, le=2_147_483_647)


class TrainingStartRequest(BaseModel):
    generation: str = Field(default="level1", min_length=1, max_length=40)
    max_steps: int | None = Field(default=None, ge=1, le=2_000_000)
    auto_prepare: bool = True
    resume_existing: bool = True
    auto_activate_best: bool = True
    auto_advance: bool = False
    initialize_from_previous: bool = True
    resume_after_restart: bool = True
    target_validation: float | None = Field(default=None, gt=0, le=20)
    max_parameters: int = Field(default=40_000_000, ge=10_000, le=2_000_000_000)


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
            "Administrative actions are disabled until SECOND_BRAIN_ADMIN_TOKEN is configured."
        ),
    )


@app.on_event("startup")
def recover_training_after_restart() -> None:
    try:
        training_orchestrator.recover()
    except Exception:
        # Recovery state is exposed in the dashboard; it must not prevent the web server starting.
        pass


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/zero")
def zero_lab() -> FileResponse:
    return FileResponse(STATIC_DIR / "zero.html")


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
        "zero": zero_runtime.status(attempt_load=False),
        "training": training_orchestrator.status(),
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


@app.get("/api/zero/status")
def zero_status(_: None = Depends(require_access)) -> dict[str, object]:
    return zero_runtime.status()


@app.post("/api/zero/generate")
def zero_generate(
    request: ZeroGenerateRequest,
    _: None = Depends(require_access),
) -> dict[str, object]:
    try:
        return zero_runtime.generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            seed=request.seed,
        )
    except ZeroUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local generation failed: {exc}") from exc


@app.post("/api/zero/checkpoint")
async def zero_upload_checkpoint(
    checkpoint: UploadFile = File(...),
    _: None = Depends(require_admin),
) -> dict[str, object]:
    filename = checkpoint.filename or ""
    if Path(filename).suffix.lower() not in {".pt", ".pth"}:
        raise HTTPException(status_code=400, detail="Upload a .pt or .pth checkpoint.")

    max_bytes = int(os.getenv("SECOND_BRAIN_ZERO_MAX_CHECKPOINT_MB", "512")) * 1024 * 1024
    target_parent = zero_runtime.checkpoint_path.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    size = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".pt",
            prefix="zero-upload-",
            dir=target_parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            while chunk := await checkpoint.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Checkpoint exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                    )
                handle.write(chunk)
        result = zero_runtime.install_checkpoint(temporary_path)
        temporary_path = None
        return {"ok": True, "size_bytes": size, "status": result}
    except HTTPException:
        raise
    except ZeroUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Checkpoint activation failed: {exc}") from exc
    finally:
        await checkpoint.close()
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


@app.get("/api/zero/training/catalog")
def zero_training_catalog(_: None = Depends(require_access)) -> dict[str, object]:
    try:
        return training_orchestrator.catalog()
    except TrainingOrchestratorError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/zero/training/status")
def zero_training_status(_: None = Depends(require_access)) -> dict[str, object]:
    return training_orchestrator.status()


@app.post("/api/zero/training/start")
def zero_training_start(
    request: TrainingStartRequest,
    _: None = Depends(require_admin),
) -> dict[str, object]:
    try:
        return training_orchestrator.start(
            generation=request.generation,
            max_steps=request.max_steps,
            auto_prepare=request.auto_prepare,
            resume_existing=request.resume_existing,
            auto_activate_best=request.auto_activate_best,
            auto_advance=request.auto_advance,
            initialize_from_previous=request.initialize_from_previous,
            resume_after_restart=request.resume_after_restart,
            target_validation=request.target_validation,
            max_parameters=request.max_parameters,
        )
    except TrainingOrchestratorError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/zero/training/pause")
def zero_training_pause(_: None = Depends(require_admin)) -> dict[str, object]:
    try:
        return training_orchestrator.pause()
    except TrainingOrchestratorError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/zero/training/resume")
def zero_training_resume(_: None = Depends(require_admin)) -> dict[str, object]:
    try:
        return training_orchestrator.resume()
    except TrainingOrchestratorError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/zero/training/stop")
def zero_training_stop(_: None = Depends(require_admin)) -> dict[str, object]:
    try:
        return training_orchestrator.stop()
    except TrainingOrchestratorError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def run() -> None:
    import uvicorn

    uvicorn.run(
        "second_brain.web:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
