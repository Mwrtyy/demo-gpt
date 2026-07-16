from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ZeroUnavailableError(RuntimeError):
    """Raised when the local scratch model cannot serve a generation request."""


@dataclass(frozen=True)
class LoadedZeroModel:
    torch: Any
    model: Any
    tokenizer: Any
    device: Any
    checkpoint_mtime_ns: int
    step: int | None
    best_validation: float | None
    parameters: int
    block_size: int


class ZeroRuntime:
    """Lazy, thread-safe inference runtime for a Second Brain Zero checkpoint."""

    def __init__(self, checkpoint_path: Path | None = None, device: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path or Path(
            os.getenv("SECOND_BRAIN_ZERO_CHECKPOINT", "runtime/zero/latest.pt")
        )
        self.requested_device = device or os.getenv("SECOND_BRAIN_ZERO_DEVICE", "auto")
        self._loaded: LoadedZeroModel | None = None
        self._load_error: str | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _dependencies() -> tuple[Any, Any, Any]:
        try:
            import torch
            from scratch.model import ByteGPT, ModelConfig
            from scratch.tokenizer import ByteTokenizer
        except ImportError as exc:
            raise ZeroUnavailableError(
                "Second Brain Zero dependencies are missing. Install scratch/requirements.txt."
            ) from exc
        return torch, (ByteGPT, ModelConfig), ByteTokenizer

    def _choose_device(self, torch: Any) -> Any:
        requested = self.requested_device.strip().lower()
        if requested != "auto":
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_from(self, path: Path) -> LoadedZeroModel:
        torch, model_types, tokenizer_type = self._dependencies()
        ByteGPT, ModelConfig = model_types
        device = self._choose_device(torch)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ZeroUnavailableError(f"Checkpoint could not be read safely: {exc}") from exc
        if not isinstance(checkpoint, dict):
            raise ZeroUnavailableError("Checkpoint must contain a dictionary.")
        if not isinstance(checkpoint.get("model_config"), dict) or "model" not in checkpoint:
            raise ZeroUnavailableError("Checkpoint is missing model or model_config.")

        config = ModelConfig.from_dict(checkpoint["model_config"])
        model = ByteGPT(config)
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
        except Exception as exc:
            raise ZeroUnavailableError(f"Checkpoint weights do not match the architecture: {exc}") from exc
        model = model.to(device)
        model.eval()

        step_raw = checkpoint.get("step")
        validation_raw = checkpoint.get("best_validation")
        return LoadedZeroModel(
            torch=torch,
            model=model,
            tokenizer=tokenizer_type(),
            device=device,
            checkpoint_mtime_ns=path.stat().st_mtime_ns,
            step=int(step_raw) if step_raw is not None else None,
            best_validation=float(validation_raw) if validation_raw is not None else None,
            parameters=int(model.parameter_count()),
            block_size=int(config.block_size),
        )

    def _ensure_loaded(self) -> LoadedZeroModel:
        with self._lock:
            if not self.checkpoint_path.exists():
                raise ZeroUnavailableError(
                    f"No trained checkpoint found at {self.checkpoint_path}. Train or upload one first."
                )
            mtime = self.checkpoint_path.stat().st_mtime_ns
            if self._loaded is None or self._loaded.checkpoint_mtime_ns != mtime:
                try:
                    self._loaded = self._load_from(self.checkpoint_path)
                    self._load_error = None
                except Exception as exc:
                    self._loaded = None
                    self._load_error = str(exc)
                    raise
            return self._loaded

    def status(self, *, attempt_load: bool = True) -> dict[str, object]:
        dependencies_available = True
        dependency_error = None
        try:
            self._dependencies()
        except ZeroUnavailableError as exc:
            dependencies_available = False
            dependency_error = str(exc)

        present = self.checkpoint_path.exists()
        if attempt_load and dependencies_available and present:
            try:
                self._ensure_loaded()
            except Exception:
                pass

        loaded = self._loaded
        size = self.checkpoint_path.stat().st_size if present else None
        return {
            "dependencies_available": dependencies_available,
            "dependency_error": dependency_error,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_present": present,
            "checkpoint_size_bytes": size,
            "ready": loaded is not None,
            "load_error": self._load_error,
            "device": str(loaded.device) if loaded else None,
            "parameters": loaded.parameters if loaded else None,
            "block_size": loaded.block_size if loaded else None,
            "step": loaded.step if loaded else None,
            "best_validation": loaded.best_validation if loaded else None,
        }

    def install_checkpoint(self, uploaded_path: Path) -> dict[str, object]:
        """Validate an administrator-provided checkpoint before atomically activating it."""
        with self._lock:
            candidate = self._load_from(uploaded_path)
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(uploaded_path, self.checkpoint_path)
            candidate = LoadedZeroModel(
                **{
                    **candidate.__dict__,
                    "checkpoint_mtime_ns": self.checkpoint_path.stat().st_mtime_ns,
                }
            )
            self._loaded = candidate
            self._load_error = None
            return self.status(attempt_load=False)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 160,
        temperature: float = 0.8,
        top_k: int = 50,
        seed: int = 1337,
    ) -> dict[str, object]:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 1 and 512.")
        if not 0.05 <= temperature <= 3.0:
            raise ValueError("temperature must be between 0.05 and 3.0.")
        if not 1 <= top_k <= 256:
            raise ValueError("top_k must be between 1 and 256.")

        with self._lock:
            loaded = self._ensure_loaded()
            torch = loaded.torch
            torch.manual_seed(seed)
            if loaded.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            prompt_tokens = loaded.tokenizer.encode(prompt)
            if len(prompt_tokens) > 8_192:
                raise ValueError("Prompt is too large; use at most 8192 UTF-8 bytes.")
            tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=loaded.device)
            started = time.perf_counter()
            generated = loaded.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            elapsed = time.perf_counter() - started
            generated_tokens = generated[0].tolist()
            continuation_tokens = generated_tokens[len(prompt_tokens) :]
            return {
                "text": loaded.tokenizer.decode(generated_tokens),
                "continuation": loaded.tokenizer.decode(continuation_tokens),
                "new_tokens": len(continuation_tokens),
                "elapsed_seconds": round(elapsed, 4),
                "tokens_per_second": round(len(continuation_tokens) / max(elapsed, 1e-9), 2),
                "device": str(loaded.device),
                "step": loaded.step,
                "parameters": loaded.parameters,
            }
