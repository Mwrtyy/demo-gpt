from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTIVE_STATUSES = {
    "preparing",
    "starting",
    "running",
    "pause_requested",
    "paused",
    "resume_requested",
    "stopping",
    "advancing",
}


class TrainingOrchestratorError(RuntimeError):
    """Raised when a training control request cannot be completed safely."""


@dataclass(frozen=True)
class Generation:
    identifier: str
    name: str
    config_path: Path
    out_dir: Path
    target_validation: float
    parameters: int
    default_max_steps: int


class TrainingOrchestrator:
    """Starts and controls scratch training jobs from the local web server."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        state_dir: Path | None = None,
        activation_path: Path | None = None,
    ) -> None:
        package_root = Path(__file__).resolve().parents[2]
        self.project_root = (project_root or package_root).resolve()
        configured_state = Path(os.getenv("SECOND_BRAIN_TRAINING_DIR", "runtime/training"))
        self.state_dir = (state_dir or self.project_root / configured_state).resolve()
        configured_checkpoint = Path(
            os.getenv("SECOND_BRAIN_ZERO_CHECKPOINT", "runtime/zero/latest.pt")
        )
        self.activation_path = (
            activation_path or self.project_root / configured_checkpoint
        ).resolve()
        self.state_path = self.state_dir / "state.json"
        self.plan_path = self.project_root / "scratch/configs/growth_plan.json"
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None

    @staticmethod
    def _model_parameter_count(model: dict[str, Any]) -> int:
        vocab = int(model["vocab_size"])
        block = int(model["block_size"])
        layers = int(model["n_layer"])
        embedding = int(model["n_embd"])
        bias = bool(model.get("bias", False))
        embeddings = (vocab + block) * embedding
        linear_and_norm = 13 * embedding if bias else 2 * embedding
        per_layer = 12 * embedding * embedding + linear_and_norm
        final_norm = 2 * embedding if bias else embedding
        return embeddings + layers * per_layer + final_norm

    def generations(self) -> list[Generation]:
        try:
            raw = json.loads(self.plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingOrchestratorError(f"Growth plan could not be read: {exc}") from exc
        entries = raw.get("generations")
        if not isinstance(entries, list) or not entries:
            raise TrainingOrchestratorError("Growth plan must define at least one generation.")

        generations: list[Generation] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise TrainingOrchestratorError("Each growth-plan generation must be an object.")
            config_path = (self.project_root / str(entry["config"])).resolve()
            out_dir = (self.project_root / str(entry["out_dir"])).resolve()
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model = config.get("model")
            training = config.get("training")
            if not isinstance(model, dict) or not isinstance(training, dict):
                raise TrainingOrchestratorError(f"Invalid model configuration: {config_path}")
            generations.append(
                Generation(
                    identifier=str(entry["id"]),
                    name=str(entry["name"]),
                    config_path=config_path,
                    out_dir=out_dir,
                    target_validation=float(entry["target_validation"]),
                    parameters=self._model_parameter_count(model),
                    default_max_steps=int(training["max_steps"]),
                )
            )
        return generations

    def catalog(self) -> dict[str, object]:
        items = []
        for generation in self.generations():
            items.append(
                {
                    "id": generation.identifier,
                    "name": generation.name,
                    "config_path": str(generation.config_path.relative_to(self.project_root)),
                    "out_dir": str(generation.out_dir.relative_to(self.project_root)),
                    "target_validation": generation.target_validation,
                    "parameters": generation.parameters,
                    "default_max_steps": generation.default_max_steps,
                    "latest_checkpoint_present": (generation.out_dir / "latest.pt").exists(),
                    "best_checkpoint_present": (generation.out_dir / "best.pt").exists(),
                }
            )
        return {"generations": items, "activation_path": str(self.activation_path)}

    def _generation(self, identifier: str) -> Generation:
        for generation in self.generations():
            if generation.identifier == identifier:
                return generation
        raise TrainingOrchestratorError(f"Unknown generation: {identifier}")

    def _generation_after(self, identifier: str) -> Generation | None:
        generations = self.generations()
        for index, generation in enumerate(generations):
            if generation.identifier == identifier:
                return generations[index + 1] if index + 1 < len(generations) else None
        return None

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "status": "idle",
                "generation": None,
                "pid": None,
                "updated_at": time.time(),
            }
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "status": "failed",
                "generation": None,
                "pid": None,
                "error": "Training state file is unreadable.",
                "updated_at": time.time(),
            }
        return state if isinstance(state, dict) else {"status": "failed"}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = time.time()
        temporary = self.state_path.with_name(f".{self.state_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temporary, self.state_path)

    @staticmethod
    def _pid_is_alive(pid: int | None) -> bool:
        if not pid or pid < 1:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @staticmethod
    def _write_control(path: Path, command: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps({"command": command, "updated_at": time.time()}), encoding="utf-8"
        )
        os.replace(temporary, path)

    @staticmethod
    def _read_events(path: Path | None, limit: int = 120) -> list[dict[str, Any]]:
        if path is None or not path.exists():
            return []
        records: deque[dict[str, Any]] = deque(maxlen=limit)
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except OSError:
            return []
        return list(records)

    @staticmethod
    def _tail_text(path: Path | None, lines: int = 40) -> str:
        if path is None or not path.exists():
            return ""
        content: deque[str] = deque(maxlen=lines)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                content.extend(line.rstrip() for line in handle)
        except OSError:
            return ""
        return "\n".join(content)

    @staticmethod
    def _metrics_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for event in events:
            event_type = event.get("event")
            if event_type == "started":
                metrics.update(
                    {
                        "device": event.get("device"),
                        "parameters": event.get("parameters"),
                        "max_steps": event.get("max_steps"),
                    }
                )
            elif event_type == "progress":
                metrics.update(
                    {
                        "step": event.get("step"),
                        "max_steps": event.get("max_steps"),
                        "loss": event.get("loss"),
                        "learning_rate": event.get("learning_rate"),
                        "steps_per_second": event.get("steps_per_second"),
                        "sequences_per_second": event.get("sequences_per_second"),
                        "elapsed_seconds": event.get("elapsed_seconds"),
                        "eta_seconds": event.get("eta_seconds"),
                    }
                )
            elif event_type == "evaluation":
                metrics.update(
                    {
                        "step": event.get("step"),
                        "train_loss": event.get("train_loss"),
                        "validation_loss": event.get("validation_loss"),
                        "best_validation": event.get("best_validation"),
                        "improved": event.get("improved"),
                    }
                )
            elif event_type == "checkpoint":
                metrics.update(
                    {
                        "latest_checkpoint": event.get("latest") or metrics.get("latest_checkpoint"),
                        "best_checkpoint": event.get("best") or metrics.get("best_checkpoint"),
                        "activated_checkpoint": event.get("activated")
                        or metrics.get("activated_checkpoint"),
                        "best_validation": event.get("best_validation"),
                    }
                )
            elif event_type in {"completed", "stopped"}:
                metrics.update(
                    {
                        "step": event.get("step"),
                        "best_validation": event.get("best_validation"),
                        "latest_checkpoint": event.get("latest")
                        or event.get("checkpoint")
                        or metrics.get("latest_checkpoint"),
                        "best_checkpoint": event.get("best")
                        or metrics.get("best_checkpoint"),
                    }
                )
        step = metrics.get("step")
        max_steps = metrics.get("max_steps")
        if isinstance(step, int) and isinstance(max_steps, int) and max_steps > 0:
            metrics["progress_percent"] = min(100.0, max(0.0, (step + 1) / max_steps * 100))
        else:
            metrics["progress_percent"] = 0.0
        return metrics

    def status(self) -> dict[str, object]:
        with self._lock:
            state = self._load_state()
            events_path = Path(state["events_path"]) if state.get("events_path") else None
            log_path = Path(state["log_path"]) if state.get("log_path") else None
            events = self._read_events(events_path)
            metrics = self._metrics_from_events(events)
            status = str(state.get("status", "idle"))
            pid = int(state["pid"]) if state.get("pid") else None
            alive = self._pid_is_alive(pid)

            if status in ACTIVE_STATUSES and not alive and not (
                self._worker and self._worker.is_alive()
            ):
                last_event = events[-1].get("event") if events else None
                if last_event == "completed":
                    status = "completed"
                elif last_event == "stopped":
                    status = "stopped"
                elif last_event == "error":
                    status = "failed"
                else:
                    status = "interrupted"
                state["status"] = status
                state["pid"] = None
                self._save_state(state)

            if events:
                last_event = events[-1].get("event")
                if last_event == "paused":
                    status = "paused"
                elif last_event == "resumed" and status != "stopping":
                    status = "running"
                elif last_event == "stop_requested":
                    status = "stopping"

            return {
                **state,
                **metrics,
                "status": status,
                "active": status in ACTIVE_STATUSES,
                "process_alive": alive,
                "recent_events": events[-20:],
                "log_tail": self._tail_text(log_path),
            }

    def start(
        self,
        *,
        generation: str,
        max_steps: int | None,
        auto_prepare: bool,
        resume_existing: bool,
        auto_activate_best: bool,
        auto_advance: bool,
        initialize_from_previous: bool,
        resume_after_restart: bool,
        target_validation: float | None,
        max_parameters: int,
    ) -> dict[str, object]:
        with self._lock:
            current = self.status()
            if current["active"]:
                raise TrainingOrchestratorError("A training job is already active.")
            selected = self._generation(generation)
            if selected.parameters > max_parameters:
                raise TrainingOrchestratorError(
                    f"{selected.name} has {selected.parameters:,} parameters, above the configured "
                    f"limit of {max_parameters:,}."
                )
            requested_steps = max_steps or selected.default_max_steps
            if requested_steps < 1 or requested_steps > 2_000_000:
                raise TrainingOrchestratorError("max_steps must be between 1 and 2,000,000.")

            run_id = time.strftime("%Y%m%d-%H%M%S")
            run_dir = self.state_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "run_id": run_id,
                "status": "preparing" if auto_prepare else "starting",
                "stage": "queued",
                "generation": selected.identifier,
                "generation_name": selected.name,
                "pid": None,
                "started_at": time.time(),
                "events_path": str(run_dir / "events.jsonl"),
                "control_path": str(run_dir / "control.json"),
                "log_path": str(run_dir / "training.log"),
                "max_steps": requested_steps,
                "auto_prepare": auto_prepare,
                "resume_existing": resume_existing,
                "auto_activate_best": auto_activate_best,
                "auto_advance": auto_advance,
                "initialize_from_previous": initialize_from_previous,
                "resume_after_restart": resume_after_restart,
                "target_validation": target_validation
                if target_validation is not None
                else selected.target_validation,
                "max_parameters": max_parameters,
                "error": None,
            }
            self._save_state(state)
            self._worker = threading.Thread(
                target=self._pipeline,
                args=(state.copy(),),
                name="second-brain-training",
                daemon=True,
            )
            self._worker.start()
            return self.status()

    def _append_log(self, log_path: Path, text: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")

    def _run_setup_command(self, command: list[str], log_path: Path) -> None:
        self._append_log(log_path, f"$ {' '.join(command)}")
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self._append_log(log_path, completed.stdout or "")
        if completed.returncode != 0:
            raise TrainingOrchestratorError(
                f"Setup command failed with exit code {completed.returncode}: {' '.join(command)}"
            )

    def _prepare_corpus(self, state: dict[str, Any]) -> None:
        prepared = self.project_root / "scratch/data/prepared"
        if (prepared / "train.bin").exists() and (prepared / "validation.bin").exists():
            return
        if not state["auto_prepare"]:
            raise TrainingOrchestratorError(
                "Prepared corpus is missing. Enable automatic corpus preparation first."
            )
        log_path = Path(state["log_path"])
        with self._lock:
            current = self._load_state()
            current.update({"status": "preparing", "stage": "downloading_corpus"})
            self._save_state(current)
        self._run_setup_command([sys.executable, "-m", "scratch.download_gutenberg"], log_path)
        with self._lock:
            current = self._load_state()
            current.update({"status": "preparing", "stage": "preparing_corpus"})
            self._save_state(current)
        self._run_setup_command([sys.executable, "-m", "scratch.prepare_corpus"], log_path)

    def _pipeline(self, initial_state: dict[str, Any]) -> None:
        try:
            self._prepare_corpus(initial_state)
            current_generation = self._generation(str(initial_state["generation"]))
            previous_best: Path | None = None
            first_generation = True

            while True:
                result = self._run_generation(
                    current_generation,
                    initial_state,
                    previous_best=previous_best,
                    first_generation=first_generation,
                )
                if result["status"] != "completed":
                    return
                if not initial_state["auto_advance"]:
                    return

                best_validation = result.get("best_validation")
                target = float(initial_state["target_validation"])
                if best_validation is None or float(best_validation) > target:
                    with self._lock:
                        state = self._load_state()
                        state.update(
                            {
                                "status": "completed",
                                "stage": "growth_gate_not_met",
                                "growth_message": (
                                    f"Best validation {best_validation} did not reach target {target}."
                                ),
                            }
                        )
                        self._save_state(state)
                    return

                next_generation = self._generation_after(current_generation.identifier)
                if next_generation is None:
                    with self._lock:
                        state = self._load_state()
                        state.update({"status": "completed", "stage": "growth_plan_complete"})
                        self._save_state(state)
                    return
                if next_generation.parameters > int(initial_state["max_parameters"]):
                    with self._lock:
                        state = self._load_state()
                        state.update(
                            {
                                "status": "completed",
                                "stage": "parameter_limit_reached",
                                "growth_message": (
                                    f"Next generation requires {next_generation.parameters:,} parameters, "
                                    f"above the limit of {int(initial_state['max_parameters']):,}."
                                ),
                            }
                        )
                        self._save_state(state)
                    return

                previous_best = Path(str(result["best_checkpoint"]))
                current_generation = next_generation
                first_generation = False
                initial_state["target_validation"] = next_generation.target_validation
                initial_state["max_steps"] = next_generation.default_max_steps
                with self._lock:
                    state = self._load_state()
                    state.update(
                        {
                            "status": "advancing",
                            "stage": "initializing_next_generation",
                            "generation": next_generation.identifier,
                            "generation_name": next_generation.name,
                            "pid": None,
                        }
                    )
                    self._save_state(state)
        except Exception as exc:
            with self._lock:
                state = self._load_state()
                state.update({"status": "failed", "stage": "failed", "pid": None, "error": str(exc)})
                self._save_state(state)

    def _run_generation(
        self,
        generation: Generation,
        options: dict[str, Any],
        *,
        previous_best: Path | None,
        first_generation: bool,
    ) -> dict[str, Any]:
        events_path = Path(options["events_path"])
        control_path = Path(options["control_path"])
        log_path = Path(options["log_path"])
        generation.out_dir.mkdir(parents=True, exist_ok=True)
        latest_path = generation.out_dir / "latest.pt"
        best_path = generation.out_dir / "best.pt"
        self._write_control(control_path, "run")

        max_steps = int(options["max_steps"])
        command = [
            sys.executable,
            "-m",
            "scratch.train",
            "--config",
            str(generation.config_path),
            "--out-dir",
            str(generation.out_dir),
            "--max-steps",
            str(max_steps),
            "--events-file",
            str(events_path),
            "--control-file",
            str(control_path),
            "--best-checkpoint",
            str(best_path),
        ]
        if options["auto_activate_best"]:
            command.extend(["--activate-path", str(self.activation_path)])
        if first_generation and options["resume_existing"] and latest_path.exists():
            command.extend(["--resume", str(latest_path)])
        elif previous_best is not None and options["initialize_from_previous"]:
            command.extend(["--init-from", str(previous_best)])

        self._append_log(log_path, f"\n$ {' '.join(command)}")
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with self._lock:
                state = self._load_state()
                state.update(
                    {
                        "status": "running",
                        "stage": "training",
                        "generation": generation.identifier,
                        "generation_name": generation.name,
                        "pid": process.pid,
                        "max_steps": max_steps,
                        "config_path": str(generation.config_path),
                        "out_dir": str(generation.out_dir),
                        "latest_checkpoint": str(latest_path),
                        "best_checkpoint": str(best_path),
                    }
                )
                self._save_state(state)
            return_code = process.wait()

        events = self._read_events(events_path, limit=500)
        metrics = self._metrics_from_events(events)
        command_state = "run"
        try:
            command_state = str(json.loads(control_path.read_text(encoding="utf-8")).get("command"))
        except (OSError, json.JSONDecodeError):
            pass

        if return_code != 0:
            status = "failed"
            error = f"Training process exited with code {return_code}."
        elif command_state == "stop" or (events and events[-1].get("event") == "stopped"):
            status = "stopped"
            error = None
        else:
            status = "completed"
            error = None

        result = {
            "status": status,
            "return_code": return_code,
            "best_validation": metrics.get("best_validation"),
            "best_checkpoint": metrics.get("best_checkpoint") or str(best_path),
            "latest_checkpoint": metrics.get("latest_checkpoint") or str(latest_path),
        }
        with self._lock:
            state = self._load_state()
            state.update(
                {
                    **result,
                    "status": status,
                    "stage": status,
                    "pid": None,
                    "error": error,
                }
            )
            self._save_state(state)
        return result

    def _control(self, command: str, requested_status: str) -> dict[str, object]:
        with self._lock:
            state = self._load_state()
            if str(state.get("status")) not in ACTIVE_STATUSES:
                raise TrainingOrchestratorError("No active training job can receive this command.")
            control_path = Path(str(state["control_path"]))
            self._write_control(control_path, command)
            state["status"] = requested_status
            self._save_state(state)
            return self.status()

    def pause(self) -> dict[str, object]:
        return self._control("pause", "pause_requested")

    def resume(self) -> dict[str, object]:
        return self._control("run", "resume_requested")

    def stop(self) -> dict[str, object]:
        return self._control("stop", "stopping")

    def recover(self) -> None:
        """Resume an interrupted opted-in run after the web server itself restarts."""
        with self._lock:
            state = self._load_state()
            if str(state.get("status")) not in ACTIVE_STATUSES:
                return
            if self._pid_is_alive(int(state["pid"]) if state.get("pid") else None):
                return
            if not state.get("resume_after_restart"):
                state.update({"status": "interrupted", "pid": None})
                self._save_state(state)
                return
            generation = str(state.get("generation") or "level1")
            max_steps = int(state.get("max_steps") or self._generation(generation).default_max_steps)

        self.start(
            generation=generation,
            max_steps=max_steps,
            auto_prepare=bool(state.get("auto_prepare", True)),
            resume_existing=True,
            auto_activate_best=bool(state.get("auto_activate_best", True)),
            auto_advance=bool(state.get("auto_advance", False)),
            initialize_from_previous=bool(state.get("initialize_from_previous", True)),
            resume_after_restart=True,
            target_validation=float(state.get("target_validation", 1.9)),
            max_parameters=int(state.get("max_parameters", 40_000_000)),
        )
