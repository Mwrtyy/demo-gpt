from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scratch.data import BinaryCorpus
from scratch.model import ByteGPT, ModelConfig


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def learning_rate(step: int, *, peak: float, warmup: int, total: int, minimum: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if step >= total:
        return minimum
    ratio = (step - warmup) / max(total - warmup, 1)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return minimum + coefficient * (peak - minimum)


def load_config(path: Path) -> tuple[ModelConfig, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError("Config must contain a model object.")
    training = raw.get("training")
    if not isinstance(training, dict):
        raise ValueError("Config must contain a training object.")
    return ModelConfig.from_dict(raw["model"]), training


def emit_event(path: Path | None, event: str, **payload: object) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "timestamp": time.time(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_control_command(path: Path | None) -> str:
    if path is None or not path.exists():
        return "run"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "run"
    command = str(payload.get("command", "run")).strip().lower()
    return command if command in {"run", "pause", "stop"} else "run"


def wait_for_training_permission(control_path: Path | None, events_path: Path | None) -> bool:
    paused = False
    while True:
        command = read_control_command(control_path)
        if command == "stop":
            emit_event(events_path, "stop_requested")
            return False
        if command == "pause":
            if not paused:
                emit_event(events_path, "paused")
                paused = True
            time.sleep(0.75)
            continue
        if paused:
            emit_event(events_path, "resumed")
        return True


def estimate_loss(
    model: ByteGPT,
    corpus: BinaryCorpus,
    *,
    batch_size: int,
    block_size: int,
    eval_batches: int,
    device: torch.device,
    autocast_context: Any,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    for name, data in (("train", corpus.train), ("validation", corpus.validation)):
        losses = []
        for _ in range(eval_batches):
            x, y = corpus.batch(
                data, batch_size=batch_size, block_size=block_size, device=device
            )
            with autocast_context():
                _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss.detach().cpu()))
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def initialize_from_checkpoint(model: ByteGPT, source_path: Path) -> int:
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Initialization checkpoint is missing model weights.")
    target_state = model.state_dict()
    copied = 0
    for name, value in checkpoint["model"].items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value
            copied += 1
    model.load_state_dict(target_state, strict=True)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Second Brain Zero from random weights.")
    parser.add_argument(
        "--config", type=Path, default=Path("scratch/configs/level1_english_19m.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("scratch/data/prepared"))
    parser.add_argument("--out-dir", type=Path, default=Path("scratch/checkpoints/level1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--events-file", type=Path)
    parser.add_argument("--control-file", type=Path)
    parser.add_argument("--best-checkpoint", type=Path)
    parser.add_argument("--activate-path", type=Path)
    args = parser.parse_args()

    if args.resume and args.init_from:
        parser.error("--resume and --init-from cannot be used together.")

    model_config, training = load_config(args.config)
    max_steps = args.max_steps or int(training["max_steps"])
    if max_steps < 1:
        parser.error("--max-steps must be at least 1.")
    seed = int(training.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    corpus = BinaryCorpus.load(args.data_dir)
    model = ByteGPT(model_config).to(device)
    parameters = model.parameter_count()
    print(f"Device: {device}", flush=True)
    print(f"Parameters: {parameters:,}", flush=True)

    optimizer_kwargs = {
        "lr": float(training["learning_rate"]),
        "betas": tuple(training.get("betas", [0.9, 0.95])),
        "weight_decay": float(training.get("weight_decay", 0.1)),
    }
    if device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    start_step = 0
    best_validation = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        best_validation = float(checkpoint["best_validation"])
        emit_event(
            args.events_file,
            "checkpoint_resumed",
            checkpoint=str(args.resume),
            start_step=start_step,
            best_validation=best_validation,
        )
    elif args.init_from:
        copied = initialize_from_checkpoint(model, args.init_from)
        print(f"Initialized {copied} compatible tensors from {args.init_from}", flush=True)
        emit_event(
            args.events_file,
            "generation_initialized",
            checkpoint=str(args.init_from),
            copied_tensors=copied,
        )

    compile_model = bool(training.get("compile", False))
    if compile_model and hasattr(torch, "compile") and device.type != "mps":
        model = torch.compile(model)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    def autocast_context():
        if use_amp:
            return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    batch_size = int(training["batch_size"])
    gradient_accumulation = int(training.get("gradient_accumulation", 1))
    eval_interval = int(training.get("eval_interval", 250))
    eval_batches = int(training.get("eval_batches", 20))
    checkpoint_interval = int(training.get("checkpoint_interval", eval_interval))
    gradient_clip = float(training.get("gradient_clip", 1.0))
    warmup_steps = int(training.get("warmup_steps", 200))
    peak_lr = float(training["learning_rate"])
    minimum_lr = float(training.get("minimum_learning_rate", peak_lr * 0.1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.out_dir / "latest.pt"
    best_path = args.best_checkpoint or args.out_dir / "best.pt"
    model.train()
    started = time.time()
    last_completed_step = start_step - 1
    emit_event(
        args.events_file,
        "started",
        device=str(device),
        parameters=parameters,
        start_step=start_step,
        max_steps=max_steps,
        config=str(args.config),
        out_dir=str(args.out_dir),
    )

    try:
        for step in range(start_step, max_steps):
            if not wait_for_training_permission(args.control_file, args.events_file):
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                stopped_step = max(last_completed_step, 0)
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": model_config.to_dict(),
                    "training_config": training,
                    "step": stopped_step,
                    "best_validation": best_validation,
                }
                atomic_torch_save(checkpoint, latest_path)
                emit_event(
                    args.events_file,
                    "stopped",
                    step=stopped_step,
                    checkpoint=str(latest_path),
                    best_validation=best_validation,
                )
                print(f"Stopped safely at step {stopped_step}", flush=True)
                return

            lr = learning_rate(
                step, peak=peak_lr, warmup=warmup_steps, total=max_steps, minimum=minimum_lr
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(gradient_accumulation):
                x, y = corpus.batch(
                    corpus.train,
                    batch_size=batch_size,
                    block_size=model_config.block_size,
                    device=device,
                )
                with autocast_context():
                    _, loss = model(x, y)
                    assert loss is not None
                    scaled_loss = loss / gradient_accumulation
                accumulated_loss += float(loss.detach().cpu()) / gradient_accumulation
                scaler.scale(scaled_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            last_completed_step = step

            if step % 10 == 0:
                elapsed = max(time.time() - started, 1e-6)
                completed_steps = step - start_step + 1
                processed = completed_steps * batch_size * gradient_accumulation
                steps_per_second = completed_steps / elapsed
                sequences_per_second = processed / elapsed
                eta_seconds = max(max_steps - step - 1, 0) / max(steps_per_second, 1e-9)
                print(
                    f"step={step:06d} loss={accumulated_loss:.4f} lr={lr:.2e} "
                    f"sequences/s={sequences_per_second:.2f}",
                    flush=True,
                )
                emit_event(
                    args.events_file,
                    "progress",
                    step=step,
                    max_steps=max_steps,
                    loss=accumulated_loss,
                    learning_rate=lr,
                    steps_per_second=steps_per_second,
                    sequences_per_second=sequences_per_second,
                    elapsed_seconds=elapsed,
                    eta_seconds=eta_seconds,
                )

            should_evaluate = step % eval_interval == 0 or step == max_steps - 1
            validation_loss: float | None = None
            improved = False
            if should_evaluate:
                losses = estimate_loss(
                    model,
                    corpus,
                    batch_size=batch_size,
                    block_size=model_config.block_size,
                    eval_batches=eval_batches,
                    device=device,
                    autocast_context=autocast_context,
                )
                validation_loss = losses["validation"]
                improved = validation_loss < best_validation
                if improved:
                    best_validation = validation_loss
                print(
                    f"evaluation step={step} train={losses['train']:.4f} "
                    f"validation={validation_loss:.4f}",
                    flush=True,
                )
                emit_event(
                    args.events_file,
                    "evaluation",
                    step=step,
                    train_loss=losses["train"],
                    validation_loss=validation_loss,
                    best_validation=best_validation,
                    improved=improved,
                )

            should_checkpoint = step % checkpoint_interval == 0 or step == max_steps - 1
            if should_checkpoint or improved:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": model_config.to_dict(),
                    "training_config": training,
                    "step": step,
                    "best_validation": best_validation,
                }
                if should_checkpoint:
                    atomic_torch_save(checkpoint, latest_path)
                    print(f"Saved {latest_path}", flush=True)
                if improved:
                    atomic_torch_save(checkpoint, best_path)
                    if args.activate_path:
                        atomic_copy(best_path, args.activate_path)
                emit_event(
                    args.events_file,
                    "checkpoint",
                    step=step,
                    latest=str(latest_path) if should_checkpoint else None,
                    best=str(best_path) if improved else None,
                    activated=str(args.activate_path) if improved and args.activate_path else None,
                    best_validation=best_validation,
                    validation_loss=validation_loss,
                )

        emit_event(
            args.events_file,
            "completed",
            step=max_steps - 1,
            max_steps=max_steps,
            best_validation=best_validation,
            latest=str(latest_path),
            best=str(best_path),
        )
    except Exception as exc:
        emit_event(args.events_file, "error", message=str(exc), step=last_completed_step)
        raise


if __name__ == "__main__":
    main()
