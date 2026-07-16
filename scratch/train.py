from __future__ import annotations

import argparse
import json
import math
import random
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
    args = parser.parse_args()

    model_config, training = load_config(args.config)
    max_steps = args.max_steps or int(training["max_steps"])
    seed = int(training.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    corpus = BinaryCorpus.load(args.data_dir)
    model = ByteGPT(model_config).to(device)
    print(f"Device: {device}")
    print(f"Parameters: {model.parameter_count():,}")

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
    model.train()
    started = time.time()
    for step in range(start_step, max_steps):
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

        if step % 10 == 0:
            elapsed = max(time.time() - started, 1e-6)
            processed = (step - start_step + 1) * batch_size * gradient_accumulation
            print(
                f"step={step:06d} loss={accumulated_loss:.4f} lr={lr:.2e} "
                f"sequences/s={processed / elapsed:.2f}"
            )

        should_evaluate = step % eval_interval == 0 or step == max_steps - 1
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
            print(
                f"evaluation step={step} train={losses['train']:.4f} "
                f"validation={losses['validation']:.4f}"
            )
            best_validation = min(best_validation, losses["validation"])

        should_checkpoint = step % checkpoint_interval == 0 or step == max_steps - 1
        if should_checkpoint:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                "training_config": training,
                "step": step,
                "best_validation": best_validation,
            }
            latest = args.out_dir / "latest.pt"
            torch.save(checkpoint, latest)
            print(f"Saved {latest}")


if __name__ == "__main__":
    main()
