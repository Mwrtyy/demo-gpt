from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scratch.model import ByteGPT, ModelConfig  # noqa: E402
from scratch.tokenizer import ByteTokenizer  # noqa: E402
from second_brain.zero_runtime import ZeroRuntime  # noqa: E402


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "The tiny model says hello."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_model_forward_backward_and_generate() -> None:
    config = ModelConfig(block_size=16, n_layer=2, n_head=4, n_embd=64, dropout=0.0)
    model = ByteGPT(config)
    tokens = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    logits, loss = model(tokens, targets)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    generated = model.generate(tokens[:, :4], max_new_tokens=3, temperature=1.0, top_k=20)
    assert generated.shape == (2, 7)


def test_zero_runtime_loads_checkpoint_and_generates(tmp_path: Path) -> None:
    config = ModelConfig(block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
    model = ByteGPT(config)
    checkpoint_path = tmp_path / "latest.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "step": 7,
            "best_validation": 4.2,
        },
        checkpoint_path,
    )

    runtime = ZeroRuntime(checkpoint_path=checkpoint_path, device="cpu")
    status = runtime.status()
    result = runtime.generate(
        "Hello",
        max_new_tokens=3,
        temperature=1.0,
        top_k=20,
        seed=10,
    )

    assert status["ready"] is True
    assert status["step"] == 7
    assert status["parameters"] == model.parameter_count()
    assert result["new_tokens"] == 3
    assert result["device"] == "cpu"


def test_controlled_training_writes_latest_best_activation_and_events(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "checkpoints"
    data_dir.mkdir()
    corpus = (b"Once upon a time, the small model learned English.\n" * 20)
    (data_dir / "train.bin").write_bytes(corpus)
    (data_dir / "validation.bin").write_bytes(corpus)

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "vocab_size": 256,
                    "block_size": 8,
                    "n_layer": 1,
                    "n_head": 2,
                    "n_embd": 16,
                    "dropout": 0.0,
                    "bias": False,
                },
                "training": {
                    "seed": 7,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "max_steps": 2,
                    "learning_rate": 0.001,
                    "minimum_learning_rate": 0.0001,
                    "warmup_steps": 1,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.95],
                    "gradient_clip": 1.0,
                    "eval_interval": 1,
                    "eval_batches": 1,
                    "checkpoint_interval": 1,
                    "compile": False,
                },
            }
        ),
        encoding="utf-8",
    )
    events_path = tmp_path / "events.jsonl"
    control_path = tmp_path / "control.json"
    activation_path = tmp_path / "active" / "latest.pt"
    control_path.write_text('{"command":"run"}', encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scratch.train",
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--max-steps",
            "2",
            "--events-file",
            str(events_path),
            "--control-file",
            str(control_path),
            "--best-checkpoint",
            str(out_dir / "best.pt"),
            "--activate-path",
            str(activation_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (out_dir / "latest.pt").exists()
    assert (out_dir / "best.pt").exists()
    assert activation_path.exists()
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    event_types = {event["event"] for event in events}
    assert {"started", "progress", "evaluation", "checkpoint", "completed"} <= event_types
