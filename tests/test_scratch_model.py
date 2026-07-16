from __future__ import annotations

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
