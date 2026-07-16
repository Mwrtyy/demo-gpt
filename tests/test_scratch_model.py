from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scratch.model import ByteGPT, ModelConfig  # noqa: E402
from scratch.tokenizer import ByteTokenizer  # noqa: E402


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
