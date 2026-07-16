from __future__ import annotations


class ByteTokenizer:
    """A dependency-free UTF-8 byte tokenizer with a fixed 256-token vocabulary."""

    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(tokens: list[int]) -> str:
        if any(token < 0 or token > 255 for token in tokens):
            raise ValueError("Byte tokens must be integers between 0 and 255.")
        return bytes(tokens).decode("utf-8", errors="replace")
