from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TextClient(Protocol):
    def generate(self, *, model: str, instructions: str, user_input: str) -> str:
        ...


@dataclass
class OpenAITextClient:
    """Small adapter around the OpenAI Responses API."""

    def generate(self, *, model: str, instructions: str, user_input: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is missing. Run: pip install -e ."
            ) from exc

        client = OpenAI()
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_input,
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise RuntimeError("The model returned no text output.")
        return str(output_text).strip()
