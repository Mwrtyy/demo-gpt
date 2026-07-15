from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .llm import OpenAITextClient, TextClient
from .memory import MemoryStore
from .prompt_store import PromptVersion, load_prompt


@dataclass(frozen=True)
class Answer:
    text: str
    interaction_id: int | None
    prompt_version: int
    memories_used: int


class SecondBrain:
    def __init__(
        self,
        settings: Settings,
        client: TextClient | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or OpenAITextClient()
        self.memory = memory or MemoryStore(settings.database_path)

    def answer(
        self,
        user_input: str,
        *,
        prompt: PromptVersion | None = None,
        remember: bool = True,
    ) -> Answer:
        if not user_input.strip():
            raise ValueError("The user input cannot be empty.")

        selected_prompt = prompt or load_prompt(self.settings.active_prompt_path)
        memories = self.memory.retrieve(user_input)
        memory_context = self.memory.format_context(memories)

        instructions = selected_prompt.render()
        if memory_context:
            instructions += (
                "\n\nMémoire récupérée. Utilise-la seulement si elle est pertinente. "
                "Ne présente jamais une mémoire comme certaine si elle est ambiguë.\n\n"
                f"{memory_context}"
            )

        text = self.client.generate(
            model=self.settings.model,
            instructions=instructions,
            user_input=user_input.strip(),
        )

        interaction_id = None
        if remember:
            interaction_id = self.memory.remember_interaction(
                user_input=user_input.strip(),
                answer=text,
                prompt_version=selected_prompt.version,
            )

        return Answer(
            text=text,
            interaction_id=interaction_id,
            prompt_version=selected_prompt.version,
            memories_used=len(memories),
        )
