from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9_'-]+")


@dataclass(frozen=True)
class MemoryItem:
    kind: str
    text: str
    score: float
    metadata: dict[str, object]


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2}


class MemoryStore:
    """SQLite memory with transparent, deterministic lexical retrieval."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    prompt_version INTEGER NOT NULL,
                    feedback_score REAL,
                    feedback_note TEXT
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL DEFAULT 'user'
                );
                """
            )

    def remember_interaction(self, user_input: str, answer: str, prompt_version: int) -> int:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO interactions(created_at, user_input, answer, prompt_version)
                VALUES (?, ?, ?, ?)
                """,
                (timestamp, user_input, answer, prompt_version),
            )
            return int(cursor.lastrowid)

    def add_fact(self, content: str, importance: float = 1.0, source: str = "user") -> int:
        if not content.strip():
            raise ValueError("A fact cannot be empty.")
        if importance <= 0:
            raise ValueError("importance must be greater than zero.")
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO facts(created_at, content, importance, source)
                VALUES (?, ?, ?, ?)
                """,
                (timestamp, content.strip(), importance, source),
            )
            return int(cursor.lastrowid)

    def record_feedback(self, interaction_id: int, score: float, note: str = "") -> None:
        if not 0 <= score <= 1:
            raise ValueError("score must be between 0 and 1.")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE interactions
                SET feedback_score = ?, feedback_note = ?
                WHERE id = ?
                """,
                (score, note.strip(), interaction_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown interaction id: {interaction_id}")

    def recent_interactions(self, limit: int = 30) -> list[dict[str, object]]:
        limit = max(1, min(int(limit), 100))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, user_input, answer, prompt_version,
                       feedback_score, feedback_note
                FROM interactions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "created_at": str(row["created_at"]),
                "user_input": str(row["user_input"]),
                "answer": str(row["answer"]),
                "prompt_version": int(row["prompt_version"]),
                "feedback_score": (
                    float(row["feedback_score"]) if row["feedback_score"] is not None else None
                ),
                "feedback_note": str(row["feedback_note"] or ""),
            }
            for row in rows
        ]

    def stats(self) -> dict[str, object]:
        with self._connect() as connection:
            interactions = connection.execute(
                "SELECT COUNT(*) AS count FROM interactions"
            ).fetchone()
            facts = connection.execute("SELECT COUNT(*) AS count FROM facts").fetchone()
            feedback = connection.execute(
                """
                SELECT COUNT(feedback_score) AS rated, AVG(feedback_score) AS average
                FROM interactions
                """
            ).fetchone()
        return {
            "interactions": int(interactions["count"]),
            "facts": int(facts["count"]),
            "rated_interactions": int(feedback["rated"]),
            "average_feedback": (
                round(float(feedback["average"]), 4)
                if feedback["average"] is not None
                else None
            ),
        }

    def retrieve(self, query: str, limit: int = 6) -> list[MemoryItem]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        candidates: list[MemoryItem] = []
        with self._connect() as connection:
            facts = connection.execute(
                "SELECT id, content, importance, source, created_at FROM facts ORDER BY id DESC LIMIT 250"
            ).fetchall()
            interactions = connection.execute(
                """
                SELECT id, user_input, answer, feedback_score, feedback_note, created_at
                FROM interactions ORDER BY id DESC LIMIT 250
                """
            ).fetchall()

        for row in facts:
            content = str(row["content"])
            overlap = len(query_tokens & _tokens(content))
            if overlap:
                score = overlap / max(len(query_tokens), 1) * float(row["importance"])
                candidates.append(
                    MemoryItem(
                        "fact",
                        content,
                        score,
                        {
                            "id": row["id"],
                            "source": row["source"],
                            "created_at": row["created_at"],
                        },
                    )
                )
        for row in interactions:
            combined = f"Question: {row['user_input']}\nRéponse: {row['answer']}"
            overlap = len(query_tokens & _tokens(combined))
            if overlap:
                feedback = row["feedback_score"]
                multiplier = 0.5 + float(feedback) if feedback is not None else 1.0
                candidates.append(
                    MemoryItem(
                        "interaction",
                        combined,
                        overlap / max(len(query_tokens), 1) * multiplier,
                        {
                            "id": row["id"],
                            "feedback_score": feedback,
                            "feedback_note": row["feedback_note"],
                            "created_at": row["created_at"],
                        },
                    )
                )

        return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]

    @staticmethod
    def format_context(items: Iterable[MemoryItem]) -> str:
        material = []
        for index, item in enumerate(items, start=1):
            metadata = json.dumps(item.metadata, ensure_ascii=False, sort_keys=True)
            material.append(
                f"[Mémoire {index} | {item.kind} | score={item.score:.3f}]\n"
                f"{item.text}\n{metadata}"
            )
        return "\n\n".join(material)
