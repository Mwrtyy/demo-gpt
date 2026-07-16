from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class BinaryCorpus:
    train: np.memmap
    validation: np.memmap

    @classmethod
    def load(cls, directory: Path) -> "BinaryCorpus":
        train_path = directory / "train.bin"
        validation_path = directory / "validation.bin"
        if not train_path.exists() or not validation_path.exists():
            raise FileNotFoundError(
                "Prepared corpus not found. Run `python -m scratch.prepare_corpus` first."
            )
        return cls(
            train=np.memmap(train_path, dtype=np.uint8, mode="r"),
            validation=np.memmap(validation_path, dtype=np.uint8, mode="r"),
        )

    @staticmethod
    def batch(
        data: np.memmap,
        *,
        batch_size: int,
        block_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(data) <= block_size + 1:
            raise ValueError("Corpus split is too small for the configured block_size.")
        starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
        x = np.stack([np.asarray(data[i : i + block_size], dtype=np.int64) for i in starts])
        y = np.stack(
            [np.asarray(data[i + 1 : i + 1 + block_size], dtype=np.int64) for i in starts]
        )
        return (
            torch.from_numpy(x).to(device=device, non_blocking=True),
            torch.from_numpy(y).to(device=device, non_blocking=True),
        )
