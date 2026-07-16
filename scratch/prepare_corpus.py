from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip() + "\n"


def prepare(raw_dir: Path, output_dir: Path, validation_fraction: float) -> dict[str, int]:
    if not 0.001 <= validation_fraction <= 0.2:
        raise ValueError("validation_fraction must be between 0.001 and 0.2.")
    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}.")

    documents: list[bytes] = []
    for path in files:
        text = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
        if text.strip():
            documents.append(text.encode("utf-8"))
    corpus = b"\n\n<|document|>\n\n".join(documents)
    if len(corpus) < 100_000:
        raise ValueError(
            "The corpus is under 100 KB. Add more English text before serious training."
        )

    split_index = int(len(corpus) * (1.0 - validation_fraction))
    train = corpus[:split_index]
    validation = corpus[split_index:]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.bin").write_bytes(train)
    (output_dir / "validation.bin").write_bytes(validation)
    metadata = {
        "documents": len(documents),
        "total_bytes": len(corpus),
        "train_bytes": len(train),
        "validation_bytes": len(validation),
        "vocab_size": 256,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build byte-level English train/validation data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("scratch/data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/data/prepared"))
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    args = parser.parse_args()
    metadata = prepare(args.raw_dir, args.output_dir, args.validation_fraction)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
