from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scratch.model import ByteGPT, ModelConfig
from scratch.tokenizer import ByteTokenizer
from scratch.train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate English text with Second Brain Zero.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = ByteGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = ByteTokenizer()
    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens:
        raise ValueError("Prompt cannot be empty.")
    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = model.generate(
        tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
