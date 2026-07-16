# Second Brain Zero — English Level 1

This directory contains a real decoder-only Transformer trained **from random weights**.
It does not call OpenAI, Anthropic, Google, Hugging Face inference, or any other model API.
PyTorch is used only as the numerical training framework.

## Target model

The default configuration is an approximately 19M-parameter English byte-level language model:

- vocabulary: all 256 possible UTF-8 bytes;
- context: 256 bytes;
- layers: 6;
- attention heads: 8;
- embedding width: 512;
- objective: predict the next byte.

This first model is a text-completion model, not yet an instruction-following assistant. Its first realistic objective is coherent simple English after training on a clean, narrow corpus.

## 1. Install

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r scratch/requirements.txt
```

## 2. Create an English corpus

Option A downloads a small public-domain corpus from Project Gutenberg:

```bash
python -m scratch.download_gutenberg
python -m scratch.prepare_corpus
```

Option B uses your own licensed UTF-8 `.txt` files:

```text
scratch/data/raw/book_1.txt
scratch/data/raw/book_2.txt
...
```

Then run:

```bash
python -m scratch.prepare_corpus
```

The prepared binary corpus is intentionally ignored by Git.

## 3. Verify the pipeline on CPU

```bash
python -m scratch.train \
  --config scratch/configs/smoke_cpu.json \
  --out-dir scratch/checkpoints/smoke
```

## 4. Train the Level 1 model

```bash
python -m scratch.train \
  --config scratch/configs/level1_english_19m.json \
  --out-dir scratch/checkpoints/level1
```

Resume later:

```bash
python -m scratch.train \
  --config scratch/configs/level1_english_19m.json \
  --out-dir scratch/checkpoints/level1 \
  --resume scratch/checkpoints/level1/latest.pt
```

## 5. Generate text

```bash
python -m scratch.generate \
  --checkpoint scratch/checkpoints/level1/latest.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 400
```

## Hardware reality

The model can run on CPU, but serious training is slow. CUDA and ROCm builds use the same `cuda` device API inside PyTorch. On a Windows PC with an AMD GPU, the official PyTorch Windows path may fall back to CPU; Linux or WSL with a supported ROCm setup is the route to investigate before expecting GPU training.

## What is genuinely ours

- architecture code;
- random initial weights;
- byte tokenizer;
- data preparation;
- optimizer loop;
- checkpoints;
- generated output.

No external model produces the responses. The quality depends entirely on our corpus, compute budget, model configuration, and training discipline.
