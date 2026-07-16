from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path


BOOKS = {
    11: "alice_in_wonderland",
    16: "peter_pan",
    55: "the_wonderful_wizard_of_oz",
    120: "treasure_island",
    1342: "pride_and_prejudice",
    1661: "sherlock_holmes",
    2701: "moby_dick",
}
START_RE = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
END_RE = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)


def fetch_book(book_id: int) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SecondBrainZero/0.1 educational language-model project"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    start = START_RE.search(text)
    end = END_RE.search(text)
    if start and end and start.end() < end.start():
        text = text[start.end() : end.start()]
    return text.strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small public-domain English corpus.")
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/data/raw"))
    parser.add_argument("--ids", type=int, nargs="*", default=list(BOOKS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for book_id in args.ids:
        name = BOOKS.get(book_id, f"gutenberg_{book_id}")
        output = args.output_dir / f"{name}.txt"
        print(f"Downloading {book_id} -> {output}")
        output.write_text(fetch_book(book_id), encoding="utf-8")


if __name__ == "__main__":
    main()
