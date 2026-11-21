#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from packages.tokenizer import ByteBPETokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Text file containing representative code")
    parser.add_argument("--output", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=16384)
    args = parser.parse_args()

    corpus = args.input.read_text().split("\n\n")
    tokenizer = ByteBPETokenizer()
    tokenizer.train(corpus, args.vocab_size)
    tokenizer.save(args.output)
    print(f"Saved tokenizer with {tokenizer.vocab_size} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
