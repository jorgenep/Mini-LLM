from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

SPECIAL_TOKENS = [
    "<|endoftext|>", "<|pad|>", "<|fim_prefix|>", "<|fim_suffix|>",
    "<|fim_middle|>", "<|file_sep|>", "<|repo_name|>", "<|unk|>",
]


class ByteBPETokenizer:
    """Small reproducible byte-level BPE implementation for development and tests."""

    def __init__(self, vocab: dict[str, int] | None = None, merges: list[list[str]] | None = None):
        self.special_tokens = SPECIAL_TOKENS
        self.vocab = dict(vocab or {})
        self.merges = [tuple(pair) for pair in (merges or [])]
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        if not self.vocab:
            offset = len(SPECIAL_TOKENS)
            self.vocab = {special: index for index, special in enumerate(SPECIAL_TOKENS)}
            for byte_value in range(256):
                self.vocab[f"<0x{byte_value:02X}>"] = offset + byte_value

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def special_token_id(self, token: str) -> int:
        if token not in self.vocab:
            raise KeyError(token)
        return self.vocab[token]

    def train(self, texts: Iterable[str], vocab_size: int = 16384) -> None:
        corpus = [list(f"<0x{byte:02X}>" for byte in text.encode("utf-8")) for text in texts]
        next_id = len(self.vocab)
        while len(self.vocab) < vocab_size:
            pairs = Counter((a, b) for symbols in corpus for a, b in zip(symbols, symbols[1:]))
            if not pairs:
                break
            pair = max(pairs.items(), key=lambda item: (item[1], tuple(item[0])))[0]
            if pair in self.merge_ranks:
                break
            merged = pair[0] + pair[1]
            if merged in self.vocab:
                continue
            rank = len(self.merges)
            self.merges.append(pair)
            self.merge_ranks[pair] = rank
            self.vocab[merged] = next_id
            next_id += 1
            corpus = [self._merge_symbols(symbols) for symbols in corpus]

    def _merge_symbols(self, symbols: list[str]) -> list[str]:
        merged: list[str] = []
        i = 0
        while i < len(symbols):
            pair = (symbols[i], symbols[i + 1]) if i + 1 < len(symbols) else None
            if pair is not None and pair in self.merge_ranks:
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        pieces: list[str] = []
        i = 0
        while i < len(text):
            matched = False
            for special in self.special_tokens:
                if text.startswith(special, i):
                    pieces.append(special)
                    i += len(special)
                    matched = True
                    break
            if matched:
                continue
            byte_value = text[i].encode("utf-8")
            pieces.extend(f"<0x{value:02X}>" for value in byte_value)
            i += 1

        pieces = self._apply_bpe(pieces)
        return [self.vocab.get(piece, self.special_token_id("<|unk|>")) for piece in pieces]

    def _apply_bpe(self, pieces: list[str]) -> list[str]:
        while True:
            candidates = [(self.merge_ranks[(a, b)], i, a, b)
                          for i, (a, b) in enumerate(zip(pieces, pieces[1:]))
                          if (a, b) in self.merge_ranks]
            if not candidates:
                return pieces
            _, i, a, b = min(candidates)
            pieces = pieces[:i] + [a + b] + pieces[i + 2:]

    def decode(self, ids: Iterable[int]) -> str:
        id_to_piece = {value: key for key, value in self.vocab.items()}
        output = bytearray()
        text_parts: list[str] = []
        for token_id in ids:
            piece = id_to_piece.get(token_id, "<|unk|>")
            if piece in self.special_tokens:
                text_parts.append(piece)
                continue
            byte_symbols = re.findall(r"<0x([0-9A-F]{2})>", piece)
            if byte_symbols:
                output.extend(int(symbol, 16) for symbol in byte_symbols)
            else:
                text_parts.append(piece)
        text = output.decode("utf-8", errors="replace")
        if text_parts:
            return text + "".join(text_parts)
        return text

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text(json.dumps({
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
        }, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "ByteBPETokenizer":
        data = json.loads((Path(directory) / "tokenizer.json").read_text())
        tokenizer = cls(data["vocab"], data["merges"])
        tokenizer.special_tokens = data["special_tokens"]
        return tokenizer
