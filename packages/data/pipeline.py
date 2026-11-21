from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class SourceRecord:
    source_id: str
    repository: str
    path: str
    license: str
    license_reference: str
    collection_date: str
    language: str
    text: str
    filtering_decisions: list[str]
    transformation_history: list[str]

    @property
    def fingerprint(self) -> str:
        normalized = "\n".join(line.rstrip() for line in self.text.strip().splitlines())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class DatasetManifest:
    name: str
    seed: int
    source_count: int
    train_count: int
    validation_count: int
    test_count: int
    deduplicated_count: int
    records: list[dict]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def deduplicate_records(records: Iterable[SourceRecord]) -> list[SourceRecord]:
    seen: set[str] = set()
    result: list[SourceRecord] = []
    for record in records:
        fingerprint = record.fingerprint
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        result.append(record)
    return result


def classify_language(path: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit.lower()
    suffix = Path(path).suffix.lower()
    return {".py": "python", ".js": "javascript", ".jsx": "javascript",
            ".ts": "typescript", ".tsx": "typescript"}.get(suffix, "unsupported")


def filter_record(record: SourceRecord) -> tuple[bool, list[str]]:
    decisions: list[str] = []
    if not record.text.strip():
        decisions.append("empty")
    if len(record.text) < 40:
        decisions.append("extremely_short")
    lower_path = record.path.lower()
    for marker in ("package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock"):
        if marker in lower_path:
            decisions.append("lockfile")
    if any(name in lower_path for name in ("/vendor/", "/node_modules/", "/dist/", "/build/")):
        decisions.append("generated_or_vendored")
    if record.language not in {"python", "javascript", "typescript"}:
        decisions.append("unsupported_language")
    return not decisions, decisions


def stable_split(records: list[SourceRecord], seed: int, ratios: tuple[float, float, float]):
    ordered = sorted(records, key=lambda record: record.fingerprint)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    total = len(ordered)
    train_end = int(total * ratios[0])
    validation_end = train_end + int(total * ratios[1])
    return ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]


def prepare_record(record: SourceRecord, tokenizer, objective: str = "causal") -> dict:
    if objective == "causal":
        return {
            "objective": objective,
            "source_fingerprint": record.fingerprint,
            "language": record.language,
            "input_ids": tokenizer.encode(record.text),
        }
    if objective not in {"psm_fim", "spm_fim"}:
        raise ValueError(f"Unknown objective: {objective}")

    boundaries = [i for i, char in enumerate(record.text) if char in "\n;{}"]
    if len(record.text) < 20 or not boundaries:
        middle_start = max(0, len(record.text) // 3)
        middle_end = max(middle_start + 1, len(record.text) * 2 // 3)
    else:
        candidates = [point for point in boundaries if len(record.text) // 4 < point < len(record.text) * 3 // 4]
        point = candidates[len(candidates) // 2] if candidates else len(record.text) // 2
        middle_start = point
        middle_end = min(len(record.text), point + max(1, len(record.text) // 8))

    prefix = record.text[:middle_start]
    middle = record.text[middle_start:middle_end]
    suffix = record.text[middle_end:] if middle_end < len(record.text) else ""
    prefix_id = tokenizer.special_token_id("<|fim_prefix|>")
    suffix_id = tokenizer.special_token_id("<|fim_suffix|>")
    middle_id = tokenizer.special_token_id("<|fim_middle|>")

    if objective == "psm_fim":
        tokens = [prefix_id] + tokenizer.encode(prefix) + [suffix_id] + tokenizer.encode(suffix) + [middle_id] + tokenizer.encode(middle)
    else:
        tokens = [suffix_id] + tokenizer.encode(suffix) + [prefix_id] + tokenizer.encode(prefix) + [middle_id] + tokenizer.encode(middle)
    return {
        "objective": objective,
        "source_fingerprint": record.fingerprint,
        "language": record.language,
        "input_ids": tokens,
    }


def pack_examples(examples: Iterable[dict], sequence_length: int, pad_id: int) -> Iterator[list[int]]:
    buffer: list[int] = []
    for example in examples:
        ids = example["input_ids"]
        for token_id in ids:
            buffer.append(token_id)
            if len(buffer) == sequence_length:
                yield buffer
                buffer = []
    if buffer:
        yield buffer + [pad_id] * (sequence_length - len(buffer))
