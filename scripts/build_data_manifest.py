#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from packages.data.pipeline import DatasetManifest, SourceRecord, deduplicate_records, stable_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSONL SourceRecord records")
    parser.add_argument("--config", type=Path, default=Path("configs/data/phase2.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_manifest.json"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    records: list[SourceRecord] = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        record = SourceRecord(**json.loads(line))
        records.append(record)

    deduplicated = deduplicate_records(records)
    train, validation, test = stable_split(
        deduplicated,
        config["seed"],
        (config["train_ratio"], config["validation_ratio"], config["test_ratio"]),
    )

    manifest = DatasetManifest(
        name="phase2-pilot",
        seed=config["seed"],
        source_count=len(records),
        train_count=len(train),
        validation_count=len(validation),
        test_count=len(test),
        deduplicated_count=len(deduplicated),
        records=[{
            "fingerprint": record.fingerprint,
            "repository": record.repository,
            "path": record.path,
            "license": record.license,
            "license_reference": record.license_reference,
            "collection_date": record.collection_date,
            "language": record.language,
        } for record in deduplicated],
    )
    manifest.save(args.output)
    print(json.dumps({
        "source_count": manifest.source_count,
        "deduplicated_count": manifest.deduplicated_count,
        "train_count": manifest.train_count,
        "validation_count": manifest.validation_count,
        "test_count": manifest.test_count,
        "manifest": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
