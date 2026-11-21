#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.model import CodeTransformer, ModelConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/model/target.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_parameter_report.json"))
    args = parser.parse_args()

    config = ModelConfig.load(args.config)
    model = CodeTransformer(config)
    report = {
        "config": config.__dict__ if hasattr(config, "__dict__") else vars(config),
        "trainable_parameters": model.num_parameters(True),
        "total_parameters": model.num_parameters(False),
        "parameter_breakdown": model.parameter_breakdown(),
    }
    report["config"] = json.loads(args.config.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
