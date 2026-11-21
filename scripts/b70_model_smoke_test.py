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
    parser.add_argument("--output", type=Path, default=Path("artifacts/b70_model_smoke_test.json"))
    args = parser.parse_args()

    report: dict = {"xpu_available": hasattr(torch, "xpu") and torch.xpu.is_available()}
    if not report["xpu_available"]:
        report.update({"passed": False, "blocker": "torch.xpu.is_available() is false"})
    else:
        try:
            config = ModelConfig.load(args.config)
            model = CodeTransformer(config).to(device="xpu", dtype=torch.bfloat16)
            ids = torch.randint(0, config.vocab_size, (2, 128), device="xpu")
            logits = model(ids)
            forward_finite = torch.isfinite(logits).all().item()

            loss = logits.float().square().mean()
            loss.backward()
            optimizer = torch.optim.AdamW(model.parameters())
            optimizer.step()

            report.update({
                "passed": forward_finite and torch.isfinite(loss).item(),
                "trainable_parameters": model.num_parameters(True),
                "logits_shape": list(logits.shape),
                "forward_finite": forward_finite,
                "loss": loss.item(),
            })

            path = args.output.with_suffix(".pt")
            torch.save(model.state_dict(), path)
            restored = CodeTransformer(config).to(device="xpu", dtype=torch.bfloat16)
            restored.load_state_dict(torch.load(path, map_location="xpu", weights_only=True))
            with torch.no_grad():
                restored_logits = restored(ids)
            report["checkpoint_restore_finite"] = torch.isfinite(restored_logits).all().item()
            report["checkpoint_path"] = str(path)
            report["passed"] = report["passed"] and report["checkpoint_restore_finite"]
        except Exception as exc:
            report.update({"passed": False, "blocker": f"{type(exc).__name__}: {exc}"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
