#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parser.parse_args() if hasattr(parser, "parser") else parser.parse_args()

    gates: list[dict] = []
    try:
        import torch
    except Exception as exc:
        gates.append(check("torch_import", False, f"{type(exc).__name__}: {exc}"))
        report = {
            "schema_version": 1,
            "passed": False,
            "gates": gates,
            "training_allowed": False,
            "blocker": "PyTorch could not be imported",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 2

    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    gates.append(check("torch_xpu_available", xpu_available, "torch.xpu.is_available()"))

    if xpu_available:
        try:
            a = torch.randn(1024, 1024, device="xpu", dtype=torch.bfloat16)
            b = torch.randn(1024, 1024, device="xpu", dtype=torch.bfloat16)
            out = a @ b
            bf16_ok = torch.isfinite(out).all().item() and out.abs().max().item() < 1e4
            gates.append(check("bf16_gemm", bf16_ok, f"finite={bool(torch.isfinite(out).all())}"))

            model = torch.nn.Linear(64, 64, dtype=torch.bfloat16, device="xpu")
            optimizer = torch.optim.AdamW(model.parameters())
            losses = []
            for _ in range(5):
                optimizer.zero_grad(set_to_none=True)
                x = torch.randn(8, 64, dtype=torch.bfloat16, device="xpu")
                loss = model(x).float().square().mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            finite_losses = all(torch.isfinite(torch.tensor(loss, dtype=torch.float32)).item() for loss in losses)
            stable_training = finite_losses and all(loss < 1e6 for loss in losses)
            gates.append(check(
                "forward_backward_optimizer_short_training",
                stable_training,
                f"losses={[round(loss, 6) for loss in losses]}",
            ))

            path = args.output.with_suffix(".pt")
            torch.save(model.state_dict(), path)
            restored = torch.nn.Linear(64, 64, dtype=torch.bfloat16, device="xpu")
            restored.load_state_dict(torch.load(path, map_location="xpu", weights_only=True))
            gates.append(check("checkpoint_save_restore", True, str(path)))
        except Exception as exc:
            gates.append(check("xpu_operation", False, f"{type(exc).__name__}: {exc}"))
    else:
        gates.append(check("bf16_gemm", False, "Not run because XPU is unavailable"))
        gates.append(check("forward_backward_optimizer", False, "Not run because XPU is unavailable"))
        gates.append(check("checkpoint_save_restore", False, "Not run because XPU is unavailable"))

    passed = all(g["passed"] for g in gates)
    report = {
        "schema_version": 1,
        "passed": passed,
        "gates": gates,
        "training_allowed": passed,
        "blocker": None if passed else "Mandatory Intel B70/XPU gates did not pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
