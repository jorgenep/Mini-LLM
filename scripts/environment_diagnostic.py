#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def pci_gpus() -> list[str]:
    output = run(["lspci", "-nn"])
    return [line for line in output.splitlines() if "VGA" in line or "3D controller" in line]


def package_version(package: str) -> str:
    return run(["pacman", "-Q", package])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os_release = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip('"')
    except OSError:
        pass

    torch_info = {}
    try:
        import torch
        torch_info = {
            "torch_version": torch.__version__,
            "xpu_available": hasattr(torch, "xpu") and torch.xpu.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            torch_info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        torch_info = {"import_error": f"{type(exc).__name__}: {exc}"}

    report = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "linux_distribution": os_release,
        "cpu_model": run(["sh", "-c", "lscpu | grep 'Model name' | cut -d: -f2-"]).strip(),
        "cpu_logical_cores": os.cpu_count(),
        "pci_accelerators": pci_gpus(),
        "dri_devices": sorted(str(p) for p in Path("/dev/dri").glob("*")) if Path("/dev/dri").exists() else [],
        "level_zero_loader_package": package_version("level-zero-loader") or None,
        "intel_oneapi_common_package": package_version("intel-oneapi-common") or None,
        "clinfo_platforms": run(["clinfo", "-l"]),
        "pytorch": torch_info,
        "nvml_probe": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "tools": {name: shutil.which(name) for name in ("python3", "sycl-ls", "ze_peak", "clinfo", "xpu-smi", "nvidia-smi")},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nReport written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
