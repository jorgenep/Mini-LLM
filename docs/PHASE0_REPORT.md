# Phase 0 — Preflight Report

**Status:** BLOCKED — B70 XPU accelerator gates did not pass.

## Machine inventory

- Linux: CachyOS Linux (rolling), kernel 7.2.6-1-cachyos
- CPU: Intel 12th Gen i7-12650H, 16 logical threads
- RAM: 31 GiB
- Storage: 730.2 GiB free
- PCI accelerators: Intel Alder Lake-P UHD Graphics and NVIDIA GeForce RTX 4060 Laptop GPU
- Intel Level Zero loader package: installed
- oneAPI common package: installed
- PyTorch: 2.15.0 CUDA build
- PyTorch CUDA: available
- PyTorch XPU: unavailable
- B70: not detected
- NVML: driver/library mismatch reported during probe

## Mandatory gate results

- Operating system detects B70: **FAIL — no B70 PCI device**
- Intel runtime and Level Zero functional: **UNKNOWN/FAIL — Level Zero loader exists, no matching Arc hardware**
- `torch.xpu.is_available()`: **FAIL**
- BF16 XPU GEMM: **NOT RUN**
- XPU forward: **NOT RUN**
- XPU backward: **NOT RUN**
- XPU optimizer: **NOT RUN**
- XPU checkpoint save/restore: **NOT RUN**
- Short stable XPU training: **NOT RUN**
- GPU utilization/memory metrics: **NOT VERIFIED**
- Training throughput: **NOT MEASURED**

## Decision

No dataset collection, model training, or inference model work will start. A supported fallback environment must provide a verified B70/XPU PyTorch build. CUDA fallback may only be used for CPU/framework development and must never be represented as B70 training.

## Exact next step

Attach the B70 to the Linux host and install the matching Intel compute runtime, Level Zero stack, and a PyTorch XPU build, then rerun both Phase 0 scripts and the tests.

## Remote B70 server — observed pass

- SSH server: `root@e-ai-server`
- OS: Ubuntu 26.04.1 LTS, kernel `7.0.0-38-generic`
- CPU: AMD Ryzen 5 3600, 12 logical threads
- PCI GPU: Intel Battlemage G31 / `8086:e223`
- Device reported by clinfo: `Intel(R) Arc(TM) Pro B70 Graphics`
- PyTorch: `2.15.0+xpu`
- Python: `3.12.13`
- PyTorch XPU: **available**
- BF16 GEMM: **passed and finite**
- Five-step BF16 forward/backward/optimizer test: **passed**
- Loss values remained finite: `0.295230, 0.344595, 0.352045, 0.338661, 0.359977`
- Checkpoint save/restore: **passed**
- Local unit tests: **2/2 passed**
- Remote unit tests: **2/2 passed**

### Remote environment

The installed isolated environment is:

```bash
source /opt/b70-code-platform/activate.sh
```

Or invoke it directly:

```bash
/opt/b70-code-platform/venv/bin/python
```

### Phase 0 result

**PASS for remote B70 training eligibility.** Full dataset collection and training may proceed according to later phase exit criteria. GPU utilization and measured training throughput still require a representative training run.
