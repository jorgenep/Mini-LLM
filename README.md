# B70 Code Completion Platform

Local-first coding completion platform. Development starts with Phase 0 environment preflight.

## Phase 0

Run:

```bash
python3 scripts/environment_diagnostic.py --output artifacts/environment_report.json
python3 scripts/xpu_smoke_test.py --output artifacts/xpu_smoke_test.json
python3 -m unittest discover -s tests -v
```

Phase 0 does not begin training. The B70 XPU gates must pass before a long training run or any claim of Intel accelerator training.
