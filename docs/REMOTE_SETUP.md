# Remote GPU Server Setup

This project uses SSH/rsync so source code remains on the laptop while expensive computation runs on the remote server.

## Server

- Address alias: `e-ai-server`
- SSH target: `root@e-ai-server`
- Remote project directory: `~/Mini-LLM`
- Required hardware: Intel Arc Pro B70
- Required software: Ubuntu, matching Intel compute runtime, Level Zero, and PyTorch XPU build

## Laptop commands

Authenticate the Tailscale SSH route in the browser, then test connectivity:

```bash
ssh root@e-ai-server
```

Sync source and scripts:

```bash
./scripts/sync_to_server.sh
```

Run Phase 0 remotely:

```bash
./scripts/sync_to_server.sh
./scripts/remote_phase0.sh
```

Open an SSH tunnel for future TensorBoard use:

```bash
ssh -L 6006:127.0.0.1:6006 root@e-ai-server
```

## Installed server environment

The server has an isolated PyTorch XPU environment at:

```text
/opt/b70-code-platform/venv
```

Activate it over SSH:

```bash
ssh root@e-ai-server
source /opt/b70-code-platform/activate.sh
cd ~/Mini-LLM
python scripts/xpu_smoke_test.py --output artifacts/xpu_smoke_test.json
```

## Privacy

The rsync script excludes Git metadata, virtual environments, dependencies, caches, and local artifacts. It does not send private source to a cloud API. Add additional excludes for sensitive local files before production use.
