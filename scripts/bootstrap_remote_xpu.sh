#!/usr/bin/env bash
set -euo pipefail

if [[ $(id -u) -ne 0 ]]; then
  echo "Run this script as root on the remote server." >&2
  exit 1
fi

apt-get update
apt-get install -y --no-install-recommends \
  python3-pip python3-venv libze1 libze-dev clinfo rsync curl ca-certificates \
  intel-opencl-icd intel-level-zero-gpu-dev

curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh
uv python install 3.12
uv venv --python 3.12 /opt/b70-code-platform/venv
uv pip install --python /opt/b70-code-platform/venv/bin/python \
  --index-url https://download.pytorch.org/whl/xpu \
  torch torchvision torchaudio

cat > /opt/b70-code-platform/activate.sh <<'EOF'
#!/usr/bin/env bash
source /opt/b70-code-platform/venv/bin/activate
EOF
chmod +x /opt/b70-code-platform/activate.sh

printf 'Remote XPU environment installed at /opt/b70-code-platform/venv\n'
