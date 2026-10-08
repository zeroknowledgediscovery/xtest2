#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/output
cp /solution/solve.py /workspace/output/solution.py
exec python3 /workspace/output/solution.py
