#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
printf '0\n' > /logs/verifier/reward.txt
if python3 /tests/test_outputs.py; then
  printf '1\n' > /logs/verifier/reward.txt
fi
