#!/usr/bin/env bash
# Build the offline wheelhouse payload for the mucli TB agent.
# Output: bench/artifacts/mucli-wheelhouse.tar.gz (~48MB, 71 wheels)
# Usage: bash bench/build_wheelhouse.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Bench-essential deps (excludes kokoro/faster-whisper/soundfile/playwright — heavy, unused in bench)
grep -viE "kokoro|faster-whisper|soundfile|playwright" requirements.txt | grep -vE "^\s*#|^\s*$" > "$TMP/requirements-bench.txt"

# Download manylinux wheels for the TB container's python (3.12 on ubuntu-24-04 base)
$PY -m pip download -r "$TMP/requirements-bench.txt" \
  --platform manylinux2014_x86_64 --python-version 3.12 \
  --only-binary=:all: -d "$TMP/wheelhouse" -q

# mucli itself (pure-python wheel)
$PY -m pip wheel . --no-deps -w "$TMP/wheelhouse-mucli" -q
cp "$TMP"/wheelhouse-mucli/mucli-*.whl "$TMP/wheelhouse/"

mkdir -p bench/artifacts
tar -czf bench/artifacts/mucli-wheelhouse.tar.gz -C "$TMP" wheelhouse
ls -la bench/artifacts/mucli-wheelhouse.tar.gz
echo "wheelhouse built: $(ls "$TMP/wheelhouse" | wc -l) wheels"
