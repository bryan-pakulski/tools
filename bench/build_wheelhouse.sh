#!/usr/bin/env bash
# Build the offline wheelhouse payload for the mucli TB agent.
# Output: bench/artifacts/mucli-wheelhouse.tar.gz
# Usage: bash bench/build_wheelhouse.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
PYTHON_VERSIONS="${PYTHON_VERSIONS:-3.10 3.11 3.12 3.13 3.14}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
WHEEL_DIR="$TMP/wheelhouse"
REQ_FILE="$WHEEL_DIR/requirements-bench.txt"
mkdir -p "$WHEEL_DIR"

# Bench-essential dependencies. Voice, browser, and developer-only packages do
# not participate in headless Terminal-Bench runs and make the payload much
# larger and more fragile.
awk '
    /^[[:space:]]*($|#)/ { next }
    tolower($0) ~ /^(kokoro|faster-whisper|soundfile|playwright|pytest|black)/ { next }
    { print }
' requirements.txt > "$REQ_FILE"

# TB core mixes Ubuntu Python 3.12, 3.13, and 3.14 images, while other tasks
# can use any MuCLI-supported Python. Keep compatible binary wheels for every
# supported minor in one directory; pip selects the matching tag at install
# time while universal/abi3 wheels are naturally shared.
for version in $PYTHON_VERSIONS; do
  platform_args=(--platform manylinux2014_x86_64)
  # tiktoken's CPython 3.14+ wheels target glibc 2.28. TB's Python 3.14
  # image is new enough for that tag. Keep manylinux2014 as a second accepted
  # tag because universal-ABI packages such as primp still publish there.
  if [ "$version" = "3.14" ]; then
    platform_args=(
      --platform manylinux_2_28_x86_64
      --platform manylinux2014_x86_64
    )
  fi
  echo "downloading wheels for CPython $version (${platform_args[*]})"
  "$PY" -m pip download -r "$REQ_FILE" pathspec python-dotenv psutil \
    "${platform_args[@]}" \
    --implementation cp \
    --python-version "$version" \
    --only-binary=:all: \
    --dest "$WHEEL_DIR" \
    --quiet
done

REQ_SHA=$(sha256sum "$REQ_FILE" | awk '{print $1}')
cat > "$WHEEL_DIR/manifest.json" <<EOF
{
  "format": 2,
  "python_versions": "${PYTHON_VERSIONS}",
  "requirements_sha256": "${REQ_SHA}"
}
EOF

mkdir -p bench/artifacts
tar -czf bench/artifacts/mucli-wheelhouse.tar.gz -C "$TMP" wheelhouse
ls -la bench/artifacts/mucli-wheelhouse.tar.gz
echo "wheelhouse built: $(find "$WHEEL_DIR" -maxdepth 1 -name '*.whl' | wc -l) wheels"
