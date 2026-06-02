#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/build_and_upload_package.sh gs://BUCKET/PREFIX" >&2
  exit 2
fi

DESTINATION_PREFIX="${1%/}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON_BIN=.venv/bin/python
  else
    PYTHON_BIN=python
  fi
fi

"$PYTHON_BIN" -m build --sdist --no-isolation >&2
PACKAGE_PATH="$(ls -t dist/monet_flow_tiny-*.tar.gz | head -n 1)"
PACKAGE_URI="${DESTINATION_PREFIX}/packages/$(basename "$PACKAGE_PATH")"
gcloud storage cp "$PACKAGE_PATH" "$PACKAGE_URI" >&2
echo "$PACKAGE_URI"
