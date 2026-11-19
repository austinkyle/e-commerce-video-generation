#!/bin/bash
set -e
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it, then run this launcher again."
  read -r -p "Press Return to close..." _
  exit 1
fi
uv run ecom-video serve
