#!/bin/bash
set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8765}"

if ! command -v lsof >/dev/null 2>&1; then
  echo "lsof is required to find the local app process."
  read -r -p "Press Return to close..." _
  exit 1
fi

PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -z "$PIDS" ]; then
  echo "No app is listening on port $PORT."
  read -r -p "Press Return to close..." _
  exit 0
fi

STOPPED=0
for PID in $PIDS; do
  COMMAND="$(ps -p "$PID" -o command= 2>/dev/null || true)"
  case "$COMMAND" in
    *"$PROJECT_DIR/.venv/bin/ecom-video serve"*)
      kill "$PID"
      STOPPED=1
      echo "Stopped E-Commerce Video Generator (PID $PID) on port $PORT."
      ;;
    *)
      echo "Skipped PID $PID because it does not appear to be this project's server."
      ;;
  esac
done

if [ "$STOPPED" -eq 0 ]; then
  echo "No matching E-Commerce Video Generator process was stopped."
fi
read -r -p "Press Return to close..." _
