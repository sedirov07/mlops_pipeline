#!/usr/bin/env bash
set -euo pipefail

# Start MLflow locally using project's .venv
# Usage:
#   ./run_mlflow.sh [--port 5000] [--backend file:/abs/path/mlruns] [--artifacts /abs/path/mlruns]

PORT=5000

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_URI="file:${PROJECT_ROOT}/mlruns"
ARTIFACT_ROOT="${PROJECT_ROOT}/mlruns"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"; shift 2 ;;
    --backend)
      BACKEND_URI="$2"; shift 2 ;;
    --artifacts)
      ARTIFACT_ROOT="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

# Activate venv if present
if [[ -f .venv/Scripts/activate ]]; then
  source .venv/Scripts/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

command -v mlflow >/dev/null 2>&1 || {
  echo "❌ mlflow is not installed in the active environment"
  exit 1
}

echo "🚀 Starting MLflow UI"
echo "  Backend URI:  $BACKEND_URI"
echo "  Artifacts:    $ARTIFACT_ROOT"
echo "  Port:         $PORT"

mlflow ui \
  --backend-store-uri "$BACKEND_URI" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --host 127.0.0.1 \
  --port "$PORT"
