#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Default action is to run the API
ACTION=${1:-api}

# Set PYTHONPATH if your src layout requires it and it's not handled by Dockerfile WORKDIR or Python packaging
export PYTHONPATH=${PYTHONPATH}:${APP_HOME}/src:${APP_HOME}

echo "Executing action: $ACTION"

if [ "$ACTION" = "api" ]; then
    echo "Starting FastAPI application..."
    exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info
elif [ "$ACTION" = "pipeline" ]; then
    STAGE=${2:-all} # Second argument for pipeline stage, default to 'all'
    echo "Running MLOps pipeline (stage: $STAGE)..."
    exec python main.py --stage "$STAGE"
elif [ "$ACTION" = "shell" ]; then
    echo "Starting bash shell..."
    exec bash
else
    echo "Unknown action: $ACTION"
    echo "Available actions: api, pipeline [stage], shell"
    exit 1
fi