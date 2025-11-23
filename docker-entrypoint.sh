#!/bin/bash
set -e

# Auto-detect GPU and set appropriate environment
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, using GPU mode"
    export SUNO_OFFLOAD_CPU=False
    export SUNO_USE_SMALL_MODELS=False
else
    echo "No GPU detected, using CPU mode with optimizations"
    export SUNO_OFFLOAD_CPU=True
    export SUNO_USE_SMALL_MODELS=True
fi

# Execute the command
exec "$@"
