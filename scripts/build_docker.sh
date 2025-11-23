#!/bin/bash
# Build script for Docker images
# Supports CPU, GPU, and production builds

set -e

echo "=========================================="
echo "Bark Infinity - Docker Build Script"
echo "=========================================="

# Parse arguments
BUILD_TARGET="${1:-production}"
IMAGE_TAG="${2:-latest}"

echo "Build target: $BUILD_TARGET"
echo "Image tag: $IMAGE_TAG"
echo ""

# Build the Docker image
echo "Building Docker image..."
docker build \
    --target "$BUILD_TARGET" \
    --tag "bark-infinity:$IMAGE_TAG" \
    --tag "bark-infinity:$BUILD_TARGET-$IMAGE_TAG" \
    .

echo ""
echo "=========================================="
echo "Build successful!"
echo "=========================================="
echo ""
echo "Available images:"
docker images | grep bark-infinity
echo ""
echo "To run the container:"
echo "  docker run -p 7860:7860 bark-infinity:$IMAGE_TAG"
echo ""
echo "With GPU support:"
echo "  docker run --gpus all -p 7860:7860 bark-infinity:gpu-$IMAGE_TAG"
echo ""
echo "Using docker-compose:"
echo "  docker-compose up bark-infinity"
echo ""
