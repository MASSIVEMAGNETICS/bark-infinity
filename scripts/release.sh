#!/bin/bash
# Release script for creating GitHub releases with artifacts
# This script builds all release artifacts and creates a release

set -e

VERSION="${1:-0.1.0}"
RELEASE_NAME="Bark Infinity v$VERSION"

echo "=========================================="
echo "Creating release: $RELEASE_NAME"
echo "=========================================="

# Create release directory
RELEASE_DIR="release-$VERSION"
mkdir -p "$RELEASE_DIR"

echo ""
echo "Step 1: Building Python package..."
python -m build
cp dist/*.whl "$RELEASE_DIR/"
cp dist/*.tar.gz "$RELEASE_DIR/"

echo ""
echo "Step 2: Building Docker images..."
bash scripts/build_docker.sh production "$VERSION"

echo ""
echo "Step 3: Creating documentation package..."
mkdir -p "$RELEASE_DIR/docs"
cp README.md "$RELEASE_DIR/docs/"
cp LICENSE "$RELEASE_DIR/docs/"
cp DEPLOYMENT.md "$RELEASE_DIR/docs/" 2>/dev/null || echo "DEPLOYMENT.md not found, skipping..."

echo ""
echo "Step 4: Creating deployment templates..."
mkdir -p "$RELEASE_DIR/deployment"
cp docker-compose.yml "$RELEASE_DIR/deployment/"
cp Dockerfile "$RELEASE_DIR/deployment/"

echo ""
echo "Step 5: Creating checksum file..."
cd "$RELEASE_DIR"
sha256sum * > checksums.txt
cd ..

echo ""
echo "=========================================="
echo "Release artifacts created in: $RELEASE_DIR/"
echo "=========================================="
echo ""
ls -lh "$RELEASE_DIR/"
echo ""
echo "Next steps:"
echo "1. Review the release artifacts"
echo "2. Create a GitHub release with tag v$VERSION"
echo "3. Upload the artifacts from $RELEASE_DIR/"
echo "4. Update CHANGELOG.md with release notes"
echo ""
