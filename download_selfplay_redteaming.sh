#!/bin/bash

# Download Self-RedTeam Repository (without .git to avoid conflicts)
# URL: https://github.com/mickelliu/selfplay-redteaming

set -e

echo "=========================================="
echo "Downloading Self-RedTeam Repository"
echo "=========================================="
echo ""

REPO_URL="https://github.com/mickelliu/selfplay-redteaming"
TARGET_DIR="selfplay-redteaming-reference"

# Remove existing directory if present
if [ -d "$TARGET_DIR" ]; then
    echo "Removing existing $TARGET_DIR directory..."
    rm -rf "$TARGET_DIR"
fi

# Clone the repository
echo "Cloning repository..."
git clone "$REPO_URL" "$TARGET_DIR"

# Remove .git directory to avoid conflicts with your main repo
echo ""
echo "Removing .git directory to avoid conflicts..."
rm -rf "$TARGET_DIR/.git"

echo ""
echo "=========================================="
echo "✅ Self-RedTeam Downloaded Successfully!"
echo "=========================================="
echo ""

# Verify download
if [ -d "$TARGET_DIR/openrlhf" ]; then
    echo "✅ OpenRLHF directory found"
else
    echo "❌ Error: OpenRLHF directory not found"
    exit 1
fi

if [ -d "$TARGET_DIR/red_team" ]; then
    echo "✅ red_team directory found"
else
    echo "⚠️  red_team directory not found (will be replaced with medical_team)"
fi

echo ""
echo "Directory structure:"
ls -la "$TARGET_DIR" | head -15

echo ""
echo "Next steps:"
echo "1. Install dependencies: ./install_dependencies.sh"
echo "2. Download model: ./download_model.sh"
echo "3. Generate data: python scripts/create_rl_training_data.py && python scripts/convert_to_openrlhf_format.py"
echo "4. Train: ./launch_training.sh"
echo ""
