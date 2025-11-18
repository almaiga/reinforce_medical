#!/bin/bash

# Fix red_team module with updated medical_team
# This applies the GameOutcome import fix

set -e

echo "=========================================="
echo "Fixing red_team Module"
echo "=========================================="
echo ""

# Check if selfplay-redteaming-reference exists
if [ ! -d "selfplay-redteaming-reference" ]; then
    echo "❌ Error: selfplay-redteaming-reference directory not found"
    echo "Please run ./download_selfplay_redteaming.sh first"
    exit 1
fi

# Remove old red_team
echo "Removing old red_team module..."
rm -rf selfplay-redteaming-reference/red_team

# Copy updated medical_team as red_team
echo "Copying updated medical_team as red_team..."
cp -r medical_team selfplay-redteaming-reference/red_team

echo ""
echo "=========================================="
echo "✅ red_team Module Updated!"
echo "=========================================="
echo ""

# Verify the fix
echo "Verifying GameOutcome import..."
if python -c "import sys; sys.path.insert(0, 'selfplay-redteaming-reference'); from red_team import GameOutcome; print('✅ GameOutcome import works')" 2>/dev/null; then
    echo "✅ Import fix verified"
else
    echo "❌ Import verification failed"
    echo "Please check medical_team/__init__.py"
    exit 1
fi

echo ""
echo "You can now run: ./launch_training.sh"
echo ""
