#!/bin/bash

# Install Dependencies for Medical Self-Play Training
# Handles OpenRLHF installation from Self-RedTeam fork

set -e

echo "=========================================="
echo "Installing Dependencies"
echo "=========================================="
echo ""

# 1. Install OpenRLHF from Self-RedTeam fork
echo "1. Installing OpenRLHF (Self-RedTeam fork with REINFORCE++)..."
if [ ! -d "selfplay-redteaming-reference" ]; then
    echo "❌ Error: selfplay-redteaming-reference directory not found"
    echo "Please ensure the repository is cloned with submodules"
    exit 1
fi

cd selfplay-redteaming-reference
pip install -e .
cd ..
echo "✅ OpenRLHF installed"

# 2. Setup medical_team as red_team module
echo ""
echo "2. Setting up medical_team module..."
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team
echo "✅ medical_team copied as red_team"

# 3. Install other requirements
echo ""
echo "3. Installing other requirements..."
pip install -r requirements.txt
echo "✅ Requirements installed"

# 4. Verify installation
echo ""
echo "4. Verifying installation..."

# Check OpenRLHF
if python -c "import openrlhf" 2>/dev/null; then
    echo "✅ OpenRLHF imported successfully"
else
    echo "❌ OpenRLHF import failed"
    exit 1
fi

# Check red_team module
if [ -f "selfplay-redteaming-reference/red_team/__init__.py" ]; then
    echo "✅ red_team module ready"
else
    echo "❌ red_team module not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Dependencies Installed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download model: huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft --local-dir trainer_output/qwen3-4b-medical-selfplay-sft"
echo "2. Generate data: python scripts/create_rl_training_data.py && python scripts/convert_to_openrlhf_format.py"
echo "3. Train: ./launch_training.sh"
echo ""
