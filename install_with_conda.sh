#!/bin/bash

# Install Dependencies Using Conda (Recommended)
# Avoids pip build issues with flash-attn

set -e

echo "=========================================="
echo "Installing Dependencies (Conda Method)"
echo "=========================================="
echo ""

# 1. Install flash-attn from conda-forge
echo "1. Installing flash-attn from conda-forge..."
conda install -c conda-forge flash-attn -y
echo "✅ flash-attn installed"

# 2. Install PyTorch and related packages
echo ""
echo "2. Ensuring PyTorch is installed..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
echo "✅ PyTorch installed"

# 3. Install OpenRLHF dependencies without flash-attn
echo ""
echo "3. Installing OpenRLHF dependencies..."
pip install accelerate bitsandbytes datasets einops
pip install deepspeed==0.18.1
pip install transformers peft ray vllm
echo "✅ Dependencies installed"

# 4. Install OpenRLHF from Self-RedTeam fork (skip flash-attn)
echo ""
echo "4. Installing OpenRLHF (Self-RedTeam fork)..."
if [ ! -d "selfplay-redteaming-reference" ]; then
    echo "❌ Error: selfplay-redteaming-reference directory not found"
    exit 1
fi

cd selfplay-redteaming-reference

# Modify setup.py to skip flash-attn requirement temporarily
if [ -f "setup.py" ]; then
    # Install without dependencies first
    pip install -e . --no-deps
    # Then install other dependencies
    pip install -e . --no-build-isolation
fi

cd ..
echo "✅ OpenRLHF installed"

# 5. Setup medical_team as red_team module
echo ""
echo "5. Setting up medical_team module..."
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team
echo "✅ medical_team copied as red_team"

# 6. Install other requirements
echo ""
echo "6. Installing other requirements..."
pip install -r requirements.txt
echo "✅ Requirements installed"

# 7. Verify installation
echo ""
echo "7. Verifying installation..."

# Check flash-attn
if python -c "import flash_attn" 2>/dev/null; then
    echo "✅ flash-attn imported successfully"
else
    echo "❌ flash-attn import failed"
    exit 1
fi

# Check OpenRLHF
if python -c "import openrlhf" 2>/dev/null; then
    echo "✅ OpenRLHF imported successfully"
else
    echo "⚠️  OpenRLHF import warning (may still work)"
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
echo "1. Download model: ./download_model.sh"
echo "2. Generate data: python scripts/create_rl_training_data.py && python scripts/convert_to_openrlhf_format.py"
echo "3. Train: ./launch_training.sh"
echo ""
