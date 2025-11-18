#!/bin/bash
set -e

echo "=========================================="
echo "Patching OpenRLHF Flash-Attention Import"
echo "=========================================="

# Find OpenRLHF installation
OPENRLHF_PATH=$(python -c "import sys; import os; paths = [p for p in sys.path if 'site-packages' in p or 'openrlhf' in p]; print([p for p in paths if os.path.exists(os.path.join(p, 'openrlhf'))][0] if any(os.path.exists(os.path.join(p, 'openrlhf')) for p in paths) else '')" 2>/dev/null || echo "")

if [ -z "$OPENRLHF_PATH" ]; then
    # Try alternative method
    OPENRLHF_PATH=$(find /workspace -type d -name "openrlhf" -path "*/site-packages/*" 2>/dev/null | head -1)
    if [ -n "$OPENRLHF_PATH" ]; then
        OPENRLHF_PATH=$(dirname "$OPENRLHF_PATH")
    fi
fi

if [ -z "$OPENRLHF_PATH" ]; then
    echo "❌ Could not find OpenRLHF installation"
    echo "Trying to find it manually..."
    find /workspace/miniconda3/envs/medical_reward -name "actor.py" -path "*/openrlhf/*" 2>/dev/null | head -5
    exit 1
fi

echo "Found OpenRLHF at: $OPENRLHF_PATH"

ACTOR_FILE="$OPENRLHF_PATH/openrlhf/models/actor.py"

if [ ! -f "$ACTOR_FILE" ]; then
    echo "❌ Could not find actor.py at $ACTOR_FILE"
    exit 1
fi

echo "Patching: $ACTOR_FILE"

# Backup original
cp "$ACTOR_FILE" "$ACTOR_FILE.backup"

# Patch the import to make it optional
python << 'EOF'
import sys

actor_file = sys.argv[1]

with open(actor_file, 'r') as f:
    content = f.read()

# Replace the hardcoded import with a try-except
old_import = "from flash_attn.utils.distributed import all_gather"
new_import = """try:
    from flash_attn.utils.distributed import all_gather
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    # Fallback: use torch's all_gather
    import torch.distributed as dist
    def all_gather(tensor, group=None):
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size(group)
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group=group)
        return torch.cat(tensor_list, dim=0)"""

if old_import in content:
    content = content.replace(old_import, new_import)
    with open(actor_file, 'w') as f:
        f.write(content)
    print(f"✅ Patched {actor_file}")
else:
    print(f"⚠️  Import already patched or not found in {actor_file}")

EOF

python "$ACTOR_FILE" "$ACTOR_FILE"

echo ""
echo "=========================================="
echo "✅ OpenRLHF Patched Successfully!"
echo "=========================================="
echo ""
echo "Backup saved at: $ACTOR_FILE.backup"
echo ""
echo "Next step: bash launch_training_no_flash.sh"
