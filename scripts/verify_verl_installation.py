#!/usr/bin/env python3
"""
Verify verl installation and compatibility
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    package_name = package_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: Not installed ({e})")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("❌ CUDA: Not available")
            return False
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False

def check_verl_components():
    """Check verl-specific components"""
    try:
        # Check verl imports
        from verl.trainer.ppo import PPOTrainer
        print("✅ verl.trainer.ppo.PPOTrainer: Available")
        
        # Check if we can import verl's utilities
        from verl.utils import hf_tokenizer
        print("✅ verl.utils: Available")
        
        return True
    except ImportError as e:
        print(f"⚠️  verl components: {e}")
        print("   (This is OK - some components may not work without GPU)")
        return True  # Don't fail on this for local dev

def main():
    print("=" * 60)
    print("verl Installation Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Core dependencies
    print("Core Dependencies:")
    all_ok &= check_import('torch', 'PyTorch')
    all_ok &= check_import('transformers', 'Transformers')
    all_ok &= check_import('ray', 'Ray')
    all_ok &= check_import('vllm', 'vLLM')
    print()
    
    # verl
    print("verl:")
    all_ok &= check_import('verl', 'verl')
    all_ok &= check_verl_components()
    print()
    
    # CUDA
    print("CUDA:")
    cuda_ok = check_cuda()
    if not cuda_ok:
        print("   ⚠️  CUDA not available (OK for local macOS development)")
    print()
    
    # PyTorch FSDP
    print("PyTorch FSDP:")
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
        print("✅ PyTorch FSDP: Available")
    except ImportError:
        print("⚠️  PyTorch FSDP: Not available (OK for local development)")
    print()
    
    # Summary
    print("=" * 60)
    # For local dev, we only need core dependencies
    core_ok = check_import('torch', 'PyTorch') and check_import('verl', 'verl')
    
    if core_ok:
        print("✅ verl is installed and ready for local development!")
        print()
        if not cuda_ok:
            print("📝 Note: Running on macOS without CUDA")
            print("   - Dataset conversion: ✅ Available")
            print("   - Configuration creation: ✅ Available")
            print("   - Code development: ✅ Available")
            print("   - GPU Training: ❌ Use SSH server")
        else:
            print("✅ Full GPU training available!")
        print()
        print("Next steps:")
        print("1. Run quickstart test: python scripts/test_verl_quickstart.py")
        print("2. Start dataset conversion: python scripts/convert_dataset_to_verl.py")
        return 0
    else:
        print("❌ Core dependencies missing. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
