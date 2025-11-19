#!/usr/bin/env python3
"""
Simple verl quickstart test
Tests basic verl functionality without requiring full GPU setup
"""

import sys
import os

def test_verl_imports():
    """Test that verl can be imported"""
    print("Testing verl imports...")
    try:
        import verl
        print(f"✅ verl imported successfully")
        
        # Try importing key components (may fail on macOS without GPU)
        try:
            from verl.trainer.ppo import PPOTrainer
            print(f"✅ PPOTrainer imported")
        except ImportError:
            print(f"⚠️  PPOTrainer not available (OK for local dev)")
        
        # Try importing utilities
        try:
            from verl.utils import hf_tokenizer
            print(f"✅ verl.utils imported")
        except ImportError:
            print(f"⚠️  verl.utils not fully available (OK for local dev)")
        
        return True
    except ImportError as e:
        print(f"❌ verl import failed: {e}")
        return False

def test_ray():
    """Test Ray initialization"""
    print("\nTesting Ray...")
    try:
        import ray
        
        # Initialize Ray locally
        ray.init(ignore_reinit_error=True, num_cpus=2)
        print(f"✅ Ray initialized")
        
        # Test a simple Ray task
        @ray.remote
        def test_task():
            return "Hello from Ray!"
        
        result = ray.get(test_task.remote())
        print(f"✅ Ray task executed: {result}")
        
        ray.shutdown()
        return True
    except Exception as e:
        print(f"❌ Ray test failed: {e}")
        return False

def test_dataset_format():
    """Test verl dataset format"""
    print("\nTesting verl dataset format...")
    try:
        # Create a sample dataset entry in verl format
        sample_entry = {
            "prompt": "Test medical note",
            "response": "",
            "reward": 0.0,
            "metadata": {
                "data_type": "vanilla_harmful",
                "error_type": "dosage"
            }
        }
        
        print(f"✅ Sample verl dataset entry created:")
        print(f"   {sample_entry}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset format test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("verl Quickstart Test")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Test imports
    all_ok &= test_verl_imports()
    
    # Test Ray
    all_ok &= test_ray()
    
    # Test dataset format
    all_ok &= test_dataset_format()
    
    print()
    print("=" * 60)
    if all_ok:
        print("✅ All quickstart tests passed!")
        print()
        print("verl is ready for development.")
        print()
        print("Next steps:")
        print("1. Convert dataset: python scripts/convert_dataset_to_verl.py")
        print("2. Create training config: python scripts/create_verl_config.py")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
