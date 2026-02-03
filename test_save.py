import os
import torch
import sys
from pathlib import Path

def test_save_mechanism():
    """Test if we can save model files to the experiments directory"""
    
    print("="*60)
    print("Testing Model Save Mechanism")
    print("="*60)
    
    # 1. Check disk space
    print("\n1. Checking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        print(f"   Free disk space: {free_gb:.2f} GB")
        if free_gb < 1:
            print("   ⚠️  WARNING: Less than 1GB free!")
        else:
            print("   ✓ Sufficient disk space")
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
    
    # 2. Check experiments directory
    print("\n2. Checking experiments directory...")
    exp_dir = Path("experiments/TanColorize_FastSprint")
    models_dir = exp_dir / "models"
    states_dir = exp_dir / "training_states"
    
    print(f"   Experiment dir: {exp_dir}")
    print(f"   Models dir: {models_dir}")
    print(f"   States dir: {states_dir}")
    
    if not exp_dir.exists():
        print("   ⚠️  Experiment directory doesn't exist")
        return False
    
    if not models_dir.exists():
        print("   ⚠️  Models directory doesn't exist, creating...")
        models_dir.mkdir(parents=True, exist_ok=True)
    
    if not states_dir.exists():
        print("   ⚠️  States directory doesn't exist, creating...")
        states_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Test write permissions
    print("\n3. Testing write permissions...")
    test_file = models_dir / "test_save.pth"
    
    try:
        # Create a dummy model checkpoint
        dummy_data = {
            'params': {'test': torch.randn(10, 10)},
            'iter': 0
        }
        torch.save(dummy_data, test_file)
        print(f"   ✓ Successfully saved test file: {test_file}")
        
        # Verify file exists and has size
        if test_file.exists():
            size = test_file.stat().st_size
            print(f"   ✓ Test file exists, size: {size} bytes")
        else:
            print("   ❌ Test file doesn't exist after save!")
            return False
        
        # Try to load it back
        loaded = torch.load(test_file)
        if 'params' in loaded:
            print("   ✓ Successfully loaded test file")
        else:
            print("   ❌ Loaded file has incorrect structure")
            return False
        
        # Clean up
        test_file.unlink()
        print("   ✓ Cleanup successful")
        
    except Exception as e:
        print(f"   ❌ Save test FAILED: {e}")
        return False
    
    # 4. Check for old files
    print("\n4. Checking for existing model files...")
    existing_models = list(models_dir.glob("*.pth"))
    if existing_models:
        print(f"   ⚠️  Found {len(existing_models)} existing model file(s):")
        for f in existing_models:
            print(f"      - {f.name}")
        print("   These will be overwritten during training")
    else:
        print("   ✓ No existing model files (clean start)")
    
    print("\n" + "="*60)
    print("✅ Save mechanism test PASSED")
    print("="*60)
    print("\nYou can proceed with training!")
    return True

if __name__ == "__main__":
    success = test_save_mechanism()
    sys.exit(0 if success else 1)
