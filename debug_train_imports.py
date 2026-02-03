"""
Test the exact import pattern that train.py uses
"""
import sys

print("="*60)
print("Testing train.py's import pattern")
print("="*60)

# This is what train.py does
print("\n1. Importing as train.py does:")
print("   from basicsr.data import build_dataloader, build_dataset")

try:
    from basicsr.data import build_dataloader, build_dataset
    print("   ✓ Import successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

print("\n2. Checking registry:")
from basicsr.utils.registry import DATASET_REGISTRY
print(f"   Registered datasets: {list(DATASET_REGISTRY._obj_map.keys())}")

if 'LabDataset' not in DATASET_REGISTRY._obj_map:
    print("   ✗ LabDataset NOT registered!")
    print("\n3. Checking __init__.py verification:")
    print("   The verification check in __init__.py should have raised ImportError")
    print("   But it didn't, which means the check isn't running!")
    sys.exit(1)

print("   ✓ LabDataset is registered")

print("\n3. Testing build_dataset:")
test_opt = {
    'type': 'LabDataset',
    'dataroot_gt': 'C:/Users/nimet/Documents/IIT/L6/FYP/Dataset/SkinTan',
    'meta_info_file': 'meta_train.txt',
    'phase': 'train',
    'io_backend': {'type': 'disk'},
    'gt_size': 128,
    'use_shuffle': True,
    'use_hflip': True,
    'use_rot': False,
    'do_cutmix': False,
    'cutmix_p': 0.5,
    'do_fmix': False,
    'fmix_p': 0.5,
}

try:
    dataset = build_dataset(test_opt.copy())
    print(f"   ✓ Dataset built: {type(dataset)}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ Train.py import pattern works!")
print("="*60)
