"""
Debug script to trace exactly when and where LabDataset gets registered
"""
import sys

print("="*60)
print("Step 1: Check if basicsr.data.lab_dataset can be imported")
print("="*60)

try:
    from basicsr.data import lab_dataset
    print("✓ basicsr.data.lab_dataset imported successfully")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Step 2: Check Registry after importing lab_dataset module")
print("="*60)

from basicsr.utils.registry import DATASET_REGISTRY
print(f"Registered datasets: {list(DATASET_REGISTRY._obj_map.keys())}")

if 'LabDataset' in DATASET_REGISTRY._obj_map:
    print("✓ LabDataset is registered")
else:
    print("✗ LabDataset is NOT registered")
    sys.exit(1)

print("\n" + "="*60)
print("Step 3: Try to build a dataset")
print("="*60)

from basicsr.data import build_dataset

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
    'batch_size_per_gpu': 2,
    'num_worker_per_gpu': 0
}

try:
    # Make a copy since build_dataset pops 'type'
    test_opt_copy = test_opt.copy()
    dataset = build_dataset(test_opt_copy)
    print(f"✓ Dataset built successfully: {type(dataset)}")
    print(f"✓ Dataset length: {len(dataset)}")
except Exception as e:
    print(f"✗ Failed to build dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL CHECKS PASSED")
print("="*60)
print("\nThe dataset system works fine when imported directly.")
print("The issue must be in how train.py imports the modules.")
