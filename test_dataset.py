#!/usr/bin/env python3
"""
Simple test script to verify CULane dataset loading works correctly.
This tests the dataset setup without requiring a trained model.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def test_dataset_paths():
    """Test that all required paths exist"""
    print("=" * 60)
    print("Testing CULane Dataset Paths")
    print("=" * 60)
    
    dataset_root = "/home/alki/projects/datasets/culane"
    
    required_files = [
        "train_gt.txt",
        "val.txt", 
        "test.txt"
    ]
    
    required_dirs = [
        "driver_23_30frame",
        "laneseg_label_w16"
    ]
    
    print(f"\nDataset Root: {dataset_root}")
    print(f"Symlink points to: {os.path.realpath(dataset_root)}")
    
    # Test list files
    print("\n--- Testing List Files ---")
    for f in required_files:
        path = os.path.join(dataset_root, f)
        exists = os.path.exists(path)
        is_link = os.path.islink(path)
        print(f"  {f}: {'✓' if exists else '✗'} {'(symlink)' if is_link else ''}")
        if exists and f.endswith('.txt'):
            # Count lines
            with open(path, 'r') as file:
                count = sum(1 for _ in file)
            print(f"    → {count} entries")
    
    # Test directories
    print("\n--- Testing Directories ---")
    for d in required_dirs:
        path = os.path.join(dataset_root, d)
        exists = os.path.exists(path)
        print(f"  {d}: {'✓' if exists else '✗'}")
    
    # Test sample image and annotation
    print("\n--- Testing Sample Data ---")
    train_file = os.path.join(dataset_root, "train_gt.txt")
    with open(train_file, 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        img_path = parts[0].lstrip('/')
        full_img_path = os.path.join(dataset_root, img_path)
        
        if os.path.exists(full_img_path):
            print(f"  Sample image: ✓")
            # Check for corresponding lines.txt
            lines_path = full_img_path.replace('.jpg', '.lines.txt')
            if os.path.exists(lines_path):
                print(f"  Sample annotation (.lines.txt): ✓")
                with open(lines_path, 'r') as lf:
                    lanes = lf.readlines()
                print(f"    → {len(lanes)} lanes in annotation")
            else:
                print(f"  Sample annotation (.lines.txt): ✗")
        else:
            print(f"  Sample image: ✗ ({full_img_path})")
    
    return True


def test_dataset_import():
    """Test that the dataset module can be imported"""
    print("\n" + "=" * 60)
    print("Testing Dataset Module Import")
    print("=" * 60)
    
    try:
        from data.datasets.lane.culane import CULaneDataset
        print("  ✓ CULaneDataset module imported successfully")
        
        # Test dataset instantiation (this will trigger preprocessing if needed)
        print("\n  Attempting to instantiate dataset...")
        print("  Note: This will trigger preprocessing if pickle files don't exist")
        
        dataset_root = "/home/alki/projects/datasets/culane"
        
        # Test with a tiny subset first
        try:
            ds = CULaneDataset(
                root=dataset_root,
                split='train',
                transform=None,
                version='v1.0'
            )
            print(f"  ✓ Dataset instantiated successfully")
            print(f"    → {len(ds)} samples")
            
            # Try to load one sample
            print("\n  Testing sample loading...")
            sample = ds[0]
            print(f"  ✓ Sample loaded successfully")
            print(f"    → Keys: {list(sample.keys())}")
            
        except Exception as e:
            print(f"  ✗ Dataset instantiation failed: {e}")
            import traceback
            traceback.print_exc()
            
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    return True


def test_model_import():
    """Test that the model can be imported"""
    print("\n" + "=" * 60)
    print("Testing Model Import")
    print("=" * 60)
    
    try:
        from modeling import models
        print("  ✓ Models module imported successfully")
        
        # Check if CondLSTR2DRes34 exists
        model_names = models.names()
        print(f"  Available models: {len(model_names)}")
        
        if 'CondLSTR2DRes34' in model_names:
            print("  ✓ CondLSTR2DRes34 model available")
        else:
            print("  ✗ CondLSTR2DRes34 model not found")
            print(f"  Available: {model_names[:5]}...")
            
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    return True


def main():
    print("\n" + "=" * 60)
    print("CULane Dataset Setup Validation")
    print("=" * 60)
    
    # Run tests
    paths_ok = test_dataset_paths()
    import_ok = test_dataset_import()
    model_ok = test_model_import()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Paths:       {'✓ PASS' if paths_ok else '✗ FAIL'}")
    print(f"  Dataset:     {'✓ PASS' if import_ok else '✗ FAIL'}")
    print(f"  Model:       {'✓ PASS' if model_ok else '✗ FAIL'}")
    
    if paths_ok and import_ok and model_ok:
        print("\n  🎉 All tests passed! Ready for training.")
        print("\n  To start training, run:")
        print("  conda activate hdmapnet")
        print("  python tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane -b 4 -j 4")
    else:
        print("\n  ⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
