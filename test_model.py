#!/usr/bin/env python3
"""
Simple test script to verify the CondLSTR2DRes34 model works correctly.
This tests model creation and forward pass without requiring trained weights.
"""

import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def test_model_creation():
    """Test that the model can be created"""
    print("=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    try:
        from modeling import models
        
        # Create model
        print("Creating CondLSTR2DRes34 model...")
        model = models.create('CondLSTR2DRes34', num_classes=21)
        
        print(f"✓ Model created successfully!")
        print(f"  Model type: {type(model).__name__}")
        
        # Count parameters
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_parameters:,}")
        
        return model
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_forward(model):
    """Test that the model can do a forward pass"""
    print("\n" + "=" * 60)
    print("Testing Model Forward Pass")
    print("=" * 60)
    
    if model is None:
        print("✗ No model to test (creation failed)")
        return False
    
    try:
        model.eval()
        
        # Create dummy input (CULane images are typically 1640x590)
        # We'll use a smaller size for testing
        batch_size = 2
        height, width = 480, 640
        print(f"Creating dummy input: batch={batch_size}, size={height}x{width}")
        
        dummy_input = torch.randn(batch_size, 3, height, width)
        
        # Move to CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        
        # Forward pass
        print("Running forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful!")
        print(f"  Output type: {type(output)}")
        
        if isinstance(output, dict):
            print(f"  Output keys: {list(output.keys())}")
            for key, value in output.items():
                if torch.is_tensor(value):
                    print(f"    {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        elif torch.is_tensor(output):
            print(f"  Output shape: {tuple(output.shape)}, dtype={output.dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_module():
    """Test that the lane_det_2d inference module works"""
    print("\n" + "=" * 60)
    print("Testing Inference Module")
    print("=" * 60)
    
    try:
        from modeling.inferences.lane.lane_det_2d import LaneDet2D
        from modeling import models
        
        # Create model
        model = models.create('CondLSTR2DRes34', num_classes=21)
        
        # Create inference module
        print("Creating LaneDet2D inference module...")
        inference = LaneDet2D(model)
        
        print("✓ Inference module created successfully!")
        print(f"  Inference type: {type(inference).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("CondLSTR2DRes34 Model Test")
    print("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    print()
    
    # Run tests
    model = test_model_creation()
    forward_ok = test_model_forward(model)
    inference_ok = test_inference_module()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Model Creation:    {'✓ PASS' if model is not None else '✗ FAIL'}")
    print(f"  Forward Pass:      {'✓ PASS' if forward_ok else '✗ FAIL'}")
    print(f"  Inference Module:  {'✓ PASS' if inference_ok else '✗ FAIL'}")
    
    if model is not None and forward_ok:
        print("\n  🎉 All tests passed! Model is ready for training.")
        print("\n  To start training, run:")
        print("  python tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane -b 4 -j 4")
    else:
        print("\n  ⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
