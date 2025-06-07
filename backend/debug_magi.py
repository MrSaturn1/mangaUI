#!/usr/bin/env python3
"""
Debug script to investigate Magi v2 predict_crop_embeddings method
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel

def debug_magi_predict_crop_embeddings():
    print("Loading Magi v2 model...")
    
    # Load Magi v2 model
    magi_model = AutoModel.from_pretrained(
        "ragavsachdeva/magiv2", 
        trust_remote_code=True
    )
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    magi_model = magi_model.to(device)
    magi_model.eval()
    
    print(f"Model loaded on device: {device}")
    print(f"Model type: {type(magi_model)}")
    print(f"Has predict_crop_embeddings: {hasattr(magi_model, 'predict_crop_embeddings')}")
    
    if hasattr(magi_model, 'predict_crop_embeddings'):
        method = getattr(magi_model, 'predict_crop_embeddings')
        print(f"Method type: {type(method)}")
        print(f"Method: {method}")
        
        # Try to inspect the method signature
        import inspect
        try:
            sig = inspect.signature(method)
            print(f"Method signature: {sig}")
        except Exception as e:
            print(f"Could not get signature: {e}")
    
    # Create a simple test image
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    images = [image]
    
    # Create bounding box
    crop_bboxes = [[0, 0, 512, 512]]
    
    print(f"Test image shape: {image.shape}")
    print(f"Test bboxes: {crop_bboxes}")
    
    # Try calling the method with different parameter combinations
    test_cases = [
        # Case 1: Basic call
        {
            "images": images,
            "crop_bboxes": crop_bboxes
        },
        # Case 2: With all parameters
        {
            "images": images,
            "crop_bboxes": crop_bboxes,
            "move_to_device_fn": None,
            "mask_ratio": 0.0,
            "batch_size": 1
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        try:
            result = magi_model.predict_crop_embeddings(**case)
            print(f"Success! Result type: {type(result)}")
            print(f"Result: {result}")
            if hasattr(result, 'shape'):
                print(f"Result shape: {result.shape}")
            elif isinstance(result, (list, tuple)):
                print(f"Result length: {len(result)}")
                if len(result) > 0:
                    print(f"First element type: {type(result[0])}")
                    if hasattr(result[0], 'shape'):
                        print(f"First element shape: {result[0].shape}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_magi_predict_crop_embeddings()