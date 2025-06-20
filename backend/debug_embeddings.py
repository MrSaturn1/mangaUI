#!/usr/bin/env python3
# Save as debug_embeddings.py

import torch
from pathlib import Path

def check_embedding_format():
    # Check a few different embedding types
    embeddings_dir = Path("character_output/character_embeddings")
    
    # Try to find different embedding files for the same character
    test_char = "Z"  # or whichever character you tested
    
    files_to_check = [
        f"{test_char}.pt",           # Main embedding (could be hybrid)
        f"{test_char}_hybrid.pt",    # Hybrid embedding
        f"{test_char}_clip.pt",      # CLIP embedding
        f"{test_char}_magi.pt",      # Magi embedding
    ]
    
    print("🔍 EMBEDDING FORMAT ANALYSIS")
    print("=" * 40)
    
    for filename in files_to_check:
        filepath = embeddings_dir / filename
        if filepath.exists():
            try:
                embedding = torch.load(filepath, map_location='cpu')
                print(f"\n📁 {filename}:")
                print(f"   Type: {type(embedding)}")
                print(f"   Shape: {embedding.shape if hasattr(embedding, 'shape') else 'No shape'}")
                print(f"   Dtype: {embedding.dtype if hasattr(embedding, 'dtype') else 'No dtype'}")
                
                if hasattr(embedding, 'shape'):
                    print(f"   Dimensions: {len(embedding.shape)}D")
                    if len(embedding.shape) == 1:
                        print(f"   ✅ 1D vector: {embedding.shape[0]} elements")
                    elif len(embedding.shape) == 2:
                        print(f"   ⚠️  2D matrix: {embedding.shape}")
                    else:
                        print(f"   ❌ Unexpected shape: {embedding.shape}")
                        
                    # Check if it's 768 dimensions (expected)
                    flat_size = embedding.numel()
                    if flat_size == 768:
                        print(f"   ✅ Correct size (768)")
                    else:
                        print(f"   ⚠️  Size mismatch: {flat_size} (expected 768)")
                
                # Show value range
                if hasattr(embedding, 'min'):
                    print(f"   Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
                    print(f"   Mean: {embedding.mean():.4f}")
                    
            except Exception as e:
                print(f"\n❌ {filename}: Error loading - {e}")
        else:
            print(f"\n⚪ {filename}: Not found")

if __name__ == "__main__":
    check_embedding_format()