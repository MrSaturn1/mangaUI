#!/usr/bin/env python3
"""
Convert Magi embeddings from numpy arrays to PyTorch tensors
to ensure compatibility with the manga generation pipeline.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_magi_embeddings_to_torch():
    """Convert all .pt files from numpy arrays to PyTorch tensors"""
    embeddings_dir = Path("character_output/character_embeddings")
    
    if not embeddings_dir.exists():
        print(f"Embeddings directory not found: {embeddings_dir}")
        return
    
    # Get all .pt files (but skip backup files)
    pt_files = list(embeddings_dir.glob("*.pt"))
    pt_files = [f for f in pt_files if not f.name.endswith("_clip_backup.pt") and not f.name.endswith("_magi.pt")]
    
    print(f"Found {len(pt_files)} embedding files to convert")
    
    converted_count = 0
    for pt_file in tqdm(pt_files, desc="Converting embeddings"):
        try:
            # Load the current embedding
            embedding = torch.load(pt_file, weights_only=False)
            
            # Check if it's a numpy array
            if isinstance(embedding, np.ndarray):
                print(f"Converting {pt_file.name} from numpy array to torch tensor")
                
                # Convert to PyTorch tensor
                tensor_embedding = torch.from_numpy(embedding).float()
                
                # Save back as PyTorch tensor
                torch.save(tensor_embedding, pt_file)
                converted_count += 1
                
            elif isinstance(embedding, torch.Tensor):
                print(f"{pt_file.name} is already a PyTorch tensor")
            else:
                print(f"Warning: {pt_file.name} contains unexpected type: {type(embedding)}")
                
        except Exception as e:
            print(f"Error processing {pt_file.name}: {e}")
    
    print(f"Converted {converted_count} embeddings from numpy arrays to PyTorch tensors")

if __name__ == "__main__":
    convert_magi_embeddings_to_torch()