#!/usr/bin/env python3
"""
Simple script to generate a proper OpenCLIP ViT-H-14 embedding for character Z
and replace the existing random embedding.
"""

import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import os

def load_embedding_model():
    """Load CLIP ViT-L/14 (768 dim) or try OpenCLIP alternatives"""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Try CLIP ViT-L/14 first (768 dim - most likely to work with drawatoon)
    try:
        import clip
        print("🔧 Loading CLIP ViT-L/14...")
        model, preprocess = clip.load("ViT-L/14", device=device)
        model.eval()
        print("✅ CLIP ViT-L/14 loaded successfully (768 dimensions)")
        return model, preprocess, device, "clip"
        
    except ImportError:
        print("⚠️  CLIP not available, trying OpenCLIP alternatives...")
        
    # Try OpenCLIP models that give 768 dimensions
    try:
        import open_clip
        
        # Try different OpenCLIP models that might give 768 dimensions
        models_to_try = [
            ('ViT-L-14', 'laion400m_e32'),
            ('ViT-L-14', 'laion2b_s32b_b82k'),
            ('ViT-B-32', 'laion2b_s34b_b79k'),
        ]
        
        for model_name, pretrained in models_to_try:
            try:
                print(f"🔧 Trying OpenCLIP {model_name} with {pretrained}...")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained=pretrained,
                    device=device
                )
                model.eval()
                
                # Test dimensions with a dummy input
                test_input = torch.randn(1, 3, 224, 224).to(device)
                with torch.no_grad():
                    test_output = model.encode_image(test_input)
                    dims = test_output.shape[-1]
                
                if dims == 768:
                    print(f"✅ OpenCLIP {model_name} loaded successfully ({dims} dimensions)")
                    return model, preprocess, device, "openclip"
                else:
                    print(f"⚠️  {model_name} gives {dims} dimensions, need 768")
                    
            except Exception as e:
                print(f"⚠️  Failed to load {model_name}: {e}")
                continue
        
        print("❌ No suitable OpenCLIP model found with 768 dimensions")
        
    except ImportError:
        print("❌ Neither CLIP nor OpenCLIP available. Please install:")
        print("  pip install clip-by-openai   # for CLIP (recommended)")
        print("  pip install open_clip_torch  # for OpenCLIP")
        
    return None, None, None, None

def find_z_image():
    """Find Z character image"""
    possible_paths = [
        "character_output/character_images/Z.png",
        "character_output/character_images/keepers/Z.png",
        "backend/character_output/character_images/Z.png", 
        "backend/character_output/character_images/keepers/Z.png"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"📁 Found Z image: {path}")
            return path
    
    print("❌ Z character image not found in expected locations:")
    for path in possible_paths:
        print(f"  - {path}")
    return None

def extract_768_embedding(image_path, model, preprocess, device, model_type):
    """Extract 768-dimensional embedding from image"""
    print(f"📸 Processing: {image_path}")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = model.encode_image(image_tensor)
    
    # Normalize features (important for consistency)
    features = features / features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy
    embedding = features.cpu().numpy().squeeze()
    
    print(f"📊 Generated embedding: shape={embedding.shape}, dtype={embedding.dtype}")
    print(f"📊 Value range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    
    # Verify dimensions
    if embedding.shape[-1] != 768:
        print(f"❌ ERROR: Expected 768 dimensions, got {embedding.shape}")
        return None
    
    print("✅ Perfect! 768-dimensional embedding generated")
    return embedding

def update_z_embedding(embedding):
    """Update Z's embedding in the project structure"""
    print("\n💾 UPDATING Z'S EMBEDDING")
    print("=" * 30)
    
    # Paths
    embedding_dir = Path("character_output/character_embeddings")
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    z_embedding_path = embedding_dir / "Z.pt"
    embeddings_map_path = embedding_dir / "character_embeddings_map.json"
    
    # Convert to tensor and save
    embedding_tensor = torch.from_numpy(embedding).float()
    torch.save(embedding_tensor, z_embedding_path)
    print(f"💾 Saved: {z_embedding_path}")
    
    # Update embeddings map
    if embeddings_map_path.exists():
        with open(embeddings_map_path, 'r') as f:
            embeddings_map = json.load(f)
    else:
        embeddings_map = {}
    
    # Update Z's entry
    embeddings_map["Z"] = {
        "name": "Z",
        "image_path": "character_output/character_images/Z.png",
        "embedding_path": str(z_embedding_path),
        "embedding_type": "openclip_vit_h_14_768d",
        "updated": "true"
    }
    
    # Save updated map
    with open(embeddings_map_path, 'w') as f:
        json.dump(embeddings_map, f, indent=2)
    print(f"💾 Updated: {embeddings_map_path}")
    
    print("✅ Z's embedding successfully updated!")
    print("\n🧪 Ready for testing!")
    print("You can now test character consistency through your frontend.")

def main():
    print("🎯 GENERATING PROPER EMBEDDING FOR CHARACTER Z")
    print("=" * 50)
    
    # Load model
    model, preprocess, device, model_type = load_embedding_model()
    if model is None:
        return
    
    # Find Z's image
    z_image_path = find_z_image()
    if z_image_path is None:
        return
    
    # Generate embedding
    embedding = extract_768_embedding(z_image_path, model, preprocess, device, model_type)
    if embedding is None:
        return
    
    # Update Z's embedding
    update_z_embedding(embedding)
    
    print("\n🎉 SUCCESS!")
    print("=" * 20)
    print("Z now has a proper OpenCLIP/CLIP embedding.")
    print("Test character consistency in your frontend by:")
    print("1. Creating a panel with Z")
    print("2. Adding a character box for Z") 
    print("3. Generating to see if the embedding works")
    print("\nIf it works well, you can generate embeddings for all characters!")

if __name__ == "__main__":
    main()