#!/usr/bin/env python3
"""
Diagnostic script to debug character embedding issues.
Let's figure out what's wrong with our embedding generation.
"""

import torch
import numpy as np
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt

def diagnose_embeddings():
    """Run comprehensive diagnostics on character embeddings"""
    
    embeddings_dir = Path("character_output/character_embeddings")
    embeddings_map_path = embeddings_dir / "character_embeddings_map.json"
    
    print("🔍 DIAGNOSING CHARACTER EMBEDDINGS")
    print("=" * 50)
    
    # 1. Check if embeddings exist and load map
    if not embeddings_map_path.exists():
        print("❌ No embeddings map found!")
        return
    
    with open(embeddings_map_path, 'r') as f:
        embeddings_map = json.load(f)
    
    print(f"📊 Found {len(embeddings_map)} character embeddings")
    
    # 2. Load and analyze embeddings
    embeddings = {}
    for char_name, info in embeddings_map.items():
        embedding_path = info["embedding_path"]
        if Path(embedding_path).exists():
            try:
                embedding = torch.load(embedding_path, map_location='cpu')
                embeddings[char_name] = embedding
                print(f"✅ Loaded {char_name}: {embedding.shape}, dtype: {embedding.dtype}")
            except Exception as e:
                print(f"❌ Failed to load {char_name}: {e}")
        else:
            print(f"❌ Missing file for {char_name}: {embedding_path}")
    
    if not embeddings:
        print("❌ No embeddings could be loaded!")
        return
    
    # 3. Analyze embedding properties
    print(f"\n📈 EMBEDDING ANALYSIS")
    print("-" * 30)
    
    all_embeddings = torch.stack(list(embeddings.values()))
    
    print(f"Shape: {all_embeddings.shape}")
    print(f"Min value: {all_embeddings.min():.4f}")
    print(f"Max value: {all_embeddings.max():.4f}")
    print(f"Mean: {all_embeddings.mean():.4f}")
    print(f"Std: {all_embeddings.std():.4f}")
    
    # 4. Check for problematic patterns
    print(f"\n🚨 PROBLEM DETECTION")
    print("-" * 30)
    
    # Check if embeddings are too similar (indicating poor distinctiveness)
    similarities = []
    char_names = list(embeddings.keys())
    for i in range(len(char_names)):
        for j in range(i+1, len(char_names)):
            sim = torch.cosine_similarity(embeddings[char_names[i]], embeddings[char_names[j]], dim=0)
            similarities.append(sim.item())
    
    avg_similarity = np.mean(similarities)
    print(f"Average pairwise similarity: {avg_similarity:.4f}")
    
    if avg_similarity > 0.9:
        print("🚨 PROBLEM: Embeddings are too similar! Characters won't be distinguishable.")
    elif avg_similarity > 0.7:
        print("⚠️  WARNING: Embeddings might not be distinctive enough.")
    else:
        print("✅ Similarity levels look reasonable.")
    
    # Check for zero/nan values
    zero_count = (all_embeddings == 0).sum().item()
    nan_count = torch.isnan(all_embeddings).sum().item()
    
    if zero_count > 0:
        print(f"⚠️  Found {zero_count} zero values in embeddings")
    if nan_count > 0:
        print(f"🚨 PROBLEM: Found {nan_count} NaN values in embeddings!")
    
    # Check if embeddings are normalized
    norms = torch.norm(all_embeddings, dim=1)
    print(f"Embedding norms - min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}")
    
    if (norms - 1.0).abs().max() > 0.1:
        print("⚠️  Embeddings don't appear to be properly normalized")
    
    # 5. Test specific character consistency
    print(f"\n🎯 CHARACTER-SPECIFIC ANALYSIS")
    print("-" * 35)
    
    # Find the character you're testing with
    test_chars = ["Z", "ANTON", "CATHERINE"]  # Add your test characters here
    
    for char_name in test_chars:
        if char_name in embeddings:
            emb = embeddings[char_name]
            print(f"\n{char_name}:")
            print(f"  Shape: {emb.shape}")
            print(f"  Range: [{emb.min():.4f}, {emb.max():.4f}]")
            print(f"  Norm: {torch.norm(emb):.4f}")
            print(f"  Non-zero elements: {(emb != 0).sum().item()}/{emb.numel()}")
            
            # Check if it's just random noise
            if emb.std() < 0.1:
                print(f"  🚨 PROBLEM: Very low variance - might be random/invalid!")
    
    # 6. Compare embedding types
    print(f"\n🏷️  EMBEDDING TYPES")
    print("-" * 20)
    
    embedding_types = {}
    for char_name, info in embeddings_map.items():
        emb_type = info.get("embedding_type", "unknown")
        if emb_type not in embedding_types:
            embedding_types[emb_type] = []
        embedding_types[emb_type].append(char_name)
    
    for emb_type, chars in embedding_types.items():
        print(f"{emb_type}: {len(chars)} characters")
    
    return embeddings, embeddings_map

def test_single_character_embedding(character_name="Z"):
    """Test embedding generation for a single character step by step"""
    
    print(f"\n🧪 TESTING SINGLE CHARACTER: {character_name}")
    print("=" * 50)
    
    # Check if image exists
    keepers_dir = Path("character_output/character_images/keepers")
    image_path = keepers_dir / f"{character_name}.png"
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"✅ Found image: {image_path}")
    
    # Test with your optimized embedder
    try:
        from optimized_character_embedder import OptimizedCharacterEmbedder
        
        embedder = OptimizedCharacterEmbedder()
        embedding = embedder.extract_optimized_embedding(image_path, character_name)
        
        print(f"✅ Generated embedding: {embedding.shape}")
        print(f"   Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        print(f"   Mean: {embedding.mean():.4f}")
        print(f"   Std: {embedding.std():.4f}")
        print(f"   Norm: {torch.norm(embedding):.4f}")
        
        # Test if it's different from random
        random_emb = torch.randn_like(embedding)
        similarity_to_random = torch.cosine_similarity(embedding, random_emb, dim=0)
        print(f"   Similarity to random: {similarity_to_random:.4f}")
        
        if similarity_to_random > 0.3:
            print("🚨 PROBLEM: Embedding is too similar to random noise!")
        
        return embedding
        
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_clip_baseline(character_name="Z"):
    """Compare our embedding with basic CLIP to see if we're making it worse"""
    
    print(f"\n⚖️  COMPARING WITH CLIP BASELINE: {character_name}")
    print("=" * 50)
    
    keepers_dir = Path("character_output/character_images/keepers")
    image_path = keepers_dir / f"{character_name}.png"
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    try:
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image
        
        # Load basic CLIP
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = clip_model.to(device)
        
        # Process image with basic CLIP
        image = Image.open(image_path).convert('RGB')
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            basic_embedding = clip_model.get_image_features(**inputs)
            basic_embedding = basic_embedding / basic_embedding.norm(dim=-1, keepdim=True)
            basic_embedding = basic_embedding.squeeze(0).cpu()
        
        print(f"✅ Basic CLIP embedding: {basic_embedding.shape}")
        print(f"   Range: [{basic_embedding.min():.4f}, {basic_embedding.max():.4f}]")
        print(f"   Std: {basic_embedding.std():.4f}")
        
        # Now test our optimized version
        from optimized_character_embedder import OptimizedCharacterEmbedder
        embedder = OptimizedCharacterEmbedder()
        our_embedding = embedder.extract_optimized_embedding(image_path, character_name)
        
        # Compare them
        similarity = torch.cosine_similarity(basic_embedding, our_embedding, dim=0)
        print(f"\n📊 Similarity between basic CLIP and our embedding: {similarity:.4f}")
        
        if similarity < 0.5:
            print("🚨 PROBLEM: Our embedding is very different from basic CLIP - we might be making it worse!")
        elif similarity > 0.95:
            print("⚠️  Our embedding is almost identical to basic CLIP - our enhancements might not be working")
        else:
            print("✅ Reasonable difference - our enhancements are working but not destroying the base features")
        
        return basic_embedding, our_embedding
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose character embedding issues")
    parser.add_argument("--character", type=str, default="Z", help="Character to test specifically")
    
    args = parser.parse_args()
    
    # Run full diagnostics
    embeddings, embeddings_map = diagnose_embeddings()
    
    # Test single character
    test_embedding = test_single_character_embedding(args.character)
    
    # Compare with CLIP baseline
    basic_emb, our_emb = compare_with_clip_baseline(args.character)
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("=" * 30)
    print("1. Check the embedding analysis above for any red flags")
    print("2. If embeddings are too similar, the enhancement network might be collapsing them")
    print("3. If very different from CLIP baseline, try reducing enhancement strength")
    print("4. Consider testing with just basic CLIP embeddings (no enhancement) first")