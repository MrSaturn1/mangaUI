#!/usr/bin/env python3
"""
Normalize the Magi embeddings - this might be the missing piece!
The embeddings look good but aren't normalized, which could break the IP adapter.
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm

def normalize_all_embeddings():
    """Normalize all existing Magi embeddings to unit length"""
    
    embeddings_dir = Path("character_output/character_embeddings")
    embeddings_map_path = embeddings_dir / "character_embeddings_map.json"
    
    print("🔧 NORMALIZING MAGI EMBEDDINGS")
    print("=" * 40)
    
    # Load embeddings map
    if not embeddings_map_path.exists():
        print("❌ No embeddings map found!")
        return
    
    with open(embeddings_map_path, 'r') as f:
        embeddings_map = json.load(f)
    
    # Filter for Magi embeddings
    magi_embeddings = {k: v for k, v in embeddings_map.items() 
                      if v.get("embedding_type", "").startswith("magi_v2")}
    
    print(f"📊 Found {len(magi_embeddings)} Magi embeddings to normalize")
    
    if len(magi_embeddings) == 0:
        print("❌ No Magi embeddings found!")
        return
    
    # Normalize each embedding
    for char_name, info in tqdm(magi_embeddings.items(), desc="Normalizing embeddings"):
        embedding_path = info["embedding_path"]
        
        try:
            # Load embedding
            embedding = torch.load(embedding_path, map_location='cpu')
            
            # Check current norm
            original_norm = torch.norm(embedding).item()
            
            # Normalize to unit length
            normalized_embedding = embedding / torch.norm(embedding)
            new_norm = torch.norm(normalized_embedding).item()
            
            # Save normalized embedding
            torch.save(normalized_embedding, embedding_path)
            
            print(f"✅ {char_name}: norm {original_norm:.4f} → {new_norm:.4f}")
            
            # Update embeddings map to indicate normalization
            embeddings_map[char_name]["features"] = embeddings_map[char_name].get("features", {})
            embeddings_map[char_name]["features"]["normalized"] = True
            embeddings_map[char_name]["features"]["original_norm"] = original_norm
            
        except Exception as e:
            print(f"❌ Error normalizing {char_name}: {e}")
    
    # Save updated embeddings map
    with open(embeddings_map_path, 'w') as f:
        json.dump(embeddings_map, f, indent=2)
    
    print(f"\n✅ NORMALIZATION COMPLETE!")
    print(f"🎯 Now test character consistency - normalized embeddings should work much better!")
    print(f"📊 All embeddings now have norm=1.0 as expected by the model")

def test_single_embedding_normalization(character_name="Z"):
    """Test normalization on a single character for verification"""
    
    embeddings_dir = Path("character_output/character_embeddings")
    embedding_path = embeddings_dir / f"{character_name}.pt"
    
    if not embedding_path.exists():
        print(f"❌ Embedding not found for {character_name}")
        return
    
    print(f"🧪 TESTING NORMALIZATION FOR {character_name}")
    print("=" * 40)
    
    # Load embedding
    embedding = torch.load(embedding_path, map_location='cpu')
    
    print(f"Original embedding:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    print(f"  Norm: {torch.norm(embedding):.4f}")
    print(f"  Mean: {embedding.mean():.4f}")
    print(f"  Std: {embedding.std():.4f}")
    
    # Normalize
    normalized = embedding / torch.norm(embedding)
    
    print(f"\nNormalized embedding:")
    print(f"  Shape: {normalized.shape}")
    print(f"  Range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    print(f"  Norm: {torch.norm(normalized):.4f}")
    print(f"  Mean: {normalized.mean():.4f}")
    print(f"  Std: {normalized.std():.4f}")
    
    # Check if features are preserved
    similarity = torch.cosine_similarity(embedding, normalized, dim=0)
    print(f"  Similarity to original: {similarity:.6f}")
    
    return normalized

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalize Magi embeddings")
    parser.add_argument("--test", type=str, help="Test normalization on specific character")
    parser.add_argument("--normalize_all", action="store_true", help="Normalize all embeddings")
    
    args = parser.parse_args()
    
    if args.test:
        test_single_embedding_normalization(args.test)
    elif args.normalize_all:
        normalize_all_embeddings()
    else:
        print("🎯 MAGI EMBEDDING NORMALIZATION")
        print("=" * 35)
        print("The diagnostics show your Magi embeddings have good features")
        print("but aren't normalized (norm=5.3 instead of 1.0)")
        print()
        print("This is likely why character consistency isn't working!")
        print("Most neural networks expect normalized embeddings.")
        print()
        print("Options:")
        print("  --test Z              Test normalization on character Z")  
        print("  --normalize_all       Normalize all Magi embeddings")
        print()
        print("Recommended: Run --normalize_all then test character consistency")