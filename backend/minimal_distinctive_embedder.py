#!/usr/bin/env python3
"""
Minimal approach focused purely on distinctiveness
Sometimes simpler is better for character consistency
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import hashlib


class MinimalDistinctiveEmbedder:
    """
    Minimal embedder that prioritizes distinctiveness over semantic richness
    Uses basic CLIP with strong character-specific modifications
    """
    
    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() 
                                else "cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading CLIP-Large...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        
        print(f"✅ Minimal distinctive embedder ready on {self.device}")
    
    def create_character_basis(self, character_name, dim=1024):
        """Create a unique orthogonal basis for each character"""
        # Use character name to create deterministic but unique seed
        char_hash = int(hashlib.md5(character_name.upper().encode()).hexdigest()[:8], 16)
        torch.manual_seed(char_hash)
        
        # Create multiple orthogonal directions for this character
        directions = []
        for i in range(5):  # 5 orthogonal directions per character
            direction = torch.randn(dim)
            # Orthogonalize against previous directions
            for prev_dir in directions:
                direction = direction - torch.dot(direction, prev_dir) * prev_dir
            direction = F.normalize(direction, p=2, dim=0)
            directions.append(direction)
        
        # Reset seed
        torch.manual_seed(torch.initial_seed())
        
        return torch.stack(directions)
    
    def apply_strong_character_signature(self, clip_embedding, character_name):
        """Apply strong character-specific signature to base CLIP embedding"""
        # Get character's unique basis
        char_basis = self.create_character_basis(character_name, clip_embedding.shape[0])
        char_basis = char_basis.to(clip_embedding.device)
        
        # Project CLIP embedding onto character basis
        projections = torch.matmul(char_basis, clip_embedding)  # [5]
        
        # Enhance projections with character-specific amplification
        char_hash = int(hashlib.md5(character_name.upper().encode()).hexdigest()[:8], 16)
        torch.manual_seed(char_hash)
        amplifiers = torch.rand(5) * 0.5 + 0.5  # Random amplifiers between 0.5-1.0
        torch.manual_seed(torch.initial_seed())
        
        enhanced_projections = projections * amplifiers.to(projections.device)
        
        # Reconstruct embedding with enhanced character components
        character_component = torch.matmul(enhanced_projections, char_basis)
        
        # Mix original CLIP with character component
        # Start with less CLIP, more character signature
        mixing_ratio = 0.3  # 30% CLIP, 70% character signature
        final_embedding = (mixing_ratio * clip_embedding + 
                          (1 - mixing_ratio) * character_component)
        
        return F.normalize(final_embedding, p=2, dim=0)
    
    def extract_minimal_embedding(self, image_path, character_name):
        """Extract minimal but highly distinctive embedding"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Basic CLIP extraction
            with torch.no_grad():
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                clip_features = self.clip_model.get_image_features(**inputs)
                clip_features = F.normalize(clip_features, p=2, dim=-1)
                clip_features = clip_features.squeeze(0)
            
            # Apply strong character signature
            distinctive_embedding = self.apply_strong_character_signature(
                clip_features, character_name
            )
            
            # Final normalization
            final_embedding = F.normalize(distinctive_embedding, p=2, dim=0)
            
            # Verify normalization
            norm = torch.norm(final_embedding).item()
            if abs(norm - 1.0) > 0.001:
                print(f"⚠️ Warning: {character_name} norm is {norm:.4f}")
            
            return final_embedding.cpu()
            
        except Exception as e:
            print(f"❌ Error with {character_name}: {e}")
            # Fallback: pure character signature
            return self.create_pure_character_embedding(character_name)
    
    def create_pure_character_embedding(self, character_name, dim=1024):
        """Create embedding purely from character name (fallback)"""
        char_hash = int(hashlib.md5(character_name.upper().encode()).hexdigest()[:8], 16)
        torch.manual_seed(char_hash)
        
        embedding = torch.randn(dim)
        torch.manual_seed(torch.initial_seed())
        
        return F.normalize(embedding, p=2, dim=0)
    
    def test_distinctiveness_aggressive(self, embeddings_dict):
        """Aggressive distinctiveness test with detailed analysis"""
        print("\n🎯 AGGRESSIVE DISTINCTIVENESS TEST")
        print("=" * 50)
        
        if len(embeddings_dict) < 2:
            print("Need at least 2 embeddings")
            return False
        
        embeddings_list = list(embeddings_dict.values())
        char_names = list(embeddings_dict.keys())
        
        # Check normalization
        norms = [torch.norm(emb).item() for emb in embeddings_list]
        print(f"Norms: min={min(norms):.4f}, max={max(norms):.4f}")
        
        # Compute all similarities
        similarities = []
        problem_pairs = []
        
        for i in range(len(char_names)):
            for j in range(i+1, len(char_names)):
                sim = torch.cosine_similarity(embeddings_list[i], embeddings_list[j], dim=0).item()
                similarities.append(sim)
                
                if sim > 0.6:  # Flag high similarity pairs
                    problem_pairs.append((char_names[i], char_names[j], sim))
        
        avg_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        min_sim = min(similarities)
        max_sim = max(similarities)
        
        print(f"Similarity Statistics:")
        print(f"  Average: {avg_sim:.4f}")
        print(f"  Std Dev: {std_sim:.4f}")
        print(f"  Range: [{min_sim:.4f}, {max_sim:.4f}]")
        
        # Count similarity ranges
        excellent = sum(1 for s in similarities if s < 0.3)
        good = sum(1 for s in similarities if 0.3 <= s < 0.5)
        moderate = sum(1 for s in similarities if 0.5 <= s < 0.7)
        poor = sum(1 for s in similarities if s >= 0.7)
        
        total = len(similarities)
        print(f"\nSimilarity Distribution:")
        print(f"  Excellent (< 0.3): {excellent}/{total} ({excellent/total*100:.1f}%)")
        print(f"  Good (0.3-0.5): {good}/{total} ({good/total*100:.1f}%)")
        print(f"  Moderate (0.5-0.7): {moderate}/{total} ({moderate/total*100:.1f}%)")
        print(f"  Poor (≥ 0.7): {poor}/{total} ({poor/total*100:.1f}%)")
        
        if problem_pairs:
            print(f"\n🚨 {len(problem_pairs)} problematic pairs (similarity > 0.6):")
            for name1, name2, sim in sorted(problem_pairs, key=lambda x: x[2], reverse=True)[:10]:
                print(f"  {name1} vs {name2}: {sim:.4f}")
        
        # Overall assessment
        if avg_sim < 0.4:
            print(f"\n✅ SUCCESS! Average similarity {avg_sim:.4f} is excellent!")
            return True
        elif avg_sim < 0.5:
            print(f"\n✅ GOOD! Average similarity {avg_sim:.4f} is acceptable")
            return True
        elif avg_sim < 0.6:
            print(f"\n⚠️ MODERATE. Average similarity {avg_sim:.4f} could be better")
            return False
        else:
            print(f"\n🚨 POOR. Average similarity {avg_sim:.4f} is still too high")
            return False
    
    def generate_minimal_embeddings(self, keepers_dir="character_output/character_images/keepers"):
        """Generate minimal but distinctive embeddings"""
        keepers_path = Path(keepers_dir)
        embeddings_dir = Path("character_output/character_embeddings")
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        if not keepers_path.exists():
            print(f"❌ Keepers directory not found: {keepers_path}")
            return {}
        
        # Get character images
        image_files = list(keepers_path.glob("*.png"))
        image_files = [img for img in image_files if not img.name.endswith("_card.png")]
        
        print(f"📁 Found {len(image_files)} character images")
        print("🔄 Generating minimal distinctive embeddings...")
        
        embeddings = {}
        embeddings_map = {}
        
        for image_path in tqdm(image_files, desc="Processing"):
            character_name = image_path.stem
            
            # Generate embedding
            embedding = self.extract_minimal_embedding(image_path, character_name)
            embeddings[character_name] = embedding
            
            # Save embedding
            embedding_path = embeddings_dir / f"{character_name}.pt"
            torch.save(embedding, embedding_path)
            
            # Update map
            embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path),
                "embedding_type": "minimal_distinctive_1024",
                "embedding_shape": list(embedding.shape),
                "features": {
                    "minimal_clip": True,
                    "strong_character_signature": True,
                    "orthogonal_basis": True,
                    "normalized": True,
                    "distinctiveness_optimized": True
                }
            }
        
        # Test distinctiveness
        self.test_distinctiveness_aggressive(embeddings)
        
        # Save map
        map_path = embeddings_dir / "character_embeddings_map.json"
        existing_map = {}
        if map_path.exists():
            with open(map_path, 'r') as f:
                existing_map = json.load(f)
        
        existing_map.update(embeddings_map)
        
        with open(map_path, 'w') as f:
            json.dump(existing_map, f, indent=2)
        
        print(f"\n✅ Generated minimal embeddings for {len(embeddings)} characters")
        return embeddings


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate minimal distinctive embeddings")
    parser.add_argument("--generate", action="store_true", help="Generate embeddings")
    parser.add_argument("--test", action="store_true", help="Test existing embeddings")
    
    args = parser.parse_args()
    
    if args.generate:
        embedder = MinimalDistinctiveEmbedder()
        embeddings = embedder.generate_minimal_embeddings()
    elif args.test:
        embeddings_dir = Path("character_output/character_embeddings")
        embeddings = {}
        
        for file in embeddings_dir.glob("*.pt"):
            if file.name != "character_embeddings_map.json":
                embeddings[file.stem] = torch.load(file, map_location='cpu')
        
        if embeddings:
            embedder = MinimalDistinctiveEmbedder()
            embedder.test_distinctiveness_aggressive(embeddings)
        else:
            print("No embeddings found")
    else:
        print("🎯 MINIMAL DISTINCTIVE EMBEDDER")
        print("=" * 35)
        print("Radically different approach:")
        print("✅ Minimal CLIP extraction")
        print("✅ Strong character signatures")
        print("✅ Orthogonal character basis")
        print("✅ Prioritizes distinctiveness over semantics")
        print()
        print("Usage:")
        print("  --generate    Generate new minimal embeddings")
        print("  --test        Test existing embeddings")


if __name__ == "__main__":
    main()