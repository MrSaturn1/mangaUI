#!/usr/bin/env python3
"""
Basic CLIP embedder without any enhancement - for testing if our enhancements are the problem.
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import json
from pathlib import Path
import argparse
from tqdm import tqdm

class BasicCLIPEmbedder:
    """Pure CLIP embeddings without any enhancement to test if that's the issue"""
    
    def __init__(self, character_data_path="characters.json", output_dir="character_output"):
        with open(character_data_path, 'r') as f:
            self.character_data = json.load(f)
            
        self.output_dir = Path(output_dir)
        self.keepers_dir = self.output_dir / "character_images" / "keepers"
        self.embeddings_dir = self.output_dir / "character_embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.init_clip_model()
        
    def init_clip_model(self):
        """Initialize just basic CLIP - no enhancements"""
        print("Loading basic CLIP ViT-L/14...")
        
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            print("✅ Basic CLIP ViT-L/14 loaded successfully")
        except Exception as e:
            print(f"❌ Error loading CLIP ViT-L/14, trying ViT-B/32: {e}")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(self.device)
            print("⚠️ Using CLIP ViT-B/32 as fallback")
    
    def extract_basic_clip_embedding(self, image_path, character_name):
        """Extract pure CLIP embedding - no preprocessing, no enhancement"""
        try:
            # Just load image and process with CLIP
            image = Image.open(image_path).convert('RGB')
            
            # Standard CLIP processing only
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                
                # Normalize (this is standard for CLIP)
                normalized_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            embedding = normalized_features.squeeze(0).cpu().float()
            
            print(f"✅ Basic CLIP embedding for {character_name}: {embedding.shape}")
            print(f"   Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
            print(f"   Std: {embedding.std():.4f}")
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error extracting embedding for {character_name}: {e}")
            return torch.randn(768).float()
    
    def generate_all_basic_embeddings(self, force_regenerate=False):
        """Generate basic CLIP embeddings for all characters"""
        if not self.keepers_dir.exists():
            print(f"❌ Keepers directory not found: {self.keepers_dir}")
            return
        
        keeper_images = list(self.keepers_dir.glob("*.png"))
        keeper_images = [img for img in keeper_images if not img.name.endswith("_card.png")]
        
        print(f"📁 Found {len(keeper_images)} keeper images")
        print("🎯 Using PURE CLIP embeddings (no enhancement, no preprocessing)")
        
        embeddings_map = {}
        
        for image_path in tqdm(keeper_images, desc="Generating basic CLIP embeddings"):
            character_name = image_path.stem
            embedding_path = self.embeddings_dir / f"{character_name}.pt"
            
            if not force_regenerate and embedding_path.exists():
                print(f"⏭️  Basic embedding for {character_name} exists. Skipping (use --force to regenerate)")
                continue
            
            # Generate basic CLIP embedding
            embedding = self.extract_basic_clip_embedding(image_path, character_name)
            
            # Save as PyTorch tensor
            torch.save(embedding, embedding_path)
            
            embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path),
                "embedding_type": "basic_clip_768",
                "embedding_shape": list(embedding.shape),
                "features": {
                    "pure_clip": True,
                    "no_enhancement": True,
                    "no_preprocessing": True,
                    "model": "clip-vit-large-patch14"
                }
            }
        
        # Save/update embeddings map
        map_path = self.embeddings_dir / "character_embeddings_map.json"
        
        # Load existing map if it exists
        existing_map = {}
        if map_path.exists():
            with open(map_path, 'r') as f:
                existing_map = json.load(f)
        
        # Update with new embeddings
        existing_map.update(embeddings_map)
        
        with open(map_path, 'w') as f:
            json.dump(existing_map, f, indent=2)
        
        print(f"✅ Generated basic CLIP embeddings for {len(embeddings_map)} characters")
        print(f"💾 Saved to: {self.embeddings_dir}")
        print(f"🗺️ Updated map: {map_path}")
        
        return embeddings_map

def main():
    parser = argparse.ArgumentParser(description="Generate basic CLIP embeddings (no enhancement)")
    parser.add_argument("--character_data", type=str, default="characters.json",
                        help="Path to character data JSON file")
    parser.add_argument("--output_dir", type=str, default="character_output",
                        help="Directory containing character images and for saving embeddings")
    parser.add_argument("--force", action="store_true", 
                        help="Force regeneration of all embeddings")
    parser.add_argument("--character", type=str, 
                        help="Test with specific character only")
    
    args = parser.parse_args()
    
    embedder = BasicCLIPEmbedder(
        character_data_path=args.character_data,
        output_dir=args.output_dir
    )
    
    if args.character:
        # Test single character
        image_path = embedder.keepers_dir / f"{args.character}.png"
        if image_path.exists():
            embedding = embedder.extract_basic_clip_embedding(image_path, args.character)
            print(f"🧪 Test embedding shape: {embedding.shape}")
            
            # Save test embedding
            test_path = embedder.embeddings_dir / f"{args.character}.pt"
            torch.save(embedding, test_path)
            print(f"💾 Saved test embedding to: {test_path}")
        else:
            print(f"❌ Character image not found: {image_path}")
    else:
        # Generate all embeddings
        embedder.generate_all_basic_embeddings(force_regenerate=args.force)
        
        print(f"\n🎉 BASIC CLIP EMBEDDINGS GENERATED!")
        print(f"🧪 Test these immediately - they should give much better character consistency")
        print(f"📊 If these work better, we know the enhancement network was the problem")

if __name__ == "__main__":
    main()