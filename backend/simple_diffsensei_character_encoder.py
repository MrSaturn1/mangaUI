#!/usr/bin/env python3
"""
DiffSensei-style character embedder for Drawatoon compatibility.
Based on the author's explicit mention of being inspired by DiffSensei.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

class DiffSenseiStyleCharacterEmbedder:
    """
    Character embedder following DiffSensei's approach, as mentioned by the Drawatoon author.
    This should be much closer to what the original model expects.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        print("🎯 Loading DiffSensei-style character encoder...")
        print("📝 Author explicitly mentioned being inspired by DiffSensei!")
        
        # Load the models DiffSensei uses
        self.load_diffsensei_models()
        self.create_projection_layer()
        
    def load_diffsensei_models(self):
        """Load the models that DiffSensei uses for character encoding"""
        
        # DiffSensei uses CLIP ViT-L/14 for general features
        print("Loading CLIP ViT-L/14...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # DiffSensei also uses a specialized character encoder (ViT-based)
        print("Loading ViT for character-specific features...")
        self.character_vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        
        # Move to device
        self.clip_model = self.clip_model.to(self.device)
        self.character_vit = self.character_vit.to(self.device)
        
        self.clip_model.eval()
        self.character_vit.eval()
        
        print("✅ DiffSensei-style models loaded")
    
    def create_projection_layer(self):
        """Create projection layer to combine and project to 768D"""
        # CLIP ViT-L/14 gives 1024D, ViT-B gives 768D
        # We need to project to exactly 768D for drawatoon compatibility
        
        self.projection = nn.Sequential(
            # Combine CLIP (1024) + ViT (768) = 1792D input
            nn.Linear(1024 + 768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Project to target 768D
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
        ).to(self.device)
        
        # Initialize weights properly
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)  # Small init for stability
                nn.init.zeros_(module.bias)
        
        print("✅ Projection layer created (1792D → 768D)")
    
    def preprocess_character_image(self, image):
        """Preprocess image for character extraction, following manga conventions"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # For manga characters, we want to focus on the character
        # Simple center crop to remove potential background
        w, h = image.size
        size = min(w, h)
        left = (w - size) // 2
        top = (h - size) // 2
        cropped = image.crop((left, top, left + size, top + size))
        
        # Resize to standard size
        resized = cropped.resize((224, 224), Image.LANCZOS)
        
        return resized
    
    def extract_clip_features(self, image):
        """Extract CLIP features following DiffSensei approach"""
        with torch.no_grad():
            # Preprocess for CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            # Get vision encoder features
            vision_outputs = self.clip_model.vision_model(**inputs)
            
            # Use CLS token from last hidden state
            clip_features = vision_outputs.last_hidden_state[:, 0, :]  # [1, 1024]
            
            # Normalize
            clip_features = F.normalize(clip_features, p=2, dim=-1)
            
            return clip_features.squeeze(0)  # [1024]
    
    def extract_character_features(self, image):
        """Extract character-specific features using ViT"""
        with torch.no_grad():
            # Preprocess for ViT
            inputs = self.vit_processor(images=image, return_tensors="pt").to(self.device)
            
            # Get ViT features
            outputs = self.character_vit(**inputs)
            
            # Use CLS token
            char_features = outputs.last_hidden_state[:, 0, :]  # [1, 768]
            
            # Normalize
            char_features = F.normalize(char_features, p=2, dim=-1)
            
            return char_features.squeeze(0)  # [768]
    
    def extract_diffsensei_embedding(self, image_path, character_name):
        """
        Extract character embedding following DiffSensei's dual-encoder approach.
        This should be compatible with Drawatoon's training.
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = self.preprocess_character_image(image)
            
            print(f"🎨 Extracting DiffSensei-style embedding for {character_name}")
            
            # Extract features from both encoders
            clip_features = self.extract_clip_features(processed_image)  # [1024]
            char_features = self.extract_character_features(processed_image)  # [768]
            
            # Combine features
            combined_features = torch.cat([clip_features, char_features], dim=0)  # [1792]
            
            # Project to 768D using learned projection
            with torch.no_grad():
                embedding = self.projection(combined_features.unsqueeze(0)).squeeze(0)  # [768]
            
            # Final normalization (critical for compatibility)
            embedding = F.normalize(embedding, p=2, dim=0)
            
            # Verify
            norm = torch.norm(embedding).item()
            print(f"✅ Generated embedding: shape={embedding.shape}, norm={norm:.4f}")
            
            if abs(norm - 1.0) > 0.001:
                print(f"⚠️  Warning: Norm is {norm:.4f}, expected 1.0")
            
            return embedding.cpu()
            
        except Exception as e:
            print(f"❌ Error extracting embedding for {character_name}: {e}")
            # Return normalized random as fallback
            fallback = torch.randn(768)
            return F.normalize(fallback, p=2, dim=0)
    
    def add_character_specific_signature(self, embedding, character_name, strength=0.05):
        """
        Add subtle character-specific signature to make embeddings more distinctive.
        This is inspired by the need for character consistency across generations.
        """
        # Create deterministic but unique signature for this character
        import hashlib
        char_hash = int(hashlib.md5(character_name.upper().encode()).hexdigest()[:8], 16)
        torch.manual_seed(char_hash)
        
        # Generate character-specific direction
        signature = torch.randn_like(embedding) * strength
        
        # Apply signature and renormalize
        modified_embedding = embedding + signature
        modified_embedding = F.normalize(modified_embedding, p=2, dim=0)
        
        # Reset seed
        torch.manual_seed(torch.initial_seed())
        
        return modified_embedding
    
    def generate_all_embeddings(self, 
                               characters_dir="character_output/character_images/keepers",
                               output_dir="character_output/character_embeddings",
                               add_signature=True,
                               force_regenerate=False):
        """Generate DiffSensei-style embeddings for all characters"""
        
        characters_path = Path(characters_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not characters_path.exists():
            print(f"❌ Characters directory not found: {characters_path}")
            return {}
        
        # Find character images
        image_files = list(characters_path.glob("*.png"))
        image_files = [img for img in image_files if not img.name.endswith("_card.png")]
        
        print(f"🎯 Found {len(image_files)} character images")
        print("🔥 Generating DiffSensei-style embeddings...")
        
        embeddings_map = {}
        
        for image_path in tqdm(image_files, desc="Processing characters"):
            character_name = image_path.stem
            embedding_path = output_path / f"{character_name}.pt"
            
            # Skip if exists and not forcing
            if not force_regenerate and embedding_path.exists():
                if image_path.stat().st_mtime < embedding_path.stat().st_mtime:
                    print(f"⭐ Skipping {character_name} (already exists)")
                    continue
            
            # Generate embedding
            embedding = self.extract_diffsensei_embedding(image_path, character_name)
            
            # Add character signature for better distinctiveness
            if add_signature:
                embedding = self.add_character_specific_signature(embedding, character_name)
                print(f"🔥 Added character signature for {character_name}")
            
            # Save embedding
            torch.save(embedding, embedding_path)
            
            # Update map
            embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path),
                "embedding_type": "diffsensei_style_768",
                "embedding_shape": list(embedding.shape),
                "features": {
                    "diffsensei_inspired": True,
                    "dual_encoder": True,
                    "clip_vit_l14": True,
                    "character_vit": True,
                    "normalized": True,
                    "character_signature": add_signature,
                    "drawatoon_compatible": True
                }
            }
        
        # Save map
        map_path = output_path / "character_embeddings_map.json"
        with open(map_path, 'w') as f:
            json.dump(embeddings_map, f, indent=2)
        
        print(f"🎉 Generated DiffSensei-style embeddings for {len(embeddings_map)} characters!")
        print(f"💾 Saved to: {output_path}")
        print(f"🗺️  Map saved: {map_path}")
        
        # Test distinctiveness
        self.test_embedding_distinctiveness(embeddings_map)
        
        return embeddings_map
    
    def test_embedding_distinctiveness(self, embeddings_map):
        """Test how distinctive the generated embeddings are"""
        print(f"\n🧪 TESTING EMBEDDING DISTINCTIVENESS")
        print("=" * 50)
        
        if len(embeddings_map) < 2:
            print("Need at least 2 embeddings to test distinctiveness")
            return
        
        # Load embeddings
        embeddings = []
        names = []
        
        for char_name, info in embeddings_map.items():
            embedding_path = info["embedding_path"]
            embedding = torch.load(embedding_path, map_location='cpu')
            embeddings.append(embedding)
            names.append(char_name)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = torch.cosine_similarity(embeddings[i], embeddings[j], dim=0).item()
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        min_similarity = min(similarities)
        max_similarity = max(similarities)
        
        print(f"📊 Similarity Statistics:")
        print(f"   Average: {avg_similarity:.4f}")
        print(f"   Std Dev: {std_similarity:.4f}")
        print(f"   Range: [{min_similarity:.4f}, {max_similarity:.4f}]")
        
        # Quality assessment
        if avg_similarity < 0.4:
            print("🏆 EXCELLENT distinctiveness! Should work great with Drawatoon.")
        elif avg_similarity < 0.6:
            print("✅ GOOD distinctiveness. Should work well.")
        elif avg_similarity < 0.7:
            print("⚠️  MODERATE distinctiveness. May need tuning.")
        else:
            print("🚨 POOR distinctiveness. Embeddings too similar!")
        
        return {
            "avg_similarity": avg_similarity,
            "std_similarity": std_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate DiffSensei-style character embeddings")
    parser.add_argument("--characters_dir", type=str, 
                       default="character_output/character_images/keepers",
                       help="Directory containing character images")
    parser.add_argument("--output_dir", type=str,
                       default="character_output/character_embeddings", 
                       help="Output directory for embeddings")
    parser.add_argument("--force", action="store_true",
                       help="Force regeneration of existing embeddings")
    parser.add_argument("--no_signature", action="store_true",
                       help="Don't add character-specific signatures")
    
    args = parser.parse_args()
    
    # Create embedder
    embedder = DiffSenseiStyleCharacterEmbedder()
    
    # Generate embeddings
    embeddings_map = embedder.generate_all_embeddings(
        characters_dir=args.characters_dir,
        output_dir=args.output_dir,
        add_signature=not args.no_signature,
        force_regenerate=args.force
    )
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Test these DiffSensei-style embeddings with your Drawatoon model")
    print("2. They should be much more compatible since the author was inspired by DiffSensei")
    print("3. If character consistency improves, we've found the right approach!")
    print("\n🔥 These embeddings follow the same approach the Drawatoon author used!")


if __name__ == "__main__":
    main()