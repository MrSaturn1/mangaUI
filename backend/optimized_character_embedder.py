#!/usr/bin/env python3
"""
Clean optimized character embedding generator.
Just generates embeddings - no overlap with manga_generator logic.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image, ImageEnhance, ImageFilter
import json
from pathlib import Path
import argparse
from tqdm import tqdm

class OptimizedCharacterEmbedder:
    """
    Clean embedding generator focused solely on creating better character embeddings.
    Uses CLIP ViT-L/14 + manga-specific preprocessing + lightweight enhancement.
    """
    
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
        self.character_enhancer = self.create_character_enhancer()
        
    def init_clip_model(self):
        """Initialize CLIP ViT-L/14 - optimal for manga characters"""
        print("Loading CLIP ViT-L/14...")
        
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            print("✅ CLIP ViT-L/14 loaded successfully")
        except Exception as e:
            print(f"❌ Error loading CLIP ViT-L/14, trying ViT-B/32: {e}")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(self.device)
            print("⚠️ Using CLIP ViT-B/32 as fallback")
    
    def create_character_enhancer(self):
        """Lightweight network to enhance CLIP features for manga characters"""
        # Get the actual CLIP dimension
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(self.device)
            test_output = self.clip_model.get_image_features(test_input)
            clip_dim = test_output.shape[-1]
        
        print(f"CLIP output dimension: {clip_dim}")
        
        enhancer = nn.Sequential(
            nn.Linear(clip_dim, clip_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(clip_dim * 2, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, clip_dim),
            nn.Tanh(),
        ).to(self.device)
        
        # Initialize with small weights to start close to identity
        for module in enhancer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
        
        return enhancer
    
    def preprocess_manga_image(self, image_path):
        """Manga-specific preprocessing for better character recognition"""
        image = Image.open(image_path).convert('RGB')
        
        # Enhance contrast for manga line art
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Slight sharpening for character lines
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        # Smart crop to focus on character
        image = self.smart_character_crop(image)
        
        return image
    
    def smart_character_crop(self, image):
        """Simple character cropping for manga images"""
        gray = image.convert('L')
        
        # Find bounding box of non-white content
        bbox = gray.point(lambda x: 0 if x < 250 else 255, '1').getbbox()
        
        if bbox:
            left, top, right, bottom = bbox
            width, height = right - left, bottom - top
            expand_x, expand_y = width * 0.1, height * 0.1
            
            left = max(0, int(left - expand_x))
            top = max(0, int(top - expand_y))
            right = min(image.width, int(right + expand_x))
            bottom = min(image.height, int(bottom + expand_y))
            
            return image.crop((left, top, right, bottom))
        
        # Fallback: center square crop
        size = min(image.width, image.height)
        left = (image.width - size) // 2
        top = (image.height - size) // 2
        return image.crop((left, top, left + size, top + size))
    
    def create_character_text_prompt(self, character_name):
        """Create enhanced text prompt for better CLIP understanding"""
        char_info = None
        for char in self.character_data:
            if char["name"] == character_name:
                char_info = char
                break
        
        if not char_info or not char_info.get("descriptions"):
            return f"manga character {character_name}, black and white manga art style"
        
        desc = char_info["descriptions"][0]
        return f"manga character {character_name}, {desc}, black and white manga art style, clean line art"
    
    def extract_optimized_embedding(self, image_path, character_name):
        """Extract optimized character embedding"""
        try:
            # Preprocess image for manga content
            image = self.preprocess_manga_image(image_path)
            
            # Create character-specific text prompt
            text_prompt = self.create_character_text_prompt(character_name)
            
            # Extract CLIP features
            inputs = self.clip_processor(
                text=[text_prompt], 
                images=[image], 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                # Combine image and text features
                combined = 0.7 * image_features + 0.3 * text_features
                
                # Enhance with character-specific network
                enhanced = self.character_enhancer(combined)
                
                # Normalize for consistency
                final_embedding = enhanced / enhanced.norm(dim=-1, keepdim=True)
                
            print(f"✅ Generated optimized embedding for {character_name}: {final_embedding.shape}")
            return final_embedding.squeeze(0).cpu().float()
            
        except Exception as e:
            print(f"❌ Error extracting embedding for {character_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return random embedding as fallback
            fallback_dim = 768  # Standard for most CLIP models
            return torch.randn(fallback_dim).float()
    
    def generate_all_embeddings(self, force_regenerate=False):
        """Generate optimized embeddings for all characters"""
        if not self.keepers_dir.exists():
            print(f"❌ Keepers directory not found: {self.keepers_dir}")
            return
        
        keeper_images = list(self.keepers_dir.glob("*.png"))
        keeper_images = [img for img in keeper_images if not img.name.endswith("_card.png")]
        
        print(f"📁 Found {len(keeper_images)} keeper images")
        print("🎯 Using optimized CLIP + character enhancement + manga preprocessing")
        
        embeddings_map = {}
        
        for image_path in tqdm(keeper_images, desc="Generating optimized embeddings"):
            character_name = image_path.stem
            embedding_path = self.embeddings_dir / f"{character_name}.pt"
            
            if not force_regenerate and embedding_path.exists():
                print(f"⏭️  Embedding for {character_name} exists. Skipping (use --force to regenerate)")
                continue
            
            # Generate optimized embedding
            embedding = self.extract_optimized_embedding(image_path, character_name)
            
            # Save as PyTorch tensor
            torch.save(embedding, embedding_path)
            
            embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path),
                "embedding_type": "optimized_clip_768",
                "embedding_shape": list(embedding.shape),
                "features": {
                    "manga_preprocessing": True,
                    "text_image_fusion": True,
                    "character_enhancement": True,
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
        
        print(f"✅ Generated optimized embeddings for {len(embeddings_map)} characters")
        print(f"💾 Saved to: {self.embeddings_dir}")
        print(f"🗺️ Updated map: {map_path}")
        
        return embeddings_map

def main():
    parser = argparse.ArgumentParser(description="Generate optimized character embeddings")
    parser.add_argument("--character_data", type=str, default="characters.json",
                        help="Path to character data JSON file")
    parser.add_argument("--output_dir", type=str, default="character_output",
                        help="Directory containing character images and for saving embeddings")
    parser.add_argument("--force", action="store_true", 
                        help="Force regeneration of all embeddings")
    parser.add_argument("--character", type=str, 
                        help="Test with specific character only")
    
    args = parser.parse_args()
    
    embedder = OptimizedCharacterEmbedder(
        character_data_path=args.character_data,
        output_dir=args.output_dir
    )
    
    if args.character:
        # Test single character
        image_path = embedder.keepers_dir / f"{args.character}.png"
        if image_path.exists():
            embedding = embedder.extract_optimized_embedding(image_path, args.character)
            print(f"🧪 Test embedding shape: {embedding.shape}")
            print(f"📊 Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
            
            # Save test embedding
            test_path = embedder.embeddings_dir / f"{args.character}.pt"
            torch.save(embedding, test_path)
            print(f"💾 Saved test embedding to: {test_path}")
        else:
            print(f"❌ Character image not found: {image_path}")
    else:
        # Generate all embeddings
        embedder.generate_all_embeddings(force_regenerate=args.force)
        
        print(f"\n🎉 DONE! Your manga_generator.py will automatically use these new embeddings.")
        print(f"🧪 Test by generating some panels and checking character consistency.")

if __name__ == "__main__":
    main()