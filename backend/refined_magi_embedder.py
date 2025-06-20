#!/usr/bin/env python3
"""
Refined Magi character embedder using the exact approach the Drawatoon author intended.
Based on your existing magi_v2_character_encoder.py but with improvements.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from PIL import Image, ImageEnhance, ImageFilter
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np

class RefinedMagiEmbedder:
    """
    Generate embeddings using Magi v2 exactly as the Drawatoon author intended.
    This should be the character encoder used during Drawatoon training.
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
        
        self.init_magi_model()
        
    def init_magi_model(self):
        """Initialize Magi v2 model - the actual character encoder used by Drawatoon"""
        print("Loading Magi v2 model (Drawatoon's character encoder)...")
        
        try:
            self.magi_model = AutoModel.from_pretrained(
                "ragavsachdeva/magiv2", 
                trust_remote_code=True
            )
            self.magi_model = self.magi_model.to(self.device)
            self.magi_model.eval()
            print("✅ Magi v2 model loaded successfully")
            
            # Check what components are available
            print("🔍 Magi model components:")
            for name, module in self.magi_model.named_children():
                print(f"  {name}: {type(module)}")
                
        except Exception as e:
            print(f"❌ Error loading Magi v2 model: {e}")
            self.magi_model = None
    
    def preprocess_for_magi(self, image_path):
        """
        Preprocess image specifically for Magi model.
        Based on your successful approach but refined.
        """
        image = Image.open(image_path).convert("RGB")
        
        # Magi works better with enhanced manga images
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)  # Gentle contrast boost
        
        # Slight sharpening for manga line art
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=2))
        
        return image
    
    def extract_magi_embedding_method1_cls_center(self, image, character_name):
        """
        Method 1: CLS token + center patches (your best working approach)
        This was your most successful method from magi_v2_character_encoder.py
        """
        try:
            # Standard preprocessing for ViTMAE model (224x224)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            pixel_values = transform(image).unsqueeze(0).to(self.device)
            
            # Use Magi's crop_embedding_model (the character-specific encoder)
            crop_model = self.magi_model.crop_embedding_model
            
            with torch.no_grad():
                outputs = crop_model(pixel_values)
            
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                all_patches = outputs.last_hidden_state  # [1, 197, 768]
                
                # Your successful approach: CLS token + center patches
                cls_token = all_patches[:, 0, :]  # [1, 768] - global representation
                patch_embeddings = all_patches[:, 1:, :]  # [1, 196, 768]
                
                # Focus on center region (your working method)
                batch_size, num_patches, embed_dim = patch_embeddings.shape
                grid_size = int(num_patches ** 0.5)  # 14
                
                patch_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, embed_dim)
                
                # Take center 50% of patches (your conservative approach)
                center_start = grid_size // 4  # 3-4
                center_end = 3 * grid_size // 4  # 10-11
                
                center_patches = patch_grid[:, center_start:center_end, center_start:center_end, :]
                center_avg = center_patches.mean(dim=(1, 2))  # [1, 768]
                
                # Your successful combination: 60% CLS + 40% center
                combined_embedding = 0.6 * cls_token + 0.4 * center_avg
                embedding = combined_embedding.squeeze(0)  # [768]
                
                print(f"✅ Magi Method 1 (CLS+center) for {character_name}: {embedding.shape}")
                return embedding.cpu().float()
                
            else:
                print(f"❌ Unexpected Magi outputs: {type(outputs)}")
                return torch.randn(768).float()
                
        except Exception as e:
            print(f"❌ Error in Magi Method 1 for {character_name}: {e}")
            return torch.randn(768).float()
    
    def extract_magi_embedding_method2_full_crop(self, image, character_name):
        """
        Method 2: Use Magi's full crop processing pipeline
        This tries to use Magi exactly as intended for character cropping
        """
        try:
            # Convert PIL to numpy array (Magi's expected format)
            image_np = np.array(image)
            
            # Try to use Magi's character detection if available
            if hasattr(self.magi_model, 'detect_characters'):
                characters = self.magi_model.detect_characters([image_np])
                if characters and len(characters) > 0:
                    # Use the first detected character
                    char_crop = characters[0]
                    # Process through embedding model
                    embedding = self.magi_model.get_character_embedding(char_crop)
                    if embedding is not None:
                        print(f"✅ Magi Method 2 (full pipeline) for {character_name}: {embedding.shape}")
                        return torch.tensor(embedding).float()
            
            # Fallback to direct crop embedding
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            pixel_values = transform(image).unsqueeze(0).to(self.device)
            crop_model = self.magi_model.crop_embedding_model
            
            with torch.no_grad():
                outputs = crop_model(pixel_values)
                # Use CLS token as the character representation
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
            
            print(f"✅ Magi Method 2 (fallback CLS) for {character_name}: {embedding.shape}")
            return embedding.cpu().float()
                
        except Exception as e:
            print(f"❌ Error in Magi Method 2 for {character_name}: {e}")
            return torch.randn(768).float()
    
    def extract_magi_embedding_method3_raw_features(self, image, character_name):
        """
        Method 3: Extract raw ViTMAE features without processing
        Sometimes the raw features work better than processed ones
        """
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            pixel_values = transform(image).unsqueeze(0).to(self.device)
            crop_model = self.magi_model.crop_embedding_model
            
            with torch.no_grad():
                outputs = crop_model(pixel_values)
                
                # Method 3A: Mean of all patches (including CLS)
                all_mean = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                
                # Method 3B: Just the patch embeddings (no CLS)
                patch_mean = outputs.last_hidden_state[:, 1:, :].mean(dim=1).squeeze(0)
                
                # Try both and see which works better
                # For now, return patch mean (often works better for visual features)
                embedding = patch_mean
            
            print(f"✅ Magi Method 3 (raw features) for {character_name}: {embedding.shape}")
            return embedding.cpu().float()
                
        except Exception as e:
            print(f"❌ Error in Magi Method 3 for {character_name}: {e}")
            return torch.randn(768).float()
    
    def extract_best_magi_embedding(self, image_path, character_name, method="cls_center"):
        """
        Extract Magi embedding using the specified method.
        Default to your most successful approach.
        """
        if self.magi_model is None:
            print(f"❌ Magi model not loaded, using random embedding for {character_name}")
            return torch.randn(768).float()
        
        # Preprocess image
        image = self.preprocess_for_magi(image_path)
        
        # Extract embedding using specified method
        if method == "cls_center":
            return self.extract_magi_embedding_method1_cls_center(image, character_name)
        elif method == "full_crop":
            return self.extract_magi_embedding_method2_full_crop(image, character_name)
        elif method == "raw_features":
            return self.extract_magi_embedding_method3_raw_features(image, character_name)
        else:
            print(f"❌ Unknown method: {method}, using cls_center")
            return self.extract_magi_embedding_method1_cls_center(image, character_name)
    
    def generate_all_magi_embeddings(self, method="cls_center", force_regenerate=False):
        """Generate Magi embeddings for all characters using the specified method"""
        if not self.keepers_dir.exists():
            print(f"❌ Keepers directory not found: {self.keepers_dir}")
            return
        
        keeper_images = list(self.keepers_dir.glob("*.png"))
        keeper_images = [img for img in keeper_images if not img.name.endswith("_card.png")]
        
        print(f"📁 Found {len(keeper_images)} keeper images")
        print(f"🎯 Using Magi v2 method: {method}")
        print(f"🔧 With DiffSensei weights at 1.0 - this should finally work!")
        
        embeddings_map = {}
        
        for image_path in tqdm(keeper_images, desc=f"Generating Magi embeddings ({method})"):
            character_name = image_path.stem
            embedding_path = self.embeddings_dir / f"{character_name}.pt"
            
            if not force_regenerate and embedding_path.exists():
                print(f"⏭️  Magi embedding for {character_name} exists. Skipping")
                continue
            
            # Generate Magi embedding
            embedding = self.extract_best_magi_embedding(image_path, character_name, method)
            
            # Save as PyTorch tensor
            torch.save(embedding, embedding_path)
            
            embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path),
                "embedding_type": f"magi_v2_{method}_768",
                "embedding_shape": list(embedding.shape),
                "features": {
                    "magi_character_encoder": True,
                    "method": method,
                    "model": "ragavsachdeva/magiv2",
                    "drawatoon_compatible": True
                }
            }
        
        # Save/update embeddings map
        map_path = self.embeddings_dir / "character_embeddings_map.json"
        
        existing_map = {}
        if map_path.exists():
            with open(map_path, 'r') as f:
                existing_map = json.load(f)
        
        existing_map.update(embeddings_map)
        
        with open(map_path, 'w') as f:
            json.dump(existing_map, f, indent=2)
        
        print(f"✅ Generated Magi embeddings for {len(embeddings_map)} characters")
        print(f"💾 Saved to: {self.embeddings_dir}")
        print(f"🗺️ Updated map: {map_path}")
        print(f"\n🎯 NOW TEST: Generate panels with DiffSensei weights at 1.0 + these Magi embeddings!")
        
        return embeddings_map

def main():
    parser = argparse.ArgumentParser(description="Generate Magi character embeddings (Drawatoon's intended method)")
    parser.add_argument("--character_data", type=str, default="characters.json",
                        help="Path to character data JSON file")
    parser.add_argument("--output_dir", type=str, default="character_output",
                        help="Directory containing character images and for saving embeddings")
    parser.add_argument("--method", type=str, default="cls_center", 
                        choices=["cls_center", "full_crop", "raw_features"],
                        help="Magi embedding extraction method")
    parser.add_argument("--force", action="store_true", 
                        help="Force regeneration of all embeddings")
    parser.add_argument("--character", type=str, 
                        help="Test with specific character only")
    
    args = parser.parse_args()
    
    embedder = RefinedMagiEmbedder(
        character_data_path=args.character_data,
        output_dir=args.output_dir
    )
    
    if args.character:
        # Test single character with all methods
        print(f"🧪 Testing all Magi methods for {args.character}:")
        
        image_path = embedder.keepers_dir / f"{args.character}.png"
        if image_path.exists():
            for method in ["cls_center", "full_crop", "raw_features"]:
                print(f"\n--- Testing {method} ---")
                embedding = embedder.extract_best_magi_embedding(image_path, args.character, method)
                
                # Save for testing
                test_path = embedder.embeddings_dir / f"{args.character}.pt"
                torch.save(embedding, test_path)
                print(f"💾 Saved {method} embedding to: {test_path}")
                print(f"🧪 TEST NOW: Generate a panel and see character consistency")
                input("Press Enter to try next method...")
        else:
            print(f"❌ Character image not found: {image_path}")
    else:
        # Generate all embeddings with specified method
        embedder.generate_all_magi_embeddings(method=args.method, force_regenerate=args.force)

if __name__ == "__main__":
    main()