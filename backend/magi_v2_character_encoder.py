#!/usr/bin/env python3
"""
Updated Magi v2 character encoder with proper normalization and directory structure.
Uses character_output/character_images/ as the main source directory.
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
from transformers import AutoModel
import torchvision.transforms as transforms

class MagiV2CharacterEncoder:
    def __init__(self, character_data_path, output_dir="character_output"):
        # Load character data
        with open(character_data_path, 'r', encoding='utf-8') as f:
            self.character_data = json.load(f)
            
        self.output_dir = Path(output_dir)
        
        # Updated paths - use character_images as main directory
        self.character_images_dir = self.output_dir / "character_images"
        self.embeddings_dir = self.output_dir / "character_embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to store the mapping between characters and their embeddings
        self.embeddings_map_path = self.embeddings_dir / "character_embeddings_map.json"
        
        # Dictionary to track character embeddings
        self.character_embeddings_map = {}
        
        # Load existing embeddings map if it exists
        if self.embeddings_map_path.exists():
            with open(self.embeddings_map_path, 'r') as f:
                self.character_embeddings_map = json.load(f)
                
        # Initialize Magi v2 model
        self.init_magi_v2_model()
        
    def init_magi_v2_model(self):
        """Initialize Magi v2 model for character encoding"""
        print("Loading Magi v2 model...")
        
        try:
            # Load Magi v2 model
            self.magi_model = AutoModel.from_pretrained(
                "ragavsachdeva/magiv2", 
                trust_remote_code=True
            )
            
            # Move to appropriate device
            self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            self.magi_model = self.magi_model.to(self.device)
            self.magi_model.eval()
            
            print("Magi v2 model loaded successfully")
            print(f"Crop embedding model type: {type(self.magi_model.crop_embedding_model)}")
            
        except Exception as e:
            print(f"Error loading Magi v2 model: {e}")
            self.magi_model = None
    
    def extract_magi_v2_embedding(self, image_path, character_name):
        """Extract character embedding using Magi v2's crop_embedding_model directly with normalization"""
        if self.magi_model is None:
            print(f"Magi model not loaded, using random normalized embedding for {character_name}")
            # Return normalized random embedding as fallback
            random_embedding = torch.randn(768).float()
            return random_embedding / torch.norm(random_embedding)
        
        try:
            # Load and preprocess the image for ViTMAE (224x224)
            image = Image.open(image_path).convert("RGB")
            
            # Preprocess for ViTMAE model (224x224)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            pixel_values = transform(image).unsqueeze(0).to(self.device)  # Add batch dimension
            
            print(f"Extracting Magi v2 embedding for {character_name}")
            print(f"Preprocessed image shape: {pixel_values.shape}")
            
            # Use Magi's crop_embedding_model directly
            crop_model = self.magi_model.crop_embedding_model
            
            with torch.no_grad():
                outputs = crop_model(pixel_values)
            
            # Extract embedding from ViTMAE outputs
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                # Use the first token (CLS token) as the character embedding
                embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
                embedding = embedding.squeeze(0)  # Remove batch dimension -> [768]
            else:
                print(f"Unexpected outputs from Magi crop model: {type(outputs)}")
                # Return normalized random embedding
                random_embedding = torch.randn(768).float()
                return random_embedding / torch.norm(random_embedding)
            
            # Move to CPU
            embedding = embedding.cpu().float()
            
            # Verify it's 768 dimensions
            if embedding.shape[0] != 768:
                print(f"Warning: Embedding has {embedding.shape[0]} dimensions, expected 768")
                if embedding.shape[0] > 768:
                    embedding = embedding[:768]
                else:
                    padding = torch.zeros(768 - embedding.shape[0])
                    embedding = torch.cat([embedding, padding])
            
            # NORMALIZE THE EMBEDDING - this is critical!
            embedding_norm = torch.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm
            else:
                print(f"Warning: Zero norm embedding for {character_name}, using random normalized embedding")
                embedding = torch.randn(768)
                embedding = embedding / torch.norm(embedding)
            
            print(f"Successfully extracted Magi v2 embedding for {character_name}: shape {embedding.shape}")
            print(f"Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
            print(f"Embedding norm: {torch.norm(embedding):.4f}")  # Should be 1.0
            return embedding
            
        except Exception as e:
            print(f"Error extracting Magi v2 embedding for {character_name}: {e}")
            import traceback
            traceback.print_exc()
            # Return normalized random embedding as fallback
            random_embedding = torch.randn(768).float()
            return random_embedding / torch.norm(random_embedding)
    
    def find_character_images(self):
        """Find all character images in the character_images directory"""
        if not self.character_images_dir.exists():
            print(f"Character images directory not found: {self.character_images_dir}")
            return []
        
        # Look for PNG files in the main character_images directory
        character_images = list(self.character_images_dir.glob("*.png"))
        
        # Also check the legacy keepers subdirectory for backwards compatibility
        keepers_dir = self.character_images_dir / "keepers"
        if keepers_dir.exists():
            keeper_images = list(keepers_dir.glob("*.png"))
            keeper_images = [img for img in keeper_images if not img.name.endswith("_card.png")]
            
            # Only add keepers images that don't exist in main directory
            for keeper_img in keeper_images:
                main_img_path = self.character_images_dir / keeper_img.name
                if not main_img_path.exists():
                    character_images.append(keeper_img)
                    print(f"Using keeper image: {keeper_img.name}")
        
        # Filter out card images
        character_images = [img for img in character_images if not img.name.endswith("_card.png")]
        
        return character_images
    
    def generate_all_character_embeddings(self, force_regenerate=False):
        """Generate Magi v2 embeddings for all character images"""
        character_images = self.find_character_images()
        
        if not character_images:
            print(f"No character images found in {self.character_images_dir}")
            print("Make sure your character images are in character_output/character_images/")
            return {}
        
        print(f"Found {len(character_images)} character images")
        
        # Create embeddings for each character image
        generated_count = 0
        skipped_count = 0
        
        for image_path in tqdm(character_images, desc="Generating Magi v2 embeddings"):
            character_name = image_path.stem  # Get filename without extension
            
            # Skip if embedding already exists and is up to date (unless force regenerate)
            embedding_path = self.embeddings_dir / f"{character_name}.pt"
            
            if not force_regenerate and embedding_path.exists() and image_path.stat().st_mtime < embedding_path.stat().st_mtime:
                print(f"Magi v2 embedding for {character_name} already exists and is up to date. Skipping.")
                skipped_count += 1
                
                # Make sure it's in the map
                if character_name not in self.character_embeddings_map:
                    self.character_embeddings_map[character_name] = {
                        "name": character_name,
                        "image_path": str(image_path),
                        "embedding_path": str(embedding_path),
                        "embedding_type": "magi_v2_768_normalized",
                        "features": {
                            "normalized": True,
                            "magi_v2": True,
                            "crop_embedding_model": True
                        }
                    }
                continue
                
            # Create the embedding
            print(f"Creating Magi v2 embedding for {character_name}")
            embedding = self.extract_magi_v2_embedding(image_path, character_name)
            
            # Save the embedding as PyTorch tensor
            torch.save(embedding, embedding_path)
            generated_count += 1
            
            # Add to the embeddings map with updated metadata
            self.character_embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path),
                "embedding_type": "magi_v2_768_normalized",
                "embedding_shape": list(embedding.shape),
                "features": {
                    "normalized": True,
                    "magi_v2": True,
                    "crop_embedding_model": True,
                    "embedding_norm": torch.norm(embedding).item()
                }
            }
        
        # Save the updated embeddings map
        with open(self.embeddings_map_path, 'w') as f:
            json.dump(self.character_embeddings_map, f, indent=2)
            
        print(f"\n📊 EMBEDDING GENERATION SUMMARY:")
        print(f"  Generated: {generated_count} new embeddings")
        print(f"  Skipped: {skipped_count} existing embeddings")
        print(f"  Total: {len(character_images)} character embeddings available")
        print(f"✅ Embeddings saved to: {self.embeddings_dir}")
        print(f"🗺️ Embeddings map saved to: {self.embeddings_map_path}")
        
        # Verify normalization
        self.verify_embeddings_normalization()
        
        return self.character_embeddings_map
    
    def verify_embeddings_normalization(self):
        """Verify that all embeddings are properly normalized"""
        print(f"\n🧪 VERIFYING EMBEDDING NORMALIZATION...")
        
        normalized_count = 0
        problem_count = 0
        
        for char_name, info in self.character_embeddings_map.items():
            try:
                embedding_path = info["embedding_path"]
                embedding = torch.load(embedding_path, map_location='cpu')
                norm = torch.norm(embedding).item()
                
                if abs(norm - 1.0) < 0.001:
                    normalized_count += 1
                else:
                    print(f"⚠️ {char_name}: norm = {norm:.4f} (should be 1.0)")
                    problem_count += 1
                    
            except Exception as e:
                print(f"❌ Error checking {char_name}: {e}")
                problem_count += 1
        
        print(f"✅ Properly normalized: {normalized_count}")
        print(f"⚠️ Problem embeddings: {problem_count}")
        
        if problem_count == 0:
            print("🎯 All embeddings are properly normalized!")
        else:
            print("🚨 Some embeddings need attention - consider regenerating with --force")
    
    def load_character_embedding(self, character_name):
        """Load a character embedding by name"""
        if character_name in self.character_embeddings_map:
            embedding_path = self.character_embeddings_map[character_name]["embedding_path"]
            return torch.load(embedding_path, map_location='cpu')
        else:
            print(f"No embedding found for character: {character_name}")
            return None
    
    def compare_embeddings(self, char1_name, char2_name):
        """Compare similarity between two character embeddings"""
        emb1 = self.load_character_embedding(char1_name)
        emb2 = self.load_character_embedding(char2_name)
        
        if emb1 is not None and emb2 is not None:
            # Calculate cosine similarity
            similarity = torch.cosine_similarity(emb1, emb2, dim=0)
            print(f"Similarity between {char1_name} and {char2_name}: {similarity.item():.4f}")
            
            # Also show norms for verification
            norm1 = torch.norm(emb1).item()
            norm2 = torch.norm(emb2).item()
            print(f"  {char1_name} norm: {norm1:.4f}")
            print(f"  {char2_name} norm: {norm2:.4f}")
            
            return similarity.item()
        else:
            print("Could not compare embeddings - one or both characters not found")
            return None
    
    def list_available_characters(self):
        """List all available characters with embeddings"""
        if not self.character_embeddings_map:
            print("No character embeddings found")
            return
        
        print(f"\n📋 AVAILABLE CHARACTER EMBEDDINGS ({len(self.character_embeddings_map)}):")
        print("=" * 50)
        
        for char_name, info in sorted(self.character_embeddings_map.items()):
            embedding_type = info.get("embedding_type", "unknown")
            is_normalized = info.get("features", {}).get("normalized", False)
            norm_status = "✅" if is_normalized else "⚠️"
            
            print(f"  {norm_status} {char_name} ({embedding_type})")

def main():
    parser = argparse.ArgumentParser(description="Generate character embeddings using Magi v2")
    parser.add_argument("--character_data", type=str, default="characters.json",
                        help="Path to character data JSON file")
    parser.add_argument("--output_dir", type=str, default="character_output",
                        help="Directory containing character images and for saving embeddings")
    parser.add_argument("--compare", nargs=2, metavar=('CHAR1', 'CHAR2'),
                        help="Compare embeddings between two characters")
    parser.add_argument("--list", action="store_true",
                        help="List all available character embeddings")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration of all embeddings")
    parser.add_argument("--verify", action="store_true",
                        help="Verify normalization of existing embeddings")
    
    args = parser.parse_args()
    
    encoder = MagiV2CharacterEncoder(
        character_data_path=args.character_data,
        output_dir=args.output_dir
    )
    
    if args.compare:
        char1, char2 = args.compare
        encoder.compare_embeddings(char1, char2)
    elif args.list:
        encoder.list_available_characters()
    elif args.verify:
        encoder.verify_embeddings_normalization()
    else:
        encoder.generate_all_character_embeddings(force_regenerate=args.force)

if __name__ == "__main__":
    main()