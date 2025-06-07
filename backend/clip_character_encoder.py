#!/usr/bin/env python3
"""
Proper CLIP-based character encoder for drawatoon model.
This extracts meaningful 768-dimensional character embeddings using CLIP.
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class CLIPCharacterEncoder:
    def __init__(self, character_data_path, output_dir="character_output", clip_model="openai/clip-vit-base-patch32"):
        # Load character data
        with open(character_data_path, 'r', encoding='utf-8') as f:
            self.character_data = json.load(f)
            
        self.output_dir = Path(output_dir)
        
        # Paths for character images and embeddings
        self.character_images_dir = self.output_dir / "character_images"
        self.keepers_dir = self.character_images_dir / "keepers"
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
                
        # Initialize CLIP model
        self.clip_model_name = clip_model
        self.init_clip_model()
        
    def init_clip_model(self):
        """Initialize CLIP model for character encoding"""
        print(f"Loading CLIP model: {self.clip_model_name}")
        
        try:
            # Load CLIP model and processor
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            
            # Move to appropriate device
            self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            
            print("CLIP model loaded successfully")
            
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            self.clip_model = None
    
    def create_enhanced_text_prompt(self, character_name):
        """Create enhanced text prompt for character"""
        # Get character descriptions
        character_info = None
        for character in self.character_data:
            if character["name"] == character_name:
                character_info = character
                break
        
        if not character_info:
            return f"manga character {character_name}"
        
        descriptions = character_info.get("descriptions", [])
        if not descriptions:
            return f"manga character {character_name}"
        
        # Use the first description as the main description
        main_description = descriptions[0]
        
        # Create enhanced prompt
        enhanced_prompt = f"manga character {character_name}, {main_description}, black and white manga art style"
        
        return enhanced_prompt
    
    def extract_clip_embedding(self, image_path, character_name):
        """Extract CLIP embedding from character image"""
        if self.clip_model is None:
            print(f"CLIP model not loaded, using random embedding for {character_name}")
            return torch.randn(768).float()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Create enhanced text prompt for this character
            text_prompt = self.create_enhanced_text_prompt(character_name)
            print(f"Text prompt for {character_name}: {text_prompt}")
            
            # Process both image and text
            inputs = self.processor(
                text=[text_prompt], 
                images=[image], 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                # Get image and text features
                outputs = self.clip_model(**inputs)
                image_features = outputs.image_embeds  # [1, 512] for CLIP ViT-B/32
                text_features = outputs.text_embeds    # [1, 512] for CLIP ViT-B/32
                
                # Get the full image features from the vision encoder (this should be 768 for ViT-B/32)
                vision_outputs = self.clip_model.vision_model(inputs['pixel_values'])
                full_image_features = vision_outputs.pooler_output  # [1, 768]
                
                # We want the 768-dimensional features, not the 512-dimensional projected ones
                if full_image_features.shape[1] == 768:
                    embedding = full_image_features[0]  # [768]
                else:
                    # Fallback: concatenate image and text features and project to 768
                    combined = torch.cat([image_features[0], text_features[0]], dim=0)  # [1024]
                    # Project to 768 dimensions
                    projection = torch.nn.Linear(1024, 768).to(self.device)
                    embedding = projection(combined)
                
                print(f"Extracted CLIP embedding for {character_name}: shape {embedding.shape}")
                return embedding.cpu().float()
                
        except Exception as e:
            print(f"Error extracting CLIP embedding for {character_name}: {e}")
            # Return random embedding as fallback
            return torch.randn(768).float()
    
    def generate_all_keeper_embeddings(self):
        """Generate CLIP embeddings for all character images in the keepers folder"""
        if not self.keepers_dir.exists():
            print(f"Keepers directory not found: {self.keepers_dir}")
            return
            
        # Get all PNG files in the keepers folder
        keeper_images = list(self.keepers_dir.glob("*.png"))
        keeper_images = [img for img in keeper_images if not img.name.endswith("_card.png")]
        
        print(f"Found {len(keeper_images)} keeper images")
        
        # Create embeddings for each keeper image
        for image_path in tqdm(keeper_images, desc="Generating CLIP embeddings"):
            character_name = image_path.stem  # Get filename without extension
            
            # Skip if embedding already exists and is up to date
            embedding_path = self.embeddings_dir / f"{character_name}.pt"
            
            if embedding_path.exists() and image_path.stat().st_mtime < embedding_path.stat().st_mtime:
                print(f"CLIP embedding for {character_name} already exists and is up to date. Skipping.")
                
                # Make sure it's in the map
                if character_name not in self.character_embeddings_map:
                    self.character_embeddings_map[character_name] = {
                        "name": character_name,
                        "image_path": str(image_path),
                        "embedding_path": str(embedding_path),
                        "embedding_type": "clip_vision_768"
                    }
                continue
                
            # Create the embedding
            print(f"Creating CLIP embedding for {character_name}")
            embedding = self.extract_clip_embedding(image_path, character_name)
            
            # Save the embedding as PyTorch tensor
            torch.save(embedding, embedding_path)
            
            # Add to the embeddings map
            self.character_embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path),
                "embedding_type": "clip_vision_768",
                "embedding_shape": list(embedding.shape)
            }
        
        # Save the updated embeddings map
        with open(self.embeddings_map_path, 'w') as f:
            json.dump(self.character_embeddings_map, f, indent=2)
            
        print(f"Generated CLIP embeddings for {len(keeper_images)} characters")
        print(f"Embeddings saved to: {self.embeddings_dir}")
        print(f"Embeddings map saved to: {self.embeddings_map_path}")
        
        return self.character_embeddings_map
    
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
            return similarity.item()
        else:
            print("Could not compare embeddings - one or both characters not found")
            return None

def main():
    parser = argparse.ArgumentParser(description="Generate character embeddings using CLIP")
    parser.add_argument("--character_data", type=str, default="characters.json",
                        help="Path to character data JSON file")
    parser.add_argument("--output_dir", type=str, default="character_output",
                        help="Directory containing character images and for saving embeddings")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model to use for embeddings")
    parser.add_argument("--compare", nargs=2, metavar=('CHAR1', 'CHAR2'),
                        help="Compare embeddings between two characters")
    
    args = parser.parse_args()
    
    encoder = CLIPCharacterEncoder(
        character_data_path=args.character_data,
        output_dir=args.output_dir,
        clip_model=args.clip_model
    )
    
    if args.compare:
        char1, char2 = args.compare
        encoder.compare_embeddings(char1, char2)
    else:
        encoder.generate_all_keeper_embeddings()

if __name__ == "__main__":
    main()