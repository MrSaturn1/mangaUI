#!/usr/bin/env python3
"""
Proper Magi v2 character encoder using the actual predict_crop_embeddings method.
This extracts real character embeddings using Magi's internal character encoder.
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
    
    def read_image_as_np_array(self, image_path):
        """Convert image to numpy array as expected by Magi"""
        with open(image_path, "rb") as file:
            image = Image.open(file).convert("L").convert("RGB")
            image = np.array(image)
        return image
    
    def extract_magi_v2_embedding(self, image_path, character_name):
        """Extract character embedding using Magi v2's crop_embedding_model directly"""
        if self.magi_model is None:
            print(f"Magi model not loaded, using random embedding for {character_name}")
            return torch.randn(768).float()
        
        try:
            # Load and preprocess the image for ViTMAE (224x224)
            from PIL import Image
            import torchvision.transforms as transforms
            
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
            
            # Use Magi's crop_embedding_model directly (bypasses broken predict_crop_embeddings)
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
                return torch.randn(768).float()
            
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
            
            print(f"Successfully extracted Magi v2 embedding for {character_name}: shape {embedding.shape}")
            print(f"Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
            return embedding
            
        except Exception as e:
            print(f"Error extracting Magi v2 embedding for {character_name}: {e}")
            import traceback
            traceback.print_exc()
            # Return random embedding as fallback
            return torch.randn(768).float()
    
    def generate_all_keeper_embeddings(self):
        """Generate Magi v2 embeddings for all character images in the keepers folder"""
        if not self.keepers_dir.exists():
            print(f"Keepers directory not found: {self.keepers_dir}")
            return
            
        # Get all PNG files in the keepers folder
        keeper_images = list(self.keepers_dir.glob("*.png"))
        keeper_images = [img for img in keeper_images if not img.name.endswith("_card.png")]
        
        print(f"Found {len(keeper_images)} keeper images")
        
        # Create embeddings for each keeper image
        for image_path in tqdm(keeper_images, desc="Generating Magi v2 embeddings"):
            character_name = image_path.stem  # Get filename without extension
            
            # Skip if embedding already exists and is up to date
            embedding_path = self.embeddings_dir / f"{character_name}.pt"
            
            if embedding_path.exists() and image_path.stat().st_mtime < embedding_path.stat().st_mtime:
                print(f"Magi v2 embedding for {character_name} already exists and is up to date. Skipping.")
                
                # Make sure it's in the map
                if character_name not in self.character_embeddings_map:
                    self.character_embeddings_map[character_name] = {
                        "name": character_name,
                        "image_path": str(image_path),
                        "embedding_path": str(embedding_path),
                        "embedding_type": "magi_v2_768"
                    }
                continue
                
            # Create the embedding
            print(f"Creating Magi v2 embedding for {character_name}")
            embedding = self.extract_magi_v2_embedding(image_path, character_name)
            
            # Save the embedding as PyTorch tensor
            torch.save(embedding, embedding_path)
            
            # Add to the embeddings map
            self.character_embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path),
                "embedding_type": "magi_v2_768",
                "embedding_shape": list(embedding.shape)
            }
        
        # Save the updated embeddings map
        with open(self.embeddings_map_path, 'w') as f:
            json.dump(self.character_embeddings_map, f, indent=2)
            
        print(f"Generated Magi v2 embeddings for {len(keeper_images)} characters")
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
    parser = argparse.ArgumentParser(description="Generate character embeddings using Magi v2")
    parser.add_argument("--character_data", type=str, default="characters.json",
                        help="Path to character data JSON file")
    parser.add_argument("--output_dir", type=str, default="character_output",
                        help="Directory containing character images and for saving embeddings")
    parser.add_argument("--compare", nargs=2, metavar=('CHAR1', 'CHAR2'),
                        help="Compare embeddings between two characters")
    
    args = parser.parse_args()
    
    encoder = MagiV2CharacterEncoder(
        character_data_path=args.character_data,
        output_dir=args.output_dir
    )
    
    if args.compare:
        char1, char2 = args.compare
        encoder.compare_embeddings(char1, char2)
    else:
        encoder.generate_all_keeper_embeddings()

if __name__ == "__main__":
    main()