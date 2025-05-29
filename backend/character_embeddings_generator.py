import os
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

class CharacterEmbeddingsGenerator:
    def __init__(self, model_path, character_data_path, output_dir="character_output"):
        # Load character data
        with open(character_data_path, 'r', encoding='utf-8') as f:
            self.character_data = json.load(f)
            
        self.model_path = model_path
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
                
        # Initialize the character encoder (this would be your chosen embedding model)
        self.init_encoder()
        
    def init_encoder(self):
        """Initialize the model used to create embeddings from character images"""
        # This would be your CLIP or other image embedding model
        # For example if using CLIP:
        # from transformers import CLIPProcessor, CLIPModel
        # self.encoder_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.encoder_model = self.encoder_model.to(self.device)
        
        print("Character encoder initialized")
    
    def create_embedding_from_image(self, image_path):
        """Create an embedding from a character image"""
        # Load the image
        image = Image.open(image_path)
        
        # Create embedding using your encoder model
        # For example with CLIP:
        # inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # with torch.no_grad():
        #     outputs = self.encoder_model.get_image_features(**inputs)
        # embedding = outputs.cpu().numpy()
        
        # For demonstration, we'll just create a dummy embedding
        # Replace this with your actual embedding code
        embedding = torch.rand(768).numpy()  # Typical size for CLIP embeddings
        
        return embedding
    
    def generate_all_keeper_embeddings(self):
        """Generate embeddings for all character images in the keepers folder"""
        if not self.keepers_dir.exists():
            print(f"Keepers directory not found: {self.keepers_dir}")
            return
            
        # Get all PNG files in the keepers folder
        keeper_images = list(self.keepers_dir.glob("*.png"))
        keeper_images = [img for img in keeper_images if not img.name.endswith("_card.png")]
        
        print(f"Found {len(keeper_images)} keeper images")
        
        # Create embeddings for each keeper image
        for image_path in tqdm(keeper_images, desc="Generating embeddings"):
            character_name = image_path.stem  # Get filename without extension
            
            # Skip if embedding already exists and is up to date (based on file modification time)
            embedding_path = self.embeddings_dir / f"{character_name}.pt"
            
            if embedding_path.exists() and image_path.stat().st_mtime < embedding_path.stat().st_mtime:
                print(f"Embedding for {character_name} already exists and is up to date. Skipping.")
                
                # Make sure it's in the map
                if character_name not in self.character_embeddings_map:
                    self.character_embeddings_map[character_name] = {
                        "name": character_name,
                        "image_path": str(image_path),
                        "embedding_path": str(embedding_path)
                    }
                continue
                
            # Create the embedding
            print(f"Creating embedding for {character_name}")
            embedding = self.create_embedding_from_image(image_path)
            
            # Save the embedding
            torch.save(embedding, embedding_path)
            
            # Add to the embeddings map
            self.character_embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path)
            }
        
        # Save the updated embeddings map
        with open(self.embeddings_map_path, 'w') as f:
            json.dump(self.character_embeddings_map, f, indent=2)
            
        print(f"Generated embeddings for {len(keeper_images)} characters")
        print(f"Embeddings saved to: {self.embeddings_dir}")
        print(f"Embeddings map saved to: {self.embeddings_map_path}")
        
        return self.character_embeddings_map

def main():
    parser = argparse.ArgumentParser(description="Generate character embeddings from keeper images")
    parser.add_argument("--model_path", type=str, default="./drawatoon-v1",
                        help="Path to the embedding model")
    parser.add_argument("--character_data", type=str, default="charactersNL.json",
                        help="Path to character data JSON file")
    parser.add_argument("--output_dir", type=str, default="character_output",
                        help="Directory containing character images and for saving embeddings")
    
    args = parser.parse_args()
    
    generator = CharacterEmbeddingsGenerator(
        model_path=args.model_path,
        character_data_path=args.character_data,
        output_dir=args.output_dir
    )
    
    generator.generate_all_keeper_embeddings()

if __name__ == "__main__":
    main()