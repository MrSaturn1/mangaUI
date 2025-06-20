#!/usr/bin/env python3
"""
Fixed hybrid character encoder inspired by DiffSensei's approach.
Combines CLIP and Magi v2 encoders with a projection layer for improved character consistency.
Removes OpenCV dependency and fixes tensor dimension issues.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageFilter
from tqdm import tqdm
import argparse
import numpy as np
from transformers import AutoModel, CLIPProcessor, CLIPModel
import torchvision.transforms as transforms

class ProjectionLayer(nn.Module):
    """
    Projection layer to combine CLIP and Magi embeddings.
    Inspired by DiffSensei's hybrid approach.
    """
    def __init__(self, clip_dim=512, magi_dim=768, output_dim=768):
        super().__init__()
        self.clip_dim = clip_dim
        self.magi_dim = magi_dim
        self.output_dim = output_dim
        
        # Learned projection for CLIP features
        self.clip_proj = nn.Linear(clip_dim, output_dim // 2)
        
        # Learned projection for Magi features  
        self.magi_proj = nn.Linear(magi_dim, output_dim // 2)
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, clip_embeds, magi_embeds):
        """
        Combine CLIP and Magi embeddings
        
        Args:
            clip_embeds: [batch_size, clip_dim] CLIP image embeddings
            magi_embeds: [batch_size, magi_dim] Magi character embeddings
        
        Returns:
            combined_embeds: [batch_size, output_dim] Combined embeddings
        """
        # Project both embeddings to half the output dimension
        clip_proj = self.clip_proj(clip_embeds)  # [batch, output_dim//2]
        magi_proj = self.magi_proj(magi_embeds)  # [batch, output_dim//2]
        
        # Concatenate projected features
        combined = torch.cat([clip_proj, magi_proj], dim=-1)  # [batch, output_dim]
        
        # Apply fusion layer
        output = self.fusion(combined)
        
        return output

class FixedHybridCharacterEncoder:
    """
    Hybrid character encoder combining CLIP and Magi v2 approaches.
    Fixed version without OpenCV dependency and with better error handling.
    """
    
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
                
        # Initialize models
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.init_clip_model()
        self.init_magi_v2_model()
        self.init_projection_layer()
        
    def init_clip_model(self):
        """Initialize CLIP model for image embeddings"""
        print("Loading CLIP model...")
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def init_magi_v2_model(self):
        """Initialize Magi v2 model for character analysis"""
        print("Loading Magi v2 model...")
        try:
            self.magi_model = AutoModel.from_pretrained(
                "ragavsachdeva/magiv2", 
                trust_remote_code=True
            )
            self.magi_model = self.magi_model.to(self.device)
            self.magi_model.eval()
            print("Magi v2 model loaded successfully")
        except Exception as e:
            print(f"Error loading Magi v2 model: {e}")
            self.magi_model = None
    
    def init_projection_layer(self):
        """Initialize the projection layer for combining embeddings"""
        print("Initializing projection layer...")
        self.projection_layer = ProjectionLayer(
            clip_dim=512,
            magi_dim=768, 
            output_dim=768
        ).to(self.device)
        
        # For now, use random initialization
        # In a full implementation, this would be trained on character matching tasks
        print("Projection layer initialized with random weights")
        print("Note: For optimal performance, this should be trained on character matching data")
    
    def detect_character_region_simple(self, image):
        """
        Simple character region detection without OpenCV.
        Uses PIL to find the main content area by detecting non-white regions.
        
        Args:
            image: PIL Image
            
        Returns:
            bbox: [x, y, width, height] or None if no character detected
        """
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            
            # Apply threshold to separate character from background
            # Assuming manga images have white/light backgrounds
            threshold = 240
            binary = gray.point(lambda x: 0 if x < threshold else 255, '1')
            
            # Find bounding box of non-white content
            bbox = binary.getbbox()  # Returns (left, top, right, bottom) or None
            
            if bbox is None:
                return None
            
            # Convert to [x, y, width, height] format and add padding
            x, y, right, bottom = bbox
            width = right - x
            height = bottom - y
            
            # Add padding (10% of image size)
            padding_x = max(5, image.width // 20)
            padding_y = max(5, image.height // 20)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            width = min(image.width - x, width + 2 * padding_x)
            height = min(image.height - y, height + 2 * padding_y)
            
            return [x, y, width, height]
            
        except Exception as e:
            print(f"Error detecting character region: {e}")
            return None
    
    def crop_character(self, image, bbox=None):
        """
        Crop character from image using bounding box.
        
        Args:
            image: PIL Image
            bbox: [x, y, width, height] or None for auto-detection
            
        Returns:
            cropped_image: PIL Image of cropped character
        """
        if bbox is None:
            bbox = self.detect_character_region_simple(image)
            
        if bbox is None:
            # If no bbox detected, return center crop (square)
            w, h = image.size
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            return image.crop((left, top, left + crop_size, top + crop_size))
        
        x, y, width, height = bbox
        return image.crop((x, y, x + width, y + height))
    
    def preprocess_for_clip(self, image):
        """Preprocess image specifically for CLIP"""
        # Enhance contrast slightly for better CLIP understanding
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.1)
        return enhanced
    
    def preprocess_for_magi(self, image):
        """Preprocess image specifically for Magi v2"""
        # Apply slight sharpening for better detail extraction
        sharpened = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=2))
        return sharpened
    
    def extract_clip_embedding(self, image, character_name):
        """Extract CLIP embedding from character image"""
        if self.clip_model is None:
            print(f"CLIP model not loaded, using random embedding for {character_name}")
            return torch.randn(512).float()
        
        try:
            # Crop character from image
            cropped_image = self.crop_character(image)
            
            # Preprocess for CLIP
            processed_image = self.preprocess_for_clip(cropped_image)
            
            # Process with CLIP
            inputs = self.clip_processor(images=processed_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Normalize and return
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = embedding.squeeze(0).cpu().float()  # Remove batch dimension
            
            print(f"Extracted CLIP embedding for {character_name}: shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            print(f"Error extracting CLIP embedding for {character_name}: {e}")
            return torch.randn(512).float()
    
    def extract_magi_v2_embedding(self, image, character_name):
        """Extract Magi v2 embedding from character image"""
        if self.magi_model is None:
            print(f"Magi model not loaded, using random embedding for {character_name}")
            return torch.randn(768).float()
        
        try:
            # Crop character from image
            cropped_image = self.crop_character(image)
            
            # Preprocess for Magi
            processed_image = self.preprocess_for_magi(cropped_image)
            
            # Preprocess for ViTMAE model (224x224)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            pixel_values = transform(processed_image).unsqueeze(0).to(self.device)
            
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
                return torch.randn(768).float()
            
            embedding = embedding.cpu().float()
            
            print(f"Extracted Magi v2 embedding for {character_name}: shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            print(f"Error extracting Magi v2 embedding for {character_name}: {e}")
            return torch.randn(768).float()
    
    def extract_hybrid_embedding(self, image_path, character_name):
        """
        Extract hybrid embedding combining CLIP and Magi v2.
        This is the main method implementing DiffSensei's approach.
        
        Args:
            image_path: Path to character image
            character_name: Name of character
            
        Returns:
            hybrid_embedding: 768-dimensional combined embedding
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            print(f"Extracting hybrid embedding for {character_name}")
            
            # Extract CLIP embedding (512-dim)
            clip_embedding = self.extract_clip_embedding(image, character_name)
            
            # Extract Magi v2 embedding (768-dim)
            magi_embedding = self.extract_magi_v2_embedding(image, character_name)
            
            # Combine using projection layer
            clip_batch = clip_embedding.unsqueeze(0).to(self.device)  # [1, 512]
            magi_batch = magi_embedding.unsqueeze(0).to(self.device)  # [1, 768]
            
            with torch.no_grad():
                hybrid_embedding = self.projection_layer(clip_batch, magi_batch)
            
            hybrid_embedding = hybrid_embedding.squeeze(0).cpu().float()  # [768]
            
            print(f"Created hybrid embedding for {character_name}: shape {hybrid_embedding.shape}")
            print(f"Embedding stats: min={hybrid_embedding.min():.4f}, max={hybrid_embedding.max():.4f}, mean={hybrid_embedding.mean():.4f}")
            
            return hybrid_embedding
            
        except Exception as e:
            print(f"Error extracting hybrid embedding for {character_name}: {e}")
            import traceback
            traceback.print_exc()
            # Return random embedding as fallback
            return torch.randn(768).float()
    
    def generate_all_keeper_embeddings(self, force_regenerate=False):
        """Generate hybrid embeddings for all character images in the keepers folder"""
        if not self.keepers_dir.exists():
            print(f"Keepers directory not found: {self.keepers_dir}")
            return
            
        # Get all PNG files in the keepers folder
        keeper_images = list(self.keepers_dir.glob("*.png"))
        keeper_images = [img for img in keeper_images if not img.name.endswith("_card.png")]
        
        print(f"Found {len(keeper_images)} keeper images")
        print("Using hybrid CLIP + Magi v2 embedding extraction with character cropping")
        
        # Create embeddings for each keeper image
        for image_path in tqdm(keeper_images, desc="Generating hybrid embeddings"):
            character_name = image_path.stem  # Get filename without extension
            
            # Skip if embedding already exists and is up to date
            embedding_path = self.embeddings_dir / f"{character_name}.pt"
            
            if not force_regenerate and embedding_path.exists() and image_path.stat().st_mtime < embedding_path.stat().st_mtime:
                # Check if it's already a hybrid embedding
                if character_name in self.character_embeddings_map:
                    if self.character_embeddings_map[character_name].get("embedding_type") == "hybrid_clip_magi_768":
                        print(f"Hybrid embedding for {character_name} already exists and is up to date. Skipping.")
                        continue
                
            # Create the hybrid embedding
            print(f"Creating hybrid embedding for {character_name}")
            embedding = self.extract_hybrid_embedding(image_path, character_name)
            
            # Save the embedding as PyTorch tensor
            torch.save(embedding, embedding_path)
            
            # Add to the embeddings map
            self.character_embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path),
                "embedding_type": "hybrid_clip_magi_768",
                "embedding_shape": list(embedding.shape),
                "components": {
                    "clip_dim": 512,
                    "magi_dim": 768,
                    "output_dim": 768,
                    "uses_cropping": True,
                    "uses_projection_layer": True
                }
            }
        
        # Save the updated embeddings map
        with open(self.embeddings_map_path, 'w') as f:
            json.dump(self.character_embeddings_map, f, indent=2)
            
        print(f"Generated hybrid embeddings for {len(keeper_images)} characters")
        print(f"Embeddings saved to: {self.embeddings_dir}")
        print(f"Embeddings map saved to: {self.embeddings_map_path}")
        
        return self.character_embeddings_map
    
    def debug_cropping(self, character_name):
        """Debug the character cropping for a specific character"""
        image_path = self.keepers_dir / f"{character_name}.png"
        if not image_path.exists():
            print(f"Image not found for {character_name}")
            return
            
        image = Image.open(image_path).convert("RGB")
        
        bbox = self.detect_character_region_simple(image)
        cropped = self.crop_character(image, bbox)
        
        # Save for inspection
        debug_dir = Path("debug_crops")
        debug_dir.mkdir(exist_ok=True)
        
        cropped.save(debug_dir / f"crop_{character_name}.png")
        print(f"Original size: {image.size}, Cropped size: {cropped.size}")
        print(f"Bounding box: {bbox}")
        print(f"Debug crop saved to: debug_crops/crop_{character_name}.png")
    
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
    parser = argparse.ArgumentParser(description="Generate hybrid character embeddings using CLIP + Magi v2")
    parser.add_argument("--character_data", type=str, default="characters.json",
                        help="Path to character data JSON file")
    parser.add_argument("--output_dir", type=str, default="character_output",
                        help="Directory containing character images and for saving embeddings")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration of all embeddings even if they exist")
    parser.add_argument("--debug_crop", type=str, metavar='CHARACTER',
                        help="Debug character cropping for a specific character")
    parser.add_argument("--compare", nargs=2, metavar=('CHAR1', 'CHAR2'),
                        help="Compare embeddings between two characters")
    
    args = parser.parse_args()
    
    encoder = FixedHybridCharacterEncoder(
        character_data_path=args.character_data,
        output_dir=args.output_dir
    )
    
    if args.debug_crop:
        encoder.debug_cropping(args.debug_crop)
    elif args.compare:
        char1, char2 = args.compare
        encoder.compare_embeddings(char1, char2)
    else:
        encoder.generate_all_keeper_embeddings(force_regenerate=args.force)

if __name__ == "__main__":
    main()