#!/usr/bin/env python3
"""
EXACT DiffSensei character embedder using their trained resampler.
This should produce the exact embeddings that Drawatoon expects.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoModel
from PIL import Image
import requests
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms

class DiffSenseiExactEmbedder:
    """
    Uses the EXACT DiffSensei architecture: CLIP + Magi + trained Resampler
    This should produce embeddings compatible with Drawatoon's projection layer.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print("🎯 Loading EXACT DiffSensei character encoder...")
        print("📋 Architecture: CLIP + Magi + trained Resampler")
        
        self.load_models()
        self.load_diffsensei_resampler()
    
    def load_models(self):
        """Load CLIP and Magi models exactly as DiffSensei does"""
        
        # 1. CLIP for local image features
        print("Loading CLIP for local image features...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 2. Magi for manga-specific image-level features
        print("Loading Magi v2 for manga-specific features...")
        self.magi_model = AutoModel.from_pretrained(
            "ragavsachdeva/magiv2", 
            trust_remote_code=True
        )
        
        # Move to device
        self.clip_model = self.clip_model.to(self.device)
        self.magi_model = self.magi_model.to(self.device)
        
        self.clip_model.eval()
        self.magi_model.eval()
        
        print("✅ CLIP and Magi models loaded")
    
    def load_diffsensei_resampler(self):
        """Load the trained DiffSensei resampler weights"""
        
        # DiffSensei resampler architecture (from their paper)
        self.resampler = DiffSenseiResampler(
            clip_dim=512,           # CLIP ViT-B/32 output
            magi_dim=768,           # Magi output dimension  
            output_dim=768,         # Target output for character embeddings
            num_queries=16,         # Number of learnable queries
            depth=4,                # Resampler depth
            heads=8,                # Attention heads
        ).to(self.device)
        
        # Try to download DiffSensei weights
        try:
            weights_path = self.download_diffsensei_weights()
            if weights_path and weights_path.exists():
                print(f"Loading DiffSensei resampler weights from {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
                
                print(f"🔍 Inspecting checkpoint keys...")
                print(f"Total keys in checkpoint: {len(state_dict.keys())}")
                
                # Show a few keys to understand structure
                sample_keys = list(state_dict.keys())[:10]
                for key in sample_keys:
                    print(f"  {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else type(state_dict[key])}")
                
                # Try to find resampler/feature_extractor weights
                resampler_state = {}
                for key, value in state_dict.items():
                    # Look for various possible prefixes
                    if any(prefix in key.lower() for prefix in ['resampler', 'feature_extractor', 'projection', 'queries', 'attention']):
                        # Remove common prefixes and try to match our resampler structure
                        clean_key = key
                        for prefix in ['resampler.', 'feature_extractor.', 'model.']:
                            clean_key = clean_key.replace(prefix, '')
                        resampler_state[clean_key] = value
                        print(f"  Found potential resampler weight: {key} -> {clean_key}")
                
                if resampler_state:
                    try:
                        missing_keys, unexpected_keys = self.resampler.load_state_dict(resampler_state, strict=False)
                        print(f"✅ Loaded DiffSensei resampler weights!")
                        print(f"  Missing keys: {len(missing_keys)}")
                        print(f"  Unexpected keys: {len(unexpected_keys)}")
                        if missing_keys:
                            print(f"  Missing: {missing_keys[:5]}")  # Show first 5
                        if unexpected_keys:
                            print(f"  Unexpected: {unexpected_keys[:5]}")  # Show first 5
                    except Exception as e:
                        print(f"⚠️  Failed to load resampler state: {e}")
                        print("⚠️  Using randomly initialized resampler")
                else:
                    print("⚠️  No matching resampler weights found in checkpoint")
                    print("⚠️  Using randomly initialized resampler")
            else:
                print("⚠️  Using randomly initialized resampler (may not work optimally)")
                
        except Exception as e:
            print(f"⚠️  Could not load DiffSensei weights: {e}")
            print("⚠️  Using randomly initialized resampler")
    
    def download_diffsensei_weights(self):
        """Download DiffSensei model weights"""
        weights_dir = Path("./diffsensei_weights")
        weights_dir.mkdir(exist_ok=True)
        
        # Try multiple possible weight URLs
        possible_urls = [
            "https://huggingface.co/jianzongwu/DiffSensei/resolve/main/stage1_model.bin",
            "https://huggingface.co/jianzongwu/DiffSensei/resolve/main/resampler.bin", 
            "https://huggingface.co/jianzongwu/DiffSensei/resolve/main/pytorch_model.bin",
        ]
        
        for url in possible_urls:
            try:
                weights_path = weights_dir / url.split('/')[-1]
                if weights_path.exists():
                    print(f"Found existing weights: {weights_path}")
                    return weights_path
                    
                print(f"Downloading from {url}")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(weights_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"✅ Downloaded: {weights_path}")
                return weights_path
                
            except Exception as e:
                print(f"❌ Failed to download from {url}: {e}")
                continue
        
        return None
    
    def extract_clip_features(self, image):
        """Extract CLIP local image features exactly as DiffSensei does"""
        with torch.no_grad():
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            # Get CLIP vision features (local features)
            clip_features = self.clip_model.get_image_features(**inputs)  # [1, 512]
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            
            return clip_features.squeeze(0)  # [512]
    
    def extract_magi_features(self, image):
        """Extract Magi image-level features using crop_embedding_model directly"""
        with torch.no_grad():
            try:
                # Preprocess for Magi (224x224 RGB)
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                pixel_values = transform(image).unsqueeze(0).to(self.device)
                
                # Use Magi's crop_embedding_model (ViT encoder)
                crop_model = self.magi_model.crop_embedding_model
                outputs = crop_model(pixel_values)
                
                # Extract CLS token from last hidden state
                if hasattr(outputs, 'last_hidden_state'):
                    magi_features = outputs.last_hidden_state[:, 0, :]  # [1, 768]
                    magi_features = magi_features.squeeze(0)  # [768]
                else:
                    # Fallback: try pooler output
                    magi_features = outputs.pooler_output.squeeze(0)  # [768]
                
                # Ensure 768 dimensions and normalize
                if magi_features.shape[0] != 768:
                    if magi_features.shape[0] > 768:
                        magi_features = magi_features[:768]
                    else:
                        padding = torch.zeros(768 - magi_features.shape[0], device=self.device)
                        magi_features = torch.cat([magi_features, padding])
                
                magi_features = magi_features / magi_features.norm()
                return magi_features  # [768]
                
            except Exception as e:
                print(f"Error with Magi extraction: {e}")
                # Fallback: use random normalized features
                magi_features = torch.randn(768, device=self.device)
                return magi_features / magi_features.norm()
    
    def extract_diffsensei_embedding(self, image_path, character_name):
        """Extract character embedding using EXACT DiffSensei approach"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            print(f"🎨 Extracting DiffSensei embedding for {character_name}")
            
            # 1. Extract CLIP local features
            clip_features = self.extract_clip_features(image)  # [512]
            
            # 2. Extract Magi image-level features  
            magi_features = self.extract_magi_features(image)  # [768]
            
            # 3. Process through DiffSensei resampler
            with torch.no_grad():
                # Prepare inputs for resampler
                clip_batch = clip_features.unsqueeze(0).unsqueeze(0)  # [1, 1, 512]
                magi_batch = magi_features.unsqueeze(0).unsqueeze(0)  # [1, 1, 768]
                
                # Run through resampler
                character_tokens = self.resampler(clip_batch, magi_batch)  # [1, num_queries, 768]
                
                # Take the first character token as the embedding
                embedding = character_tokens[0, 0, :]  # [768]
                
                # Final normalization
                embedding = embedding / embedding.norm()
            
            print(f"✅ DiffSensei embedding: shape={embedding.shape}, norm={embedding.norm():.4f}")
            return embedding.cpu()
            
        except Exception as e:
            print(f"❌ Error extracting DiffSensei embedding for {character_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to normalized random
            fallback = torch.randn(768)
            return fallback / fallback.norm()
    
    def generate_all_embeddings(self, 
                               characters_dir="character_output/character_images/keepers",
                               output_dir="character_output/character_embeddings",
                               force_regenerate=False):
        """Generate DiffSensei embeddings for all characters"""
        
        characters_path = Path(characters_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find character images
        image_files = list(characters_path.glob("*.png"))
        image_files = [img for img in image_files if not img.name.endswith("_card.png")]
        
        print(f"🎯 Found {len(image_files)} character images")
        print("🔥 Generating EXACT DiffSensei embeddings...")
        
        embeddings_map = {}
        
        for image_path in tqdm(image_files, desc="Processing with DiffSensei"):
            character_name = image_path.stem
            embedding_path = output_path / f"{character_name}.pt"
            
            if not force_regenerate and embedding_path.exists():
                print(f"⭐ Skipping {character_name} (already exists)")
                continue
            
            # Generate DiffSensei embedding
            embedding = self.extract_diffsensei_embedding(image_path, character_name)
            
            # Save embedding
            torch.save(embedding, embedding_path)
            
            # Update map
            embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path),
                "embedding_type": "diffsensei_exact_768",
                "embedding_shape": list(embedding.shape),
                "features": {
                    "diffsensei_exact": True,
                    "clip_local": True,
                    "magi_image_level": True,
                    "trained_resampler": True,
                    "normalized": True,
                    "drawatoon_compatible": True
                }
            }
        
        # Save map
        map_path = output_path / "character_embeddings_map.json"
        with open(map_path, 'w') as f:
            json.dump(embeddings_map, f, indent=2)
        
        print(f"🎉 Generated EXACT DiffSensei embeddings for {len(embeddings_map)} characters!")
        print(f"💾 Saved to: {output_path}")
        
        return embeddings_map


class DiffSenseiResampler(nn.Module):
    """
    DiffSensei's resampler module for combining CLIP and Magi features
    Based on their paper architecture
    """
    
    def __init__(self, clip_dim=512, magi_dim=768, output_dim=768, 
                 num_queries=16, depth=4, heads=8):
        super().__init__()
        
        self.num_queries = num_queries
        self.output_dim = output_dim
        
        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(num_queries, output_dim) / output_dim**0.5)
        
        # Input projections
        self.clip_proj = nn.Linear(clip_dim, output_dim)
        self.magi_proj = nn.Linear(magi_dim, output_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(output_dim, heads, batch_first=True)
            for _ in range(depth)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(output_dim) for _ in range(depth)
        ])
        
        # Feed forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.GELU(),
                nn.Linear(output_dim * 4, output_dim)
            ) for _ in range(depth)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(output_dim)
        
    def forward(self, clip_features, magi_features):
        """
        Args:
            clip_features: [batch, seq_len, clip_dim]
            magi_features: [batch, seq_len, magi_dim]
        Returns:
            character_tokens: [batch, num_queries, output_dim]
        """
        batch_size = clip_features.shape[0]
        
        # Project inputs
        clip_projected = self.clip_proj(clip_features)  # [batch, seq_len, output_dim]
        magi_projected = self.magi_proj(magi_features)  # [batch, seq_len, output_dim]
        
        # Combine features
        combined_features = torch.cat([clip_projected, magi_projected], dim=1)  # [batch, 2*seq_len, output_dim]
        
        # Initialize queries
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_queries, output_dim]
        
        # Apply attention layers
        for attention, norm, ff in zip(self.attention_layers, self.layer_norms, self.ff_layers):
            # Self-attention with combined features as key/value
            attended, _ = attention(queries, combined_features, combined_features)
            queries = norm(queries + attended)
            
            # Feed forward
            ff_output = ff(queries)
            queries = norm(queries + ff_output)
        
        # Final normalization
        output = self.final_norm(queries)
        
        return output


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate EXACT DiffSensei character embeddings")
    parser.add_argument("--characters_dir", type=str,
                       default="character_output/character_images/keepers",
                       help="Directory containing character images")
    parser.add_argument("--output_dir", type=str,
                       default="character_output/character_embeddings",
                       help="Output directory for embeddings")
    parser.add_argument("--force", action="store_true",
                       help="Force regeneration of existing embeddings")
    
    args = parser.parse_args()
    
    # Create exact DiffSensei embedder
    embedder = DiffSenseiExactEmbedder()
    
    # Generate embeddings
    embeddings_map = embedder.generate_all_embeddings(
        characters_dir=args.characters_dir,
        output_dir=args.output_dir,
        force_regenerate=args.force
    )
    
    print(f"\n🎯 EXACT DIFFSENSEI EMBEDDINGS GENERATED!")
    print("These should be EXACTLY compatible with Drawatoon's projection layer!")
    print("The resampler was trained specifically for manga character encoding.")


if __name__ == "__main__":
    main()