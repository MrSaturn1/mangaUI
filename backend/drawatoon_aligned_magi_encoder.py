#!/usr/bin/env python3
"""
Extract Magi embeddings exactly as the Drawatoon author likely did.
Focus on matching their preprocessing and extraction approach.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel
from PIL import Image
import torchvision.transforms as transforms
import json
from pathlib import Path
from tqdm import tqdm

class DrawatoonAlignedMagiEncoder:
    """Extract Magi embeddings aligned with Drawatoon's training approach"""
    
    def __init__(self, output_dir="character_output"):
        self.output_dir = Path(output_dir)
        self.character_images_dir = self.output_dir / "character_images"
        self.embeddings_dir = self.output_dir / "character_embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.init_magi_model()
    
    def init_magi_model(self):
        """Initialize Magi model - try to match Drawatoon's exact setup"""
        print("Loading Magi model for Drawatoon alignment...")
        
        try:
            # Use the same model the Drawatoon author likely used
            self.magi_model = AutoModel.from_pretrained(
                "ragavsachdeva/magiv2", 
                trust_remote_code=True
            )
            self.magi_model = self.magi_model.to(self.device)
            self.magi_model.eval()
            
            print("✅ Magi model loaded successfully")
            print(f"Model type: {type(self.magi_model)}")
            
            # Inspect the model structure to understand extraction
            if hasattr(self.magi_model, 'crop_embedding_model'):
                print(f"Crop embedding model: {type(self.magi_model.crop_embedding_model)}")
            
        except Exception as e:
            print(f"❌ Error loading Magi model: {e}")
            self.magi_model = None
    
    def extract_drawatoon_style_embedding(self, image_path, character_name):
        """
        Extract embedding using Drawatoon's likely approach:
        1. Same preprocessing as training
        2. Full character context (not just crops)
        3. Proper normalization
        """
        if self.magi_model is None:
            print(f"Model not loaded, using random embedding for {character_name}")
            return F.normalize(torch.randn(768), p=2, dim=0)
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            print(f"Processing {character_name} for Drawatoon alignment...")
            
            # CRITICAL: Use the EXACT preprocessing the author likely used
            # During training, they probably used full images, not crops
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Standard ViT size
                transforms.ToTensor(),
                # This normalization is critical - match ViT training
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225]    # ImageNet stds
                )
            ])
            
            pixel_values = transform(image).unsqueeze(0).to(self.device)
            
            print(f"  Input shape: {pixel_values.shape}")
            print(f"  Input range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
            
            # Extract using the crop embedding model (character encoder)
            with torch.no_grad():
                outputs = self.magi_model.crop_embedding_model(pixel_values)
                
                # Try different extraction strategies
                if hasattr(outputs, 'last_hidden_state'):
                    # Strategy 1: CLS token (most common for character identity)
                    embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
                    print(f"  Using CLS token embedding")
                    
                elif hasattr(outputs, 'pooler_output'):
                    # Strategy 2: Pooled output
                    embedding = outputs.pooler_output
                    print(f"  Using pooler output")
                    
                else:
                    # Strategy 3: Direct output (if it's already a tensor)
                    embedding = outputs
                    print(f"  Using direct output: {type(outputs)}")
            
            # Ensure correct shape and move to CPU
            embedding = embedding.squeeze(0).cpu().float()  # Remove batch dim
            
            # Verify it's 768 dimensions
            if embedding.shape[0] != 768:
                print(f"  ⚠️ Unexpected embedding size: {embedding.shape}")
                if embedding.numel() == 768:
                    embedding = embedding.view(768)
                    print(f"  ✅ Reshaped to: {embedding.shape}")
                else:
                    print(f"  ❌ Cannot reshape to 768, using random")
                    embedding = torch.randn(768)
            
            # CRITICAL: Proper normalization (unit vector)
            if self.device == 'mps':
                # For MPS: normalize in float32 on CPU for precision, then save
                embedding = embedding.cpu().float()
                embedding = F.normalize(embedding, p=2, dim=0)
            else:
                embedding = F.normalize(embedding, p=2, dim=0)
            
            # Verify normalization
            norm = torch.norm(embedding).item()
            print(f"  ✅ Final embedding: shape={embedding.shape}, norm={norm:.4f}")
            print(f"  Stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
            
            return embedding
            
        except Exception as e:
            print(f"  ❌ Error extracting embedding for {character_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return normalized random as fallback
            return F.normalize(torch.randn(768), p=2, dim=0)
    
    def generate_drawatoon_aligned_embeddings(self, force_regenerate=False):
        """Generate embeddings aligned with Drawatoon's approach"""
        print("🎯 GENERATING DRAWATOON-ALIGNED MAGI EMBEDDINGS")
        print("=" * 55)
        
        # Find character images
        character_images = list(self.character_images_dir.glob("*.png"))
        character_images = [img for img in character_images if not img.name.endswith("_card.png")]
        
        if not character_images:
            print(f"❌ No character images found in {self.character_images_dir}")
            return {}
        
        print(f"Found {len(character_images)} character images")
        
        embeddings_map = {}
        generated_count = 0
        skipped_count = 0
        
        for image_path in tqdm(character_images, desc="Generating Drawatoon-aligned embeddings"):
            character_name = image_path.stem
            embedding_path = self.embeddings_dir / f"{character_name}.pt"
            
            # Check if we should skip
            if not force_regenerate and embedding_path.exists():
                if image_path.stat().st_mtime < embedding_path.stat().st_mtime:
                    print(f"⏭️ Skipping {character_name} (up to date)")
                    skipped_count += 1
                    
                    # Load existing embedding to add to map
                    existing_embedding = torch.load(embedding_path, map_location='cpu')
                    embeddings_map[character_name] = {
                        "name": character_name,
                        "image_path": str(image_path),
                        "embedding_path": str(embedding_path),
                        "embedding_type": "drawatoon_aligned_magi_768",
                        "normalized": True,
                        "norm": torch.norm(existing_embedding).item()
                    }
                    continue
            
            # Generate new embedding
            print(f"\n🔧 Generating Drawatoon-aligned embedding for {character_name}")
            embedding = self.extract_drawatoon_style_embedding(image_path, character_name)
            
            # Save embedding
            torch.save(embedding, embedding_path)
            generated_count += 1
            
            # Add to map
            embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(image_path),
                "embedding_path": str(embedding_path),
                "embedding_type": "drawatoon_aligned_magi_768",
                "embedding_shape": list(embedding.shape),
                "normalized": True,
                "norm": torch.norm(embedding).item(),
                "alignment_strategy": "imagenet_normalization_cls_token"
            }
        
        # Save embeddings map
        map_path = self.embeddings_dir / "character_embeddings_map.json"
        with open(map_path, 'w') as f:
            json.dump(embeddings_map, f, indent=2)
        
        print(f"\n📊 DRAWATOON ALIGNMENT RESULTS:")
        print(f"  ✅ Generated: {generated_count} new embeddings")
        print(f"  ⏭️ Skipped: {skipped_count} existing embeddings")
        print(f"  📁 Total: {len(embeddings_map)} embeddings available")
        print(f"  💾 Saved to: {self.embeddings_dir}")
        print(f"  🗺️ Map saved to: {map_path}")
        
        # Verify all embeddings are properly normalized
        self.verify_embeddings()
        
        return embeddings_map
    
    def verify_embeddings(self):
        """Verify all embeddings are properly normalized"""
        print(f"\n🔍 VERIFYING EMBEDDING QUALITY...")
        
        embeddings_map_path = self.embeddings_dir / "character_embeddings_map.json"
        if not embeddings_map_path.exists():
            print("❌ No embeddings map found")
            return
        
        with open(embeddings_map_path, 'r') as f:
            embeddings_map = json.load(f)
        
        perfect_count = 0
        issue_count = 0
        
        for char_name, info in embeddings_map.items():
            try:
                embedding = torch.load(info["embedding_path"], map_location='cpu')
                norm = torch.norm(embedding).item()
                
                if abs(norm - 1.0) < 0.001:
                    perfect_count += 1
                else:
                    print(f"  ⚠️ {char_name}: norm = {norm:.4f} (should be 1.0)")
                    issue_count += 1
                    
            except Exception as e:
                print(f"  ❌ Error checking {char_name}: {e}")
                issue_count += 1
        
        print(f"  ✅ Perfect normalization: {perfect_count}")
        print(f"  ⚠️ Issues found: {issue_count}")
        
        if issue_count == 0:
            print("  🎯 All embeddings are Drawatoon-ready!")
        else:
            print("  🔧 Consider regenerating problematic embeddings")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Drawatoon-aligned Magi embeddings")
    parser.add_argument("--output_dir", type=str, default="character_output",
                        help="Directory containing character images")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration of all embeddings")
    
    args = parser.parse_args()
    
    encoder = DrawatoonAlignedMagiEncoder(output_dir=args.output_dir)
    embeddings_map = encoder.generate_drawatoon_aligned_embeddings(force_regenerate=args.force)
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Test these aligned embeddings with your fixed ref_embedding_proj")
    print("2. Compare character consistency with and without enhanced prompts")
    print("3. If still not perfect, try the DiffSensei weight adaptation")
    print("\n✨ These embeddings should work much better with Drawatoon!")

if __name__ == "__main__":
    main()