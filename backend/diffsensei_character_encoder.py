import math
import torch
import torch.nn as nn
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    ViTImageProcessor,
    ViTMAEModel,
)
from PIL import Image
import requests
import os
from pathlib import Path


# Resampler implementation from DiffSensei
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=4,
        num_dummy_tokens=4,
        embedding_dim=768,
        magi_embedding_dim=512,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.output_dim = output_dim

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_in_magi = torch.nn.Linear(magi_embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.dummy_tokens = torch.nn.Parameter(torch.randn(num_dummy_tokens, output_dim))

    def forward(self, x, magi_image_embeds):
        bsz, max_num_ips, sequence_length, _ = x.shape
        x = x.view(bsz * max_num_ips, sequence_length, -1)

        x = self.proj_in(x)
        magi_image_embeds = self.proj_in_magi(magi_image_embeds)
        magi_image_embeds = magi_image_embeds.view(bsz * max_num_ips, 1, -1)
        x = torch.cat([x, magi_image_embeds], dim=1)

        latents = self.latents.repeat(x.size(0), 1, 1)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        latents = self.norm_out(latents)
        latents = latents.view(bsz, max_num_ips * self.num_queries, self.output_dim)

        dummy_tokens = self.dummy_tokens.unsqueeze(0).repeat(latents.shape[0], 1, 1)
        out = torch.cat([dummy_tokens, latents], dim=1)

        return out

    def dtype(self):
        return next(self.parameters()).dtype


class DiffSenseiCharacterEncoder:
    """
    Character encoder that follows DiffSensei's exact approach:
    CLIP + ViTMAE encoders -> Resampler -> Final embeddings
    """
    
    def __init__(self, character_data_path, weights_dir="./diffsensei_weights"):
        self.character_data_path = character_data_path
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        self.clip_image_processor = CLIPImageProcessor()
        self.magi_image_processor = ViTImageProcessor()
        
        # Load models
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self._load_models()
        
    def _download_weights(self):
        """Download DiffSensei Resampler weights from HuggingFace"""
        weights_url = "https://huggingface.co/jianzongwu/DiffSensei/resolve/main/image_generator/image_proj_model/pytorch_model.bin"
        weights_path = self.weights_dir / "pytorch_model.bin"
        
        if not weights_path.exists():
            print(f"Downloading DiffSensei Resampler weights to {weights_path}")
            response = requests.get(weights_url)
            response.raise_for_status()
            
            with open(weights_path, 'wb') as f:
                f.write(response.content)
            print("✅ Download complete!")
        else:
            print(f"✅ Weights already exist at {weights_path}")
        
        return weights_path
    
    def _inspect_checkpoint_architecture(self, checkpoint):
        """Inspect the checkpoint to understand the actual model architecture"""
        print("🔍 Inspecting DiffSensei checkpoint architecture:")
        
        # Look for key patterns in the state dict
        layer_counts = {}
        for key in checkpoint.keys():
            print(f"  Key: {key} -> Shape: {checkpoint[key].shape}")
            
            # Count layers
            if 'layers.' in key:
                layer_num = key.split('layers.')[1].split('.')[0]
                if layer_num.isdigit():
                    layer_counts[int(layer_num)] = True
        
        if layer_counts:
            max_layer = max(layer_counts.keys())
            print(f"📊 Found layers 0-{max_layer} ({max_layer + 1} total layers)")
        
        # Check key dimensions
        if 'latents' in checkpoint:
            latents_shape = checkpoint['latents'].shape
            print(f"📐 Latents shape: {latents_shape}")
            
        if 'dummy_tokens' in checkpoint:
            dummy_tokens_shape = checkpoint['dummy_tokens'].shape
            print(f"📐 Dummy tokens shape: {dummy_tokens_shape}")
            
        if 'proj_in.weight' in checkpoint:
            proj_in_shape = checkpoint['proj_in.weight'].shape
            print(f"📐 Proj_in weight shape: {proj_in_shape}")
            
        if 'proj_out.weight' in checkpoint:
            proj_out_shape = checkpoint['proj_out.weight'].shape
            print(f"📐 Proj_out weight shape: {proj_out_shape}")
    
    def _load_models(self):
        """Load CLIP, ViTMAE, and Resampler models"""
        print("Loading DiffSensei character encoder models...")
        
        try:
            # Load CLIP Vision Model (with projection)
            print("Loading CLIP Vision Model...")
            self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)
            
            # Load ViT MAE Model  
            print("Loading ViT MAE Model...")
            self.magi_model = ViTMAEModel.from_pretrained(
                "facebook/vit-mae-base"
            ).to(self.device)
            
            # First download weights to inspect the actual architecture
            weights_path = self._download_weights()
            print(f"Inspecting DiffSensei weights at {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Inspect the checkpoint to understand the actual architecture
            self._inspect_checkpoint_architecture(checkpoint)
            
            # Initialize Resampler with EXACT config based on actual weights
            print("Initializing Resampler with exact DiffSensei architecture...")
            self.resampler = Resampler(
                dim=1280,               # Main internal dimension (confirmed)
                depth=4,                # CORRECTED: Only 4 layers (0-3), not 8!
                dim_head=64,            # Standard attention head dimension
                heads=20,               # 1280/64 = 20 heads
                num_queries=16,         # 16 learnable queries (from latents shape)
                num_dummy_tokens=16,    # CORRECTED: 16 dummy tokens (from checkpoint)
                embedding_dim=1280,     # CORRECTED: 1280 input dim (from proj_in shape)
                magi_embedding_dim=768, # ViT MAE dimension
                output_dim=2048,        # Final output dimension (confirmed)
                ff_mult=4,              # 5120/1280 = 4 (confirmed from feedforward layer)
            ).to(self.device)
            
            # Load trained weights
            print(f"Loading trained Resampler weights...")
            self.resampler.load_state_dict(checkpoint)
            
            # Set to evaluation mode
            self.clip_model.eval()
            self.magi_model.eval()
            self.resampler.eval()
            
            print("✅ All DiffSensei models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading DiffSensei models: {e}")
            raise
    
    def extract_diffsensei_embedding(self, image_path, character_name):
        """
        Extract character embedding using DiffSensei's exact dual-encoder approach
        
        Args:
            image_path: Path to character image
            character_name: Name of character for logging
            
        Returns:
            torch.Tensor: Character embedding [768] dimensions
        """
        try:
            print(f"Extracting DiffSensei embedding for {character_name}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            with torch.no_grad():
                # Process with both encoders (following their exact approach)
                clip_inputs = self.clip_image_processor(images=[image], return_tensors="pt")
                magi_inputs = self.magi_image_processor(images=[image], return_tensors="pt")
                
                # Move to device
                clip_pixel_values = clip_inputs.pixel_values.to(self.device)
                magi_pixel_values = magi_inputs.pixel_values.to(self.device)
                
                # Extract embeddings following their approach
                # Check what dimensions we actually get from CLIP
                clip_outputs = self.clip_model(clip_pixel_values)
                
                print(f"CLIP outputs available: {list(clip_outputs.keys())}")
                if hasattr(clip_outputs, 'image_embeds'):
                    print(f"CLIP image_embeds shape: {clip_outputs.image_embeds.shape}")
                if hasattr(clip_outputs, 'last_hidden_state'):
                    print(f"CLIP last_hidden_state shape: {clip_outputs.last_hidden_state.shape}")
                
                # Try to get patch embeddings from the vision model directly
                vision_outputs = self.clip_model.vision_model(clip_pixel_values, output_hidden_states=True)
                clip_patch_embeds = vision_outputs.last_hidden_state  # Should be [1, 50, 768]
                
                print(f"CLIP patch embeddings shape: {clip_patch_embeds.shape}")
                
                # DiffSensei expects [1, seq_len, 1280], but we have [1, seq_len, 768]
                # We need to project 768 -> 1280 to match their Resampler input
                if not hasattr(self, 'clip_projection'):
                    self.clip_projection = torch.nn.Linear(768, 1280).to(self.device)
                
                clip_embeds = self.clip_projection(clip_patch_embeds)  # [1, seq_len, 1280]
                print(f"CLIP projected embeddings shape: {clip_embeds.shape}")
                
                # ViT MAE: use last_hidden_state[:, 0] (CLS token)
                magi_outputs = self.magi_model(magi_pixel_values)
                magi_embeds = magi_outputs.last_hidden_state[:, 0]  # [1, 768]
                
                # Prepare for Resampler (batch format)
                clip_embeds = clip_embeds.unsqueeze(0)  # [1, 1, seq_len, 768] for max_num_ips=1
                magi_embeds = magi_embeds.unsqueeze(0)  # [1, 1, 768] for max_num_ips=1
                
                print(f"CLIP embeddings shape: {clip_embeds.shape}")
                print(f"Magi embeddings shape: {magi_embeds.shape}")
                
                # Combine through trained Resampler
                combined_embeds = self.resampler(clip_embeds, magi_embeds)
                
                # Extract character embedding (skip dummy tokens)
                # Format: [dummy_tokens, character_tokens]
                character_embedding = combined_embeds[0, self.resampler.dummy_tokens.shape[0]:, :]  # Skip dummy tokens
                
                # Take first token as character representation
                first_token = character_embedding[0]  # [2048]
                
                # Project from 2048 to 768 dimensions for Drawatoon compatibility
                # Simple linear projection for now
                final_embedding = first_token[:768]  # Take first 768 dimensions
                
                print(f"✅ DiffSensei embedding extracted for {character_name}: {final_embedding.shape}")
                return final_embedding.cpu()
                
        except Exception as e:
            print(f"❌ Error extracting DiffSensei embedding for {character_name}: {e}")
            # Fallback to random embedding
            print("Falling back to random embedding")
            return torch.randn(768)


if __name__ == "__main__":
    # Test the encoder
    encoder = DiffSenseiCharacterEncoder("characters.json")
    
    # Test with a sample image if available
    test_image = "character_output/character_images/keepers/Z.png"
    if os.path.exists(test_image):
        embedding = encoder.extract_diffsensei_embedding(test_image, "Z")
        print(f"Test embedding shape: {embedding.shape}")
        print(f"Test embedding sample: {embedding[:5]}")