#!/usr/bin/env python3
"""
Debug script to try using Magi's crop_embedding_model directly with correct 224x224 size
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, ViTImageProcessor
import torchvision.transforms as transforms

def debug_magi_direct_crop_embedding_224():
    print("Loading Magi v2 model...")
    
    # Load Magi v2 model
    magi_model = AutoModel.from_pretrained(
        "ragavsachdeva/magiv2", 
        trust_remote_code=True
    )
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    magi_model = magi_model.to(device)
    magi_model.eval()
    
    print(f"Model loaded on device: {device}")
    
    # Access the crop_embedding_model directly
    crop_model = magi_model.crop_embedding_model
    print(f"Crop embedding model type: {type(crop_model)}")
    
    # Create a simple test image with correct size
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)  # ViTMAE expects 224x224
    print(f"Test image shape: {image.shape}")
    
    # Convert to PIL for easier processing
    pil_image = Image.fromarray(image)
    
    # Try manual preprocessing for ViT (224x224)
    try:
        print("Trying manual preprocessing for 224x224...")
        
        # Manual preprocessing for ViT
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        pixel_values = transform(pil_image).unsqueeze(0).to(device)  # Add batch dimension
        print(f"Manual preprocessed image shape: {pixel_values.shape}")
        
        with torch.no_grad():
            outputs = crop_model(pixel_values)
            
        print(f"Manual crop model outputs type: {type(outputs)}")
        print(f"Manual crop model outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")
        
        if hasattr(outputs, 'last_hidden_state'):
            print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
            # Take the mean of the patch embeddings to get a single embedding
            embedding = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            print(f"Mean pooled embedding shape: {embedding.shape}")
            
        if hasattr(outputs, 'pooler_output'):
            print(f"Pooler output shape: {outputs.pooler_output.shape}")
            embedding = outputs.pooler_output
            
        # Extract first token if available
        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            first_token_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            print(f"First token (CLS) embedding shape: {first_token_embedding.shape}")
            embedding = first_token_embedding
            
        print(f"Final embedding shape: {embedding.shape}")
        print(f"Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
        
        # Check if it's 768 dimensions as expected
        if embedding.shape[-1] == 768:
            print("✓ Successfully extracted 768-dimensional embedding from Magi's crop model!")
            return embedding
        else:
            print(f"✗ Embedding has {embedding.shape[-1]} dimensions, need 768")
            
        return embedding
        
    except Exception as e:
        print(f"Error with manual preprocessing: {e}")
        import traceback
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    result = debug_magi_direct_crop_embedding_224()
    if result is not None:
        print(f"\nSUCCESS: Extracted {result.shape} embedding")
    else:
        print("\nFAILED: Could not extract embedding")