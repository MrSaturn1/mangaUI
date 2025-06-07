#!/usr/bin/env python3
"""
Debug script to try using Magi's crop_embedding_model directly
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, ViTImageProcessor
import torchvision.transforms as transforms

def debug_magi_direct_crop_embedding():
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
    print(f"Crop embedding model: {crop_model}")
    
    # Create a simple test image
    image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)  # ViT usually wants 384x384
    print(f"Test image shape: {image.shape}")
    
    # Convert to PIL for easier processing
    pil_image = Image.fromarray(image)
    
    # Try using ViTImageProcessor to preprocess the image
    try:
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-384")
        inputs = processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        print(f"Preprocessed image shape: {pixel_values.shape}")
        
        with torch.no_grad():
            # Try calling the crop embedding model directly
            outputs = crop_model(pixel_values)
            
        print(f"Direct crop model outputs type: {type(outputs)}")
        print(f"Direct crop model outputs: {outputs}")
        
        if hasattr(outputs, 'last_hidden_state'):
            print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
            # Take the mean of the patch embeddings to get a single embedding
            embedding = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            print(f"Mean pooled embedding shape: {embedding.shape}")
            
        if hasattr(outputs, 'pooler_output'):
            print(f"Pooler output shape: {outputs.pooler_output.shape}")
            embedding = outputs.pooler_output
            
        print(f"Final embedding shape: {embedding.shape}")
        print(f"Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
        
        return embedding
        
    except Exception as e:
        print(f"Error with ViTImageProcessor: {e}")
        import traceback
        traceback.print_exc()
        
    # Try alternative approach with manual preprocessing
    try:
        print("\nTrying manual preprocessing...")
        
        # Manual preprocessing for ViT
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        pixel_values = transform(pil_image).unsqueeze(0).to(device)  # Add batch dimension
        print(f"Manual preprocessed image shape: {pixel_values.shape}")
        
        with torch.no_grad():
            outputs = crop_model(pixel_values)
            
        print(f"Manual crop model outputs type: {type(outputs)}")
        
        if hasattr(outputs, 'last_hidden_state'):
            print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
            embedding = outputs.last_hidden_state.mean(dim=1)
            print(f"Mean pooled embedding shape: {embedding.shape}")
            
        if hasattr(outputs, 'pooler_output'):
            print(f"Pooler output shape: {outputs.pooler_output.shape}")
            embedding = outputs.pooler_output
            
        print(f"Final embedding shape: {embedding.shape}")
        print(f"Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
        
        return embedding
        
    except Exception as e:
        print(f"Error with manual preprocessing: {e}")
        import traceback
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    debug_magi_direct_crop_embedding()