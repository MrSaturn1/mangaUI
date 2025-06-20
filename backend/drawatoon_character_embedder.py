"""
Complete solution for generating character embeddings for Drawatoon model.
Based on analysis of the IPAdapter architecture, this uses OpenCLIP ViT-H-14.
"""

import torch
import open_clip
from PIL import Image
import numpy as np
from pathlib import Path
import cv2

class DrawatoonCharacterEmbedder:
    """
    Generates 768-dimensional character embeddings for Drawatoon model.
    Uses OpenCLIP ViT-H-14 which is the standard for IPAdapter implementations.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None
        self.preprocess = None
        self.load_model()
    
    def load_model(self):
        """Load OpenCLIP ViT-H-14 model - most likely candidate for drawatoon"""
        print("Loading OpenCLIP ViT-H-14 model...")
        
        # Primary candidate - OpenCLIP ViT-H-14 (768 dimensions)
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-H-14', 
                pretrained='laion2b_s32b_b79k',
                device=self.device
            )
            self.model.eval()
            print("✓ Successfully loaded OpenCLIP ViT-H-14")
            return
        except Exception as e:
            print(f"Failed to load OpenCLIP: {e}")
        
        # Fallback to standard CLIP ViT-L/14 (also 768 dimensions)
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
            self.model.eval()
            print("✓ Successfully loaded CLIP ViT-L/14 as fallback")
            return
        except Exception as e:
            print(f"Failed to load CLIP: {e}")
        
        raise RuntimeError("Could not load any suitable image encoder")
    
    def preprocess_character_image(self, image_path, crop_character=True, target_size=224):
        """
        Preprocess character image for optimal embedding extraction.
        
        Args:
            image_path: Path to character image
            crop_character: Whether to crop/focus on the character
            target_size: Target size for the model input
        """
        image = Image.open(image_path).convert('RGB')
        
        if crop_character:
            # Simple center crop - you might want to implement face detection here
            # For better results with character portraits
            w, h = image.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            image = image.crop((left, top, left + min_dim, top + min_dim))
        
        return image
    
    def extract_embedding(self, image_path, crop_character=True):
        """
        Extract 768-dimensional character embedding from image.
        
        Args:
            image_path: Path to character image
            crop_character: Whether to preprocess/crop the image
            
        Returns:
            numpy array of shape (768,) - ready for drawatoon model
        """
        # Preprocess image
        image = self.preprocess_character_image(image_path, crop_character)
        
        # Apply model preprocessing
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            if hasattr(self.model, 'encode_image'):
                # Standard CLIP interface
                features = self.model.encode_image(image_tensor)
            else:
                # OpenCLIP interface
                features = self.model.encode_image(image_tensor)
        
        # Normalize features (important for consistent results)
        features = features / features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy array
        embedding = features.cpu().numpy().squeeze()
        
        # Verify correct dimensions
        assert embedding.shape == (768,), f"Expected (768,), got {embedding.shape}"
        
        return embedding
    
    def extract_multiple_embeddings(self, image_paths, crop_character=True):
        """Extract embeddings from multiple images"""
        embeddings = []
        for path in image_paths:
            embedding = self.extract_embedding(path, crop_character)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def save_embedding(self, embedding, save_path):
        """Save embedding to file"""
        np.save(save_path, embedding)
        print(f"Saved embedding to {save_path}")
    
    def load_embedding(self, embedding_path):
        """Load embedding from file"""
        return np.load(embedding_path)


class DrawatoonAdvancedEmbedder(DrawatoonCharacterEmbedder):
    """
    Advanced version with multiple strategies for optimal embeddings
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(device)
        self.face_detector = None
        self.load_face_detector()
    
    def load_face_detector(self):
        """Load face detector for better character cropping"""
        try:
            import cv2
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("✓ Loaded face detector for better character cropping")
        except Exception as e:
            print(f"Could not load face detector: {e}")
    
    def smart_character_crop(self, image_path):
        """Intelligent character cropping using face detection"""
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.face_detector is not None:
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the largest face
                areas = [w * h for (x, y, w, h) in faces]
                largest_face_idx = np.argmax(areas)
                x, y, w, h = faces[largest_face_idx]
                
                # Expand crop around face
                expand_factor = 1.5
                center_x, center_y = x + w // 2, y + h // 2
                new_w = int(w * expand_factor)
                new_h = int(h * expand_factor)
                
                x1 = max(0, center_x - new_w // 2)
                y1 = max(0, center_y - new_h // 2)
                x2 = min(image.shape[1], x1 + new_w)
                y2 = min(image.shape[0], y1 + new_h)
                
                cropped = image[y1:y2, x1:x2]
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                return cropped_pil
        
        # Fallback to center crop
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return super().preprocess_character_image(None, crop_character=True)
    
    def multi_view_embedding(self, image_paths, weights=None):
        """
        Create composite embedding from multiple character views.
        Useful when you have multiple images of the same character.
        """
        embeddings = []
        for path in image_paths:
            embedding = self.extract_embedding(path)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        if weights is None:
            weights = np.ones(len(embeddings)) / len(embeddings)
        
        # Weighted average of embeddings
        composite_embedding = np.average(embeddings, axis=0, weights=weights)
        
        # Renormalize
        composite_embedding = composite_embedding / np.linalg.norm(composite_embedding)
        
        return composite_embedding
    
    def optimize_embedding_for_target(self, initial_embedding, target_characteristics=None, 
                                    learning_rate=0.01, iterations=100):
        """
        Optimize embedding using gradient descent (requires target images/feedback).
        This is for advanced use cases where you want to fine-tune embeddings.
        """
        embedding = torch.tensor(initial_embedding, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([embedding], lr=learning_rate)
        
        for i in range(iterations):
            # You would implement your loss function here based on
            # generated images vs target characteristics
            # This is a placeholder for the optimization loop
            pass
        
        return embedding.detach().cpu().numpy()


def test_embedding_extraction():
    """Test the embedding extraction pipeline"""
    
    # Initialize embedder
    embedder = DrawatoonCharacterEmbedder()
    
    # Test with a sample image (you'll need to provide actual image path)
    test_image_path = "path/to/your/character_image.jpg"
    
    if Path(test_image_path).exists():
        print("Testing embedding extraction...")
        
        # Extract embedding
        embedding = embedder.extract_embedding(test_image_path)
        print(f"✓ Extracted embedding shape: {embedding.shape}")
        print(f"✓ Embedding dtype: {embedding.dtype}")
        print(f"✓ Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
        
        # Save embedding
        embedder.save_embedding(embedding, "test_character_embedding.npy")
        
        # Load and verify
        loaded_embedding = embedder.load_embedding("test_character_embedding.npy")
        assert np.allclose(embedding, loaded_embedding)
        print("✓ Save/load test passed")
        
        return embedding
    else:
        print("Please provide a valid image path to test")
        return None


def integrate_with_drawatoon(character_embedding):
    """
    Example of how to use the extracted embedding with drawatoon model.
    This shows the format expected by the model.
    """
    
    # Convert numpy array to torch tensor
    reference_embedding = torch.from_numpy(character_embedding).float()
    
    # The drawatoon model expects a list of embeddings for batch processing
    batch_reference_embeddings = [
        [reference_embedding]  # One character embedding per image in batch
    ]
    
    # Example usage in the model forward pass:
    """
    outputs = drawatoon_model(
        hidden_states=latents,
        encoder_hidden_states=text_embeddings,
        encoder_attention_mask=text_mask,
        text_bboxes=[[]],  # No text bboxes
        character_bboxes=[[(0.1, 0.1, 0.9, 0.9)]],  # Character bbox
        reference_embeddings=batch_reference_embeddings,  # Our extracted embedding!
        timestep=timestep
    )
    """
    
    print("✓ Character embedding ready for drawatoon model")
    print(f"  Shape: {reference_embedding.shape}")
    print(f"  Format: {type(reference_embedding)}")
    
    return batch_reference_embeddings


if __name__ == "__main__":
    # Run the test
    embedding = test_embedding_extraction()
    
    if embedding is not None:
        # Show how to integrate with drawatoon
        integrate_with_drawatoon(embedding)
        
        print("\n" + "="*50)
        print("SUCCESS: Character embedding pipeline ready!")
        print("="*50)
        print("\nNext steps:")
        print("1. Replace 'path/to/your/character_image.jpg' with actual character images")
        print("2. Extract embeddings for your characters")
        print("3. Use the embeddings with drawatoon model as shown above")
        print("4. Experiment with different character images to find optimal results")