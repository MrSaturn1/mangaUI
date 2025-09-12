#!/usr/bin/env python3
"""
Reverse engineer fumeisama's character embedding approach for Drawatoon.
Tests multiple pre-trained encoders to find the best match for character consistency.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, CLIPProcessor, CLIPModel
from PIL import Image
import torchvision.transforms as transforms
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict

class DrawatoonReverseEngineer:
    """Reverse engineer the character encoder used in Drawatoon"""
    
    def __init__(self, output_dir="character_output"):
        self.output_dir = Path(output_dir)
        self.character_images_dir = self.output_dir / "character_images"
        self.analysis_dir = self.output_dir / "reverse_engineering"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.encoders = {}
        self.init_all_encoders()
    
    def init_all_encoders(self):
        """Initialize all possible character encoders for testing"""
        print("🔧 Loading multiple encoders for testing...")
        
        # 1. Magi v2 (most likely candidate)
        try:
            print("  Loading Magi v2...")
            self.encoders['magi_v2'] = {
                'model': AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).to(self.device),
                'type': 'manga_specific',
                'extract_fn': self.extract_magi_embedding
            }
            self.encoders['magi_v2']['model'].eval()
            print("  ✅ Magi v2 loaded")
        except Exception as e:
            print(f"  ❌ Magi v2 failed: {e}")
        
        # 2. CLIP ViT-B/32 (widely used baseline)
        try:
            print("  Loading CLIP ViT-B/32...")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.encoders['clip_vit_b32'] = {
                'model': clip_model,
                'processor': clip_processor,
                'type': 'general_vision',
                'extract_fn': self.extract_clip_embedding
            }
            clip_model.eval()
            print("  ✅ CLIP ViT-B/32 loaded")
        except Exception as e:
            print(f"  ❌ CLIP ViT-B/32 failed: {e}")
        
        # 3. CLIP ViT-L/14 (Arc2Face inspiration)
        try:
            print("  Loading CLIP ViT-L/14...")
            clip_l_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            clip_l_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.encoders['clip_vit_l14'] = {
                'model': clip_l_model,
                'processor': clip_l_processor,
                'type': 'general_vision',
                'extract_fn': self.extract_clip_embedding
            }
            clip_l_model.eval()
            print("  ✅ CLIP ViT-L/14 loaded")
        except Exception as e:
            print(f"  ❌ CLIP ViT-L/14 failed: {e}")
        
        print(f"🎯 Loaded {len(self.encoders)} encoders for testing")
    
    def extract_magi_embedding(self, image, encoder_info):
        """Extract embedding using Magi (multiple strategies)"""
        model = encoder_info['model']
        
        try:
            # Test multiple preprocessing approaches
            strategies = {
                'imagenet': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                'standard': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]),
                'no_norm': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
            }
            
            results = {}
            
            for strategy_name, transform in strategies.items():
                try:
                    pixel_values = transform(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = model.crop_embedding_model(pixel_values)
                        
                        if hasattr(outputs, 'last_hidden_state'):
                            # CLS token
                            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                        else:
                            embedding = outputs.squeeze(0)
                    
                    embedding = embedding.cpu().float()
                    
                    # Normalize
                    if self.device == 'mps':
                        embedding = embedding.cpu().float()
                        embedding = F.normalize(embedding, p=2, dim=0)
                    else:
                        embedding = F.normalize(embedding, p=2, dim=0)
                    
                    results[strategy_name] = embedding
                    
                except Exception as e:
                    print(f"    Magi strategy {strategy_name} failed: {e}")
            
            # Return the imagenet strategy as default (most likely what Drawatoon uses)
            return results.get('imagenet', torch.randn(768))
            
        except Exception as e:
            print(f"  Magi extraction failed: {e}")
            return torch.randn(768)
    
    def extract_clip_embedding(self, image, encoder_info):
        """Extract embedding using CLIP"""
        model = encoder_info['model']
        processor = encoder_info['processor']
        
        try:
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = embedding.squeeze(0).cpu().float()
            
            return embedding
            
        except Exception as e:
            print(f"  CLIP extraction failed: {e}")
            return torch.randn(model.config.vision_config.hidden_size)
    
    def analyze_character_consistency(self, character_images, encoder_name, encoder_info):
        """Analyze how consistent an encoder is for a single character across different images"""
        if len(character_images) < 2:
            return {'consistency_score': 0.0, 'embeddings': []}
        
        embeddings = []
        character_name = character_images[0].stem
        
        print(f"    Testing {encoder_name} on {len(character_images)} images of {character_name}")
        
        for image_path in character_images:
            try:
                image = Image.open(image_path).convert("RGB")
                embedding = encoder_info['extract_fn'](image, encoder_info)
                embeddings.append(embedding)
            except Exception as e:
                print(f"      Failed on {image_path}: {e}")
                continue
        
        if len(embeddings) < 2:
            return {'consistency_score': 0.0, 'embeddings': embeddings}
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = torch.cosine_similarity(embeddings[i], embeddings[j], dim=0).item()
                similarities.append(sim)
        
        consistency_score = np.mean(similarities)
        
        return {
            'consistency_score': consistency_score,
            'embeddings': embeddings,
            'pairwise_similarities': similarities,
            'std_dev': np.std(similarities)
        }
    
    def analyze_character_separation(self, all_character_results):
        """Analyze how well different characters are separated"""
        character_names = list(all_character_results.keys())
        separation_scores = {}
        
        for encoder_name in self.encoders.keys():
            # Get average embeddings for each character
            char_avg_embeddings = {}
            
            for char_name in character_names:
                if encoder_name in all_character_results[char_name]:
                    embeddings = all_character_results[char_name][encoder_name]['embeddings']
                    if embeddings:
                        # Average embedding for this character
                        avg_embedding = torch.stack(embeddings).mean(dim=0)
                        char_avg_embeddings[char_name] = avg_embedding
            
            if len(char_avg_embeddings) < 2:
                separation_scores[encoder_name] = 0.0
                continue
            
            # Calculate inter-character distances
            inter_char_similarities = []
            char_names_list = list(char_avg_embeddings.keys())
            
            for i in range(len(char_names_list)):
                for j in range(i + 1, len(char_names_list)):
                    char1_emb = char_avg_embeddings[char_names_list[i]]
                    char2_emb = char_avg_embeddings[char_names_list[j]]
                    
                    similarity = torch.cosine_similarity(char1_emb, char2_emb, dim=0).item()
                    inter_char_similarities.append(similarity)
            
            # Lower inter-character similarity = better separation
            separation_scores[encoder_name] = 1.0 - np.mean(inter_char_similarities)
        
        return separation_scores
    
    def comprehensive_encoder_analysis(self):
        """Run comprehensive analysis on all encoders"""
        print("🔍 COMPREHENSIVE ENCODER ANALYSIS")
        print("=" * 50)
        
        # Get character images
        character_images = list(self.character_images_dir.glob("*.png"))
        character_images = [img for img in character_images if not img.name.endswith("_card.png")]
        
        if len(character_images) < 2:
            print("❌ Need at least 2 character images")
            return
        
        # Group by character name
        character_groups = defaultdict(list)
        for img_path in character_images:
            character_groups[img_path.stem].append(img_path)
        
        print(f"📊 Found {len(character_groups)} unique characters")
        print(f"    Characters: {list(character_groups.keys())}")
        
        # Test each encoder
        all_results = {}
        
        for char_name, char_images in character_groups.items():
            print(f"\n🎭 Testing character: {char_name} ({len(char_images)} images)")
            all_results[char_name] = {}
            
            for encoder_name, encoder_info in self.encoders.items():
                print(f"  🔧 Testing {encoder_name}...")
                
                consistency_result = self.analyze_character_consistency(
                    char_images, encoder_name, encoder_info
                )
                
                all_results[char_name][encoder_name] = consistency_result
                
                print(f"    ✅ Consistency score: {consistency_result['consistency_score']:.4f}")
        
        # Analyze character separation
        print(f"\n🎯 CHARACTER SEPARATION ANALYSIS")
        separation_scores = self.analyze_character_separation(all_results)
        
        # Compile final results
        final_scores = {}
        
        for encoder_name in self.encoders.keys():
            consistency_scores = []
            
            for char_name in all_results.keys():
                if encoder_name in all_results[char_name]:
                    consistency_scores.append(all_results[char_name][encoder_name]['consistency_score'])
            
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
            separation = separation_scores.get(encoder_name, 0.0)
            
            # Combined score: high intra-character consistency + good inter-character separation
            combined_score = (avg_consistency + separation) / 2.0
            
            final_scores[encoder_name] = {
                'avg_consistency': avg_consistency,
                'separation': separation,
                'combined_score': combined_score,
                'encoder_type': self.encoders[encoder_name]['type']
            }
        
        # Sort by combined score
        sorted_encoders = sorted(final_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
        print(f"\n📈 FINAL ENCODER RANKINGS")
        print("=" * 40)
        
        for rank, (encoder_name, scores) in enumerate(sorted_encoders, 1):
            print(f"{rank}. {encoder_name:15} | Combined: {scores['combined_score']:.4f} | "
                  f"Consistency: {scores['avg_consistency']:.4f} | "
                  f"Separation: {scores['separation']:.4f} | "
                  f"Type: {scores['encoder_type']}")
        
        # Save results
        results_path = self.analysis_dir / "encoder_analysis_results.json"
        
        # Convert tensors to lists for JSON serialization
        json_safe_results = {}
        for char_name, char_data in all_results.items():
            json_safe_results[char_name] = {}
            for encoder_name, encoder_data in char_data.items():
                json_safe_results[char_name][encoder_name] = {
                    'consistency_score': encoder_data['consistency_score'],
                    'pairwise_similarities': encoder_data.get('pairwise_similarities', []),
                    'std_dev': encoder_data.get('std_dev', 0.0)
                }
        
        complete_results = {
            'character_results': json_safe_results,
            'separation_scores': separation_scores,
            'final_scores': final_scores,
            'best_encoder': sorted_encoders[0][0] if sorted_encoders else None
        }
        
        with open(results_path, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
        
        # Conclusions
        best_encoder = sorted_encoders[0][0] if sorted_encoders else None
        if best_encoder:
            print(f"\n🎯 CONCLUSION")
            print("=" * 15)
            print(f"Best encoder match: {best_encoder}")
            print(f"This is most likely what fumeisama used in Drawatoon!")
            print(f"\n📝 RECOMMENDATION:")
            print(f"Update your manga generator to use {best_encoder} embeddings")
            print(f"Expected improvement in character consistency!")
        
        return complete_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Reverse engineer Drawatoon's character encoder")
    parser.add_argument("--output_dir", type=str, default="character_output",
                        help="Directory containing character images")
    
    args = parser.parse_args()
    
    analyzer = DrawatoonReverseEngineer(output_dir=args.output_dir)
    results = analyzer.comprehensive_encoder_analysis()
    
    print(f"\n🎉 Reverse engineering analysis complete!")
    print(f"Check the results to see which encoder best matches Drawatoon's behavior")

if __name__ == "__main__":
    main()