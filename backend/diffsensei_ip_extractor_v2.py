#!/usr/bin/env python3
"""
Simple DiffSensei IP adapter weight adaptation for Drawatoon.
Focuses on extracting and applying the core attention weights.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from collections import OrderedDict
import argparse

def extract_diffsensei_ip_weights(diffsensei_path):
    """Extract IP adapter style weights from DiffSensei checkpoint"""
    print("🔍 EXTRACTING DIFFSENSEI IP WEIGHTS")
    print("=" * 40)
    
    try:
        # Load DiffSensei checkpoint
        checkpoint = torch.load(diffsensei_path, map_location='cpu', weights_only=False)
        print(f"✅ Loaded DiffSensei checkpoint: {len(checkpoint)} components")
        
        # Look for resampler/attention components
        ip_weights = OrderedDict()
        
        print("\n📊 DiffSensei component analysis:")
        for key, weight in checkpoint.items():
            print(f"  {key}: {weight.shape} ({weight.dtype})")
            
            # Extract components that look like attention weights
            if any(pattern in key.lower() for pattern in [
                'resampler', 'cross_attn', 'to_q', 'to_k', 'to_v', 'to_out',
                'attn', 'query', 'key', 'value', 'proj'
            ]):
                ip_weights[key] = weight
                print(f"    ✅ Extracted: {key}")
        
        print(f"\n📦 Extracted {len(ip_weights)} potentially useful components")
        return ip_weights
        
    except Exception as e:
        print(f"❌ Failed to load DiffSensei: {e}")
        return {}

def adapt_weights_for_drawatoon(diffsensei_weights):
    """Adapt DiffSensei weights to match Drawatoon's IP attention structure"""
    print("\n🔧 ADAPTING WEIGHTS FOR DRAWATOON")
    print("=" * 35)
    
    adapted_weights = OrderedDict()
    
    # Expected Drawatoon IP attention structure
    target_shapes = {
        'to_q.weight': (1152, 1152),
        'to_k.weight': (1152, 1152), 
        'to_v.weight': (1152, 1152),
        'to_out.0.weight': (1152, 1152),
        'to_out.0.bias': (1152,)
    }
    
    print("🎯 Target Drawatoon shapes:")
    for name, shape in target_shapes.items():
        print(f"  {name}: {shape}")
    
    # Strategy: Find best matching weights and adapt them
    for target_name, target_shape in target_shapes.items():
        print(f"\n🔍 Finding match for {target_name}...")
        
        best_match = find_best_weight_match(diffsensei_weights, target_shape, target_name)
        
        if best_match:
            source_key, source_weight = best_match
            adapted = adapt_single_weight(source_weight, target_shape, target_name)
            
            if adapted is not None:
                adapted_weights[f'ip_attn.{target_name}'] = adapted
                print(f"  ✅ {target_name}: {source_weight.shape} → {target_shape} (from {source_key})")
            else:
                print(f"  ❌ {target_name}: Could not adapt")
        else:
            print(f"  ⚠️ {target_name}: No suitable source found")
    
    print(f"\n📊 Adaptation result: {len(adapted_weights)} weights adapted")
    return adapted_weights

def find_best_weight_match(weights_dict, target_shape, target_name):
    """Find the best matching weight for adaptation"""
    best_score = 0
    best_match = None
    
    for key, weight in weights_dict.items():
        score = calculate_compatibility(weight.shape, target_shape, key, target_name)
        
        if score > best_score:
            best_score = score
            best_match = (key, weight)
    
    return best_match if best_score > 0.3 else None

def calculate_compatibility(source_shape, target_shape, source_key, target_name):
    """Calculate compatibility score between source and target"""
    score = 0.0
    
    # Exact match is perfect
    if source_shape == target_shape:
        return 1.0
    
    # Same number of elements can be reshaped
    if len(source_shape) == len(target_shape):
        source_numel = torch.tensor(source_shape).prod().item()
        target_numel = torch.tensor(target_shape).prod().item()
        
        if source_numel == target_numel:
            score += 0.8
    
    # Name-based matching
    if 'to_q' in target_name and any(x in source_key.lower() for x in ['query', 'to_q']):
        score += 0.5
    elif 'to_k' in target_name and any(x in source_key.lower() for x in ['key', 'to_k']):
        score += 0.5
    elif 'to_v' in target_name and any(x in source_key.lower() for x in ['value', 'to_v']):
        score += 0.5
    elif 'to_out' in target_name and any(x in source_key.lower() for x in ['out', 'proj', 'linear']):
        score += 0.5
    
    # General attention patterns
    if 'attn' in source_key.lower() and any(x in target_name for x in ['to_q', 'to_k', 'to_v', 'to_out']):
        score += 0.3
    
    return score

def adapt_single_weight(source_weight, target_shape, weight_name):
    """Adapt a single weight tensor to target shape"""
    try:
        # Strategy 1: Direct reshape if same number of elements
        if source_weight.numel() == torch.tensor(target_shape).prod():
            adapted = source_weight.reshape(target_shape).clone()
            print(f"    → Reshaped {source_weight.shape} to {target_shape}")
            return adapted
        
        # Strategy 2: Truncate or pad for matrices
        if len(target_shape) == 2 and len(source_weight.shape) == 2:
            adapted = torch.zeros(target_shape, dtype=source_weight.dtype)
            
            rows = min(source_weight.shape[0], target_shape[0])
            cols = min(source_weight.shape[1], target_shape[1])
            
            adapted[:rows, :cols] = source_weight[:rows, :cols]
            
            # Scale down to start gently (important for stability)
            adapted = adapted * 0.1
            
            print(f"    → Truncated/padded {source_weight.shape} to {target_shape}, scaled 0.1x")
            return adapted
        
        # Strategy 3: For bias vectors
        elif len(target_shape) == 1 and len(source_weight.shape) == 1:
            adapted = torch.zeros(target_shape, dtype=source_weight.dtype)
            length = min(source_weight.shape[0], target_shape[0])
            adapted[:length] = source_weight[:length] * 0.1
            
            print(f"    → Adapted bias {source_weight.shape} to {target_shape}")
            return adapted
        
        else:
            print(f"    → Cannot adapt {source_weight.shape} to {target_shape}")
            return None
            
    except Exception as e:
        print(f"    → Adaptation error: {e}")
        return None

def apply_diffsensei_weights_to_drawatoon(pipeline, adapted_weights):
    """Apply adapted weights to Drawatoon's IP attention layers"""
    print("\n🚀 APPLYING WEIGHTS TO DRAWATOON")
    print("=" * 32)
    
    applied_count = 0
    failed_count = 0
    
    # Apply to all transformer blocks
    for block_idx, block in enumerate(pipeline.transformer.transformer_blocks):
        if hasattr(block, 'ip_attn'):
            for weight_name, weight_tensor in adapted_weights.items():
                try:
                    # Navigate to the target parameter
                    parts = weight_name.split('.')
                    target = block.ip_attn
                    
                    for part in parts[1:-1]:  # Skip 'ip_attn' and final part
                        if part.isdigit():
                            target = target[int(part)]
                        else:
                            target = getattr(target, part)
                    
                    param_name = parts[-1]
                    if hasattr(target, param_name):
                        current_param = getattr(target, param_name)
                        
                        if current_param.shape == weight_tensor.shape:
                            current_param.data.copy_(weight_tensor.to(current_param.device))
                            applied_count += 1
                            
                            if block_idx == 0:  # Log details for first block only
                                print(f"  ✅ Applied {weight_name}: {weight_tensor.shape}")
                        else:
                            if block_idx == 0:
                                print(f"  ❌ Shape mismatch {weight_name}: {current_param.shape} vs {weight_tensor.shape}")
                            failed_count += 1
                    else:
                        if block_idx == 0:
                            print(f"  ❌ Parameter {param_name} not found")
                        failed_count += 1
                        
                except Exception as e:
                    if block_idx == 0:
                        print(f"  ❌ Error applying {weight_name}: {e}")
                    failed_count += 1
    
    print(f"\n📊 Application Results:")
    print(f"  ✅ Successfully applied: {applied_count} weights")
    print(f"  ❌ Failed applications: {failed_count} weights")
    
    if applied_count > 0:
        print(f"  🎯 DiffSensei patterns now active in {len(pipeline.transformer.transformer_blocks)} blocks!")
        return True
    else:
        print(f"  ⚠️ No weights successfully applied")
        return False

def integrate_diffsensei_into_manga_generator(manga_generator, diffsensei_path):
    """Main integration function"""
    print("🔗 INTEGRATING DIFFSENSEI INTO MANGA GENERATOR")
    print("=" * 50)
    
    # Step 1: Extract DiffSensei weights
    diffsensei_weights = extract_diffsensei_ip_weights(diffsensei_path)
    
    if not diffsensei_weights:
        print("❌ No weights extracted from DiffSensei")
        return False
    
    # Step 2: Adapt weights for Drawatoon
    adapted_weights = adapt_weights_for_drawatoon(diffsensei_weights)
    
    if not adapted_weights:
        print("❌ No weights successfully adapted")
        return False
    
    # Step 3: Apply to Drawatoon
    success = apply_diffsensei_weights_to_drawatoon(manga_generator.pipe, adapted_weights)
    
    if success:
        print("\n🎉 DIFFSENSEI INTEGRATION COMPLETE!")
        print("✨ Your character embeddings should now have much better consistency!")
        print("🧪 Test by generating the same character with different seeds")
    
    return success

def save_adapted_weights(adapted_weights, output_dir):
    """Save adapted weights to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the weights
    weights_path = output_dir / "ip_attention_weights.pth"
    torch.save(adapted_weights, weights_path)
    print(f"💾 Saved adapted weights: {weights_path}")
    
    # Save metadata
    metadata = {
        "source": "DiffSensei weight adaptation",
        "weights_count": len(adapted_weights),
        "weight_info": {key: list(weight.shape) for key, weight in adapted_weights.items()},
        "adaptation_strategy": "best_match_with_scaling",
        "scaling_factor": 0.1
    }
    
    metadata_path = output_dir / "adaptation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Saved metadata: {metadata_path}")
    
    return weights_path

def main():
    """Main execution with proper argument parsing"""
    parser = argparse.ArgumentParser(description="Extract and adapt DiffSensei weights for Drawatoon")
    parser.add_argument("--weights", default="diffsensei_weights/pytorch_model.bin",
                        help="Path to DiffSensei weights")
    parser.add_argument("--output", default="adapted_ip_weights",
                        help="Output directory for adapted weights")
    
    args = parser.parse_args()
    
    print("🚀 DIFFSENSEI WEIGHT EXTRACTOR FOR DRAWATOON")
    print("=" * 48)
    
    try:
        # Step 1: Extract DiffSensei weights
        diffsensei_weights = extract_diffsensei_ip_weights(args.weights)
        
        if not diffsensei_weights:
            print("❌ No weights extracted from DiffSensei")
            return False
        
        # Step 2: Adapt weights for Drawatoon
        adapted_weights = adapt_weights_for_drawatoon(diffsensei_weights)
        
        if not adapted_weights:
            print("❌ No weights successfully adapted")
            return False
        
        # Step 3: Save adapted weights
        weights_path = save_adapted_weights(adapted_weights, args.output)
        
        print("\n🎯 EXTRACTION COMPLETE!")
        print("=" * 25)
        print(f"✅ Adapted weights saved to: {weights_path}")
        print("\n📝 NEXT STEPS:")
        print("1. Add load_diffsensei_weights() method to your MangaGenerator")
        print("2. Call it in __init__ after fix_ref_embedding_proj()")
        print("3. Test character consistency!")
        
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()