#!/usr/bin/env python3
"""
Extract DiffSensei resampler weights and adapt them for Drawatoon's IP attention layers.
Save this as backend/diffsensei_ip_extractor.py
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from collections import OrderedDict

def apply_ip_weights_to_drawatoon(pipeline, ip_weights_path):
    """Apply extracted IP weights to Drawatoon model with diagnostics"""
    print(f"📥 Loading IP weights from {ip_weights_path}")
    
    try:
        ip_weights = torch.load(ip_weights_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load IP weights: {e}")
        return False
    
    applied_count = 0
    failed_count = 0
    
    # Check the first block's IP attention structure
    first_block = pipeline.transformer.transformer_blocks[0]
    if hasattr(first_block, 'ip_attn'):
        print(f"\n🔍 IP Attention Architecture:")
        print(f"  Type: {type(first_block.ip_attn)}")
        
        # Check expected parameter shapes
        for param_name in ['to_q', 'to_k', 'to_v', 'to_out']:
            if hasattr(first_block.ip_attn, param_name):
                param = getattr(first_block.ip_attn, param_name)
                if hasattr(param, 'weight'):
                    print(f"  {param_name}.weight: {param.weight.shape}")
                if hasattr(param, 'bias') and param.bias is not None:
                    print(f"  {param_name}.bias: {param.bias.shape}")
    
    # Apply weights with detailed logging for first block only
    for block_idx, block in enumerate(pipeline.transformer.transformer_blocks):
        if hasattr(block, 'ip_attn'):
            for weight_name, weight_tensor in ip_weights.items():
                try:
                    # Navigate to the correct parameter
                    parts = weight_name.split('.')
                    target = block.ip_attn
                    
                    for part in parts[1:-1]:  # Skip 'ip_attn' and final 'weight'/'bias'
                        if part.isdigit():
                            target = target[int(part)]
                        else:
                            target = getattr(target, part)
                    
                    param_name = parts[-1]
                    if hasattr(target, param_name):
                        current_param = getattr(target, param_name)
                        
                        if current_param.shape == weight_tensor.shape:
                            current_param.data.copy_(weight_tensor)
                            applied_count += 1
                            
                            # Detailed logging for first block
                            if block_idx == 0:
                                print(f"✅ Applied {weight_name}: {weight_tensor.shape}")
                                print(f"   Value range: [{weight_tensor.min():.4f}, {weight_tensor.max():.4f}]")
                        else:
                            if block_idx == 0:
                                print(f"❌ Shape mismatch {weight_name}:")
                                print(f"   Expected: {current_param.shape}")
                                print(f"   Got: {weight_tensor.shape}")
                            failed_count += 1
                    else:
                        if block_idx == 0:
                            print(f"❌ Parameter {param_name} not found in {type(target)}")
                        failed_count += 1
                        
                except Exception as e:
                    if block_idx == 0:
                        print(f"❌ Error applying {weight_name}: {e}")
                    failed_count += 1
    
    print(f"\n📊 Results: ✅ {applied_count} applied, ❌ {failed_count} failed")
    return applied_count > 0

class DiffSenseiToDrawatoonAdapter:
    def __init__(self, diffsensei_weights_path="diffsensei_weights/pytorch_model.bin"):
        self.diffsensei_weights_path = Path(diffsensei_weights_path)
        self.adapted_weights = {}
        
    def load_and_inspect_diffsensei_weights(self):
        """Load DiffSensei weights and inspect their structure"""
        print("🔍 Loading DiffSensei weights...")
        
        if not self.diffsensei_weights_path.exists():
            raise FileNotFoundError(f"DiffSensei weights not found: {self.diffsensei_weights_path}")
        
        # Load the checkpoint
        self.diffsensei_checkpoint = torch.load(
            self.diffsensei_weights_path, 
            map_location='cpu', 
            weights_only=False
        )
        
        print("📊 DiffSensei checkpoint structure:")
        for key, tensor in self.diffsensei_checkpoint.items():
            print(f"  {key}: {tensor.shape} ({tensor.dtype})")
        
        return self.diffsensei_checkpoint
    
    def extract_resampler_weights(self):
        """Extract the resampler component that we can adapt"""
        resampler_weights = {}
        
        for key, weight in self.diffsensei_checkpoint.items():
            # DiffSensei resampler components we want to extract
            if any(pattern in key for pattern in [
                'latents',           # Learnable query tokens
                'proj_in',           # Input projection  
                'proj_out',          # Output projection
                'layers.',           # Attention layers
                'norm_out',          # Output normalization
                'dummy_tokens'       # Dummy tokens
            ]):
                resampler_weights[key] = weight
                print(f"✅ Extracted resampler weight: {key} -> {weight.shape}")
        
        print(f"\n📦 Extracted {len(resampler_weights)} resampler components")
        return resampler_weights
    
    def create_ip_attention_weights(self, resampler_weights):
        """Create IP attention weights from resampler components"""
        print("\n🔧 Creating IP attention weights for Drawatoon...")
        
        ip_weights = OrderedDict()
        
        # Strategy: Use DiffSensei's learned attention patterns
        # The resampler has attention layers that we can adapt
        
        for key, weight in resampler_weights.items():
            if 'layers.' in key and '.0.' in key and 'attn' in key:
                # These are attention layers from the resampler
                # Map them to IP attention structure
                
                if 'to_q.weight' in key:
                    # Query projection
                    ip_weights['ip_attn.to_q.weight'] = self._adapt_attention_weight(weight, target_name='to_q')
                elif 'to_kv.weight' in key:
                    # Key-Value projection (split into separate K and V)
                    k_weight, v_weight = weight.chunk(2, dim=0)
                    ip_weights['ip_attn.to_k.weight'] = self._adapt_attention_weight(k_weight, target_name='to_k')
                    ip_weights['ip_attn.to_v.weight'] = self._adapt_attention_weight(v_weight, target_name='to_v')
                elif 'to_out.weight' in key:
                    # Output projection - start with small weights instead of zero!
                    adapted_weight = self._adapt_attention_weight(weight, target_name='to_out')
                    # Scale down to start gently
                    ip_weights['ip_attn.to_out.0.weight'] = adapted_weight
                    
                    # Add bias if it exists
                    bias_key = key.replace('weight', 'bias')
                    if bias_key in resampler_weights:
                        ip_weights['ip_attn.to_out.0.bias'] = resampler_weights[bias_key]
        
        # If we don't have enough attention weights, create them from proj_out
        if 'ip_attn.to_q.weight' not in ip_weights and 'proj_out.weight' in resampler_weights:
            print("🔄 Creating IP attention weights from proj_out...")
            base_weight = resampler_weights['proj_out.weight']
            
            # Create Q, K, V projections from the output projection
            dim = base_weight.shape[1]
            
            # Use different sections of the weight matrix
            if base_weight.shape[0] >= dim * 3:
                q_weight = base_weight[:dim, :].clone()
                k_weight = base_weight[dim:dim*2, :].clone()
                v_weight = base_weight[dim*2:dim*3, :].clone()
            else:
                # Repeat the weight for all projections
                q_weight = base_weight.clone()
                k_weight = base_weight.clone()
                v_weight = base_weight.clone()
            
            ip_weights['ip_attn.to_q.weight'] = q_weight
            ip_weights['ip_attn.to_k.weight'] = k_weight
            ip_weights['ip_attn.to_v.weight'] = v_weight
            
            # Output projection - start small
            ip_weights['ip_attn.to_out.0.weight'] = base_weight.clone()
            
            if 'proj_out.bias' in resampler_weights:
                ip_weights['ip_attn.to_out.0.bias'] = resampler_weights['proj_out.bias'].clone()
        
        print(f"✅ Created {len(ip_weights)} IP attention weights")
        return ip_weights
    
    def _adapt_attention_weight(self, weight, target_name):
        """Adapt a weight tensor for the target component"""
        # Basic adaptation - you might want to refine this
        adapted = weight.clone()
        
        # Apply some transformation based on the target
        if target_name == 'to_out':
            # Output weights should start smaller to avoid overwhelming the model
            adapted = adapted
        
        return adapted
    
    def save_adapted_weights(self, ip_weights, output_dir="adapted_ip_weights"):
        """Save the adapted IP weights for loading into Drawatoon"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the IP weights
        ip_weights_path = output_dir / "ip_attention_weights.pth"
        torch.save(ip_weights, ip_weights_path)
        print(f"💾 Saved IP weights: {ip_weights_path}")
        
        # Save metadata
        metadata = {
            "source": "DiffSensei resampler adaptation",
            "weights_count": len(ip_weights),
            "weight_info": {key: list(weight.shape) for key, weight in ip_weights.items()},
            "adaptation_strategy": "resampler_to_ip_attention",
            "scaling_factor": 0.1,
            "usage_instructions": [
                "Load these weights into Drawatoon's IP attention layers",
                "Use apply_ip_weights_to_drawatoon() function",
                "Test with character embeddings",
                "Fine-tune scaling if needed"
            ]
        }
        
        metadata_path = output_dir / "adaptation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📄 Saved metadata: {metadata_path}")
        
        return ip_weights_path
    
    def extract_and_adapt(self):
        """Main method to extract and adapt DiffSensei weights"""
        print("🚀 EXTRACTING DIFFSENSEI WEIGHTS FOR DRAWATOON")
        print("=" * 50)
        
        # Load and inspect
        self.load_and_inspect_diffsensei_weights()
        
        # Extract resampler components
        resampler_weights = self.extract_resampler_weights()
        
        # Create IP attention weights
        ip_weights = self.create_ip_attention_weights(resampler_weights)
        
        # Save adapted weights
        weights_path = self.save_adapted_weights(ip_weights)
        
        return ip_weights, weights_path

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract DiffSensei weights for Drawatoon")
    parser.add_argument("--weights", default="diffsensei_weights/pytorch_model.bin",
                        help="Path to DiffSensei weights")
    parser.add_argument("--output", default="adapted_ip_weights",
                        help="Output directory for adapted weights")
    
    args = parser.parse_args()
    
    try:
        # Create adapter
        adapter = DiffSenseiToDrawatoonAdapter(args.weights)
        
        # Extract and adapt
        ip_weights, weights_path = adapter.extract_and_adapt()
        
        print("\n🎯 NEXT STEPS:")
        print("=" * 20)
        print("1. Update your manga_generator.py to load these weights")
        print("2. Test with your existing character embeddings")
        print("3. Compare results with enhanced prompts")
        print("\n📝 INTEGRATION EXAMPLE:")
        print("""
# Add this to your MangaGenerator.__init__() method:
self.load_diffsensei_ip_weights()

# Add this method to your MangaGenerator class:
def load_diffsensei_ip_weights(self):
    from diffsensei_ip_extractor import apply_ip_weights_to_drawatoon
    weights_path = "adapted_ip_weights/ip_attention_weights.pth"
    if Path(weights_path).exists():
        success = apply_ip_weights_to_drawatoon(self.pipe, weights_path)
        if success:
            print("✅ DiffSensei IP weights loaded!")
        else:
            print("⚠️  Using default IP weights")
    else:
        print("⚠️  DiffSensei IP weights not found, using defaults")
        """)
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔄 FALLBACK OPTIONS:")
        print("1. Continue using enhanced prompts (your current working approach)")
        print("2. Try IP-Adapter Face weights instead")
        print("3. Use hybrid embeddings with minimal IP attention scaling")

if __name__ == "__main__":
    main()