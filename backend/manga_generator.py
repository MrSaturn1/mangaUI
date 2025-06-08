import os
import re
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from diffusers import PixArtSigmaPipeline
from torch.nn.utils.rnn import pad_sequence
import torch.serialization
from numpy._core.multiarray import _reconstruct
from diffusers.models.modeling_outputs import Transformer2DModelOutput

class PanelEstimator:
    """Estimates appropriate number of panels for a scene based on content analysis"""
    
    def __init__(self):
        # Manga panel allocation rules
        self.panel_rules = {
            'establishing_shot': 1,  # Always 1 panel to establish scene setting
            'dialogue_exchange': 1.5,  # Approximately 1.5 panels per dialogue exchange
            'action_sentence': 0.7,  # About 0.7 panels per significant action sentence
            'location_change': 1,  # 1 panel for each location change within scene
            'character_intro': 0.5,  # 0.5 panels when a new character appears
            'max_per_scene': 12,  # Cap on panels per scene
            'min_per_scene': 2,  # Minimum panels per scene
        }
    
    def count_dialogue_exchanges(self, scene_elements):
        """Count dialogue exchanges (speaker changes) in scene"""
        exchanges = 0
        prev_speaker = None
        
        for element in scene_elements:
            if element['type'] == 'dialogue':
                speaker = element['character']
                if speaker != prev_speaker:
                    exchanges += 1
                    prev_speaker = speaker
        
        return exchanges
    
    def count_action_sentences(self, scene_elements):
        """Count significant action sentences in scene"""
        sentence_count = 0
        
        for element in scene_elements:
            if element['type'] == 'action':
                text = element['text']
                # Split into sentences and count
                sentences = re.split(r'[.!?]+', text)
                # Filter out empty or very short sentences
                sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
                sentence_count += len(sentences)
        
        return sentence_count
    
    def count_unique_characters(self, scene_elements):
        """Count unique characters mentioned or speaking in scene"""
        characters = set()
        
        for element in scene_elements:
            if element['type'] == 'dialogue':
                characters.add(element['character'])
            
        return len(characters)
    
    def estimate_panels(self, scene):
        """Estimate appropriate number of panels for a scene"""
        elements = scene['elements']
        
        # Base panel for establishing the scene
        panel_count = self.panel_rules['establishing_shot']
        
        # Add panels for dialogue exchanges
        dialogue_exchanges = self.count_dialogue_exchanges(elements)
        panel_count += dialogue_exchanges * self.panel_rules['dialogue_exchange']
        
        # Add panels for action sentences
        action_sentences = self.count_action_sentences(elements)
        panel_count += action_sentences * self.panel_rules['action_sentence']
        
        # Add panels for character introductions
        unique_characters = self.count_unique_characters(elements)
        panel_count += min(unique_characters, 3) * self.panel_rules['character_intro']
        
        # Round to nearest whole number
        panel_count = round(panel_count)
        
        # Apply min/max constraints
        panel_count = max(panel_count, self.panel_rules['min_per_scene'])
        panel_count = min(panel_count, self.panel_rules['max_per_scene'])
        
        return panel_count

class ScreenplayParser:
    def __init__(self, screenplay_path, character_data_path):
        with open(screenplay_path, 'r', encoding='utf-8') as f:
            self.screenplay_text = f.read()
            
        with open(character_data_path, 'r', encoding='utf-8') as f:
            self.character_data = json.load(f)
            
        # Extract named characters for easy reference
        self.named_characters = []
        for character in self.character_data:
            if character.get("character_type") == "NAMED":
                self.named_characters.append(character["name"])
        
        # Initialize the panel estimator
        self.panel_estimator = PanelEstimator()
        
        # Regex patterns for screenplay elements
        self.scene_header_pattern = r'(INT\.|EXT\.|INT\/EXT\.)\s+([\w\s\-\'\,\.\!\?\&\;\:]+)\s*\-\s*([\w\s\-\'\,\.\!\?\&\;\:]+)'
        self.character_name_pattern = r'^\s*([A-Z][A-Z\s\(\)\']+)(?:\s*\(.*\))?$'
        self.dialogue_pattern = r'^\s{10,}([\w\s\-\'\,\.\!\?\&\;\:\"\…\[\]]+)$'
        self.action_pattern = r'^(?!\s{10,})([\w\s\-\'\,\.\!\?\&\;\:\"\…\[\]]+)$'
    
    def parse(self):
        """Parse the screenplay into a structured format"""
        lines = self.screenplay_text.split('\n')
        
        scenes = []
        current_scene = {}
        current_dialogue = None
        
        for line in lines:
            # Check for scene header
            scene_match = re.match(self.scene_header_pattern, line)
            if scene_match:
                if current_scene:
                    scenes.append(current_scene)
                
                interior_exterior, location, time = scene_match.groups()
                current_scene = {
                    'location': location.strip(),
                    'time': time.strip(),
                    'interior_exterior': interior_exterior,
                    'elements': []
                }
                continue
            
            # Check for character names
            char_match = re.match(self.character_name_pattern, line)
            if char_match:
                character_name = char_match.group(1).strip()
                # Check if this is actually a character name in our list
                if any(char == character_name for char in self.named_characters):
                    current_dialogue = {
                        'character': character_name,
                        'dialogue': [],
                        'type': 'dialogue'
                    }
                    current_scene['elements'].append(current_dialogue)
                continue
            
            # Check for dialogue
            if current_dialogue and re.match(self.dialogue_pattern, line):
                dialogue_text = re.match(self.dialogue_pattern, line).group(1)
                current_dialogue['dialogue'].append(dialogue_text.strip())
                continue
                
            # Check for action lines (anything that's not dialogue or character names)
            action_match = re.match(self.action_pattern, line)
            if action_match and line.strip():
                action_text = action_match.group(1).strip()
                if action_text:  # Skip empty lines
                    current_scene['elements'].append({
                        'type': 'action',
                        'text': action_text
                    })
        
        # Add the last scene
        if current_scene:
            scenes.append(current_scene)
            
        # Estimate panel count for each scene
        for scene in scenes:
            estimated_panels = self.panel_estimator.estimate_panels(scene)
            scene['estimated_panels'] = estimated_panels
            
        return scenes
    
    def get_character_descriptions(self, character_name):
        """Get all descriptions for a specific character"""
        for character in self.character_data:
            if character["name"] == character_name:
                return character.get("descriptions", [])
        return []
    
    def analyze_panel_content(self, scene_element):
        """Determine what should be in a panel based on the scene element"""
        if scene_element['type'] == 'dialogue':
            character = scene_element['character']
            descriptions = self.get_character_descriptions(character)
            
            # Combine descriptions to create a character appearance prompt
            character_prompt = f"{character}, " + ", ".join(descriptions[:3])
            
            return {
                'type': 'dialogue',
                'character': character,
                'character_prompt': character_prompt,
                'dialogue': " ".join(scene_element['dialogue']),
                'characters_present': [character]
            }
        elif scene_element['type'] == 'action':
            text = scene_element['text']
            
            # Extract mentioned characters from action text
            mentioned_characters = []
            for character in self.named_characters:
                if character in text:
                    mentioned_characters.append(character)
            
            return {
                'type': 'action',
                'text': text,
                'characters_present': mentioned_characters
            }
        
        return None

    def generate_panel_prompts(self, scenes, start_scene=0, end_scene=None):
        """Convert scenes to panel prompts for image generation using dynamic panel count"""
        all_panels = []
        
        # Handle scene range
        if end_scene is None:
            end_scene = len(scenes)
        else:
            end_scene = min(end_scene, len(scenes))
        
        for scene_index in range(start_scene, end_scene):
            scene = scenes[scene_index]
            location = scene['location']
            time = scene['time']
            interior_exterior = scene['interior_exterior']
            
            # Get estimated panel count for this scene
            max_panels = scene['estimated_panels']
            print(f"Scene {scene_index}: {location} - {time} - Estimated panels: {max_panels}")
            
            # Basic setting description to include in prompts
            setting_prompt = f"{interior_exterior} {location}, {time}, manga panel"
            
            # Initialize variables for panel creation
            panel_count = 0
            current_panel = {
                'setting': setting_prompt,
                'elements': [],
                'characters': set(),
                'scene_index': scene_index
            }
            
            # Track whether we've created an establishing shot
            establishing_shot_created = False
            
            # First, create an establishing shot if there are enough elements
            if len(scene['elements']) > 2 and not establishing_shot_created:
                # Look for the first action element to use for establishing shot
                for element in scene['elements']:
                    if element['type'] == 'action':
                        panel_content = self.analyze_panel_content(element)
                        if panel_content:
                            current_panel['elements'].append(panel_content)
                            if 'characters_present' in panel_content:
                                for character in panel_content['characters_present']:
                                    current_panel['characters'].add(character)
                            
                            # Add the establishing shot panel
                            all_panels.append(current_panel)
                            panel_count += 1
                            establishing_shot_created = True
                            
                            # Reset panel for the next one
                            current_panel = {
                                'setting': setting_prompt,
                                'elements': [],
                                'characters': set(),
                                'scene_index': scene_index
                            }
                            break
            
            # Variables to track dialogue flow
            prev_speaker = None
            dialogue_elements = []
            
            # Process the remaining elements
            for element in scene['elements']:
                panel_content = self.analyze_panel_content(element)
                
                if not panel_content:
                    continue
                
                # Special handling for dialogue
                if panel_content['type'] == 'dialogue':
                    current_speaker = panel_content['character']
                    
                    # Start a new panel if speaker changes
                    if prev_speaker is not None and current_speaker != prev_speaker:
                        # Add the accumulated dialogue to the current panel
                        if dialogue_elements:
                            for dialogue_element in dialogue_elements:
                                current_panel['elements'].append(dialogue_element)
                                if 'characters_present' in dialogue_element:
                                    for character in dialogue_element['characters_present']:
                                        current_panel['characters'].add(character)
                            
                            # Add the panel if it has elements
                            if current_panel['elements']:
                                all_panels.append(current_panel)
                                panel_count += 1
                            
                            # Reset for next panel
                            current_panel = {
                                'setting': setting_prompt,
                                'elements': [],
                                'characters': set(),
                                'scene_index': scene_index
                            }
                            dialogue_elements = []
                    
                    # Add to dialogue elements
                    dialogue_elements.append(panel_content)
                    prev_speaker = current_speaker
                else:
                    # For action elements
                    
                    # If we have accumulated dialogue, add it first
                    if dialogue_elements:
                        for dialogue_element in dialogue_elements:
                            current_panel['elements'].append(dialogue_element)
                            if 'characters_present' in dialogue_element:
                                for character in dialogue_element['characters_present']:
                                    current_panel['characters'].add(character)
                        
                        dialogue_elements = []
                    
                    # Add action to panel
                    current_panel['elements'].append(panel_content)
                    if 'characters_present' in panel_content:
                        for character in panel_content['characters_present']:
                            current_panel['characters'].add(character)
                    
                    # If panel is getting full or we're parsing a significant action
                    if len(current_panel['elements']) >= 2:
                        # Add the panel
                        all_panels.append(current_panel)
                        panel_count += 1
                        
                        # Reset for next panel
                        current_panel = {
                            'setting': setting_prompt,
                            'elements': [],
                            'characters': set(),
                            'scene_index': scene_index
                        }
                
                # Check if we've reached the panel limit for this scene
                if panel_count >= max_panels:
                    break
            
            # Add any remaining elements
            # First add any accumulated dialogue
            if dialogue_elements:
                for dialogue_element in dialogue_elements:
                    current_panel['elements'].append(dialogue_element)
                    if 'characters_present' in dialogue_element:
                        for character in dialogue_element['characters_present']:
                            current_panel['characters'].add(character)
            
            # Add the final panel if it has elements
            if current_panel['elements']:
                all_panels.append(current_panel)
        
        return all_panels

class MangaGenerator:
    def __init__(self, model_path, character_data_path, character_embedding_path, output_dir):
        # Load character data
        with open(character_data_path, 'r', encoding='utf-8') as f:
            self.character_data = json.load(f)

        self.character_data_path = character_data_path
        
        # Load character embeddings (the mapping of characters to their generated images)
        if os.path.exists(character_embedding_path):
            with open(character_embedding_path, 'r') as f:
                self.character_embeddings = json.load(f)
        else:
            print(f"Warning: Character embedding file not found at {character_embedding_path}")
            self.character_embeddings = {}
            
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.panels_dir = self.output_dir / "panels"
        self.pages_dir = self.output_dir / "pages"
        self.panels_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        
        # Load character embeddings map
        self.load_character_embeddings()
        
        # Initialize the model pipeline
        self.init_pipeline()
        
        # Initialize the upscaler
        self.init_upscaler()
    
    def init_pipeline(self):
        """Initialize the Drawatoon pipeline"""
        # Use PixArtSigmaPipeline as that's what Drawatoon is based on
        print("Loading model pipeline...")
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16
        )
        
        # We need to patch the transformer's forward method just like in CharacterGenerator
        def modified_forward(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
            ip_hidden_states=None,
            ip_attention_mask=None,
            text_bboxes=None,
            character_bboxes=None,
            reference_embeddings=None,
            cfg_on_10_percent=False,
            timestep=None,
            added_cond_kwargs=None,
            cross_attention_kwargs=None,
            return_dict=True,
        ):
            """
            The modified forward method that ensures the XOR condition is properly handled.
            """
            # Check for stored parameters first (our addition)
            if text_bboxes is None and hasattr(self, '_temp_text_bboxes'):
                text_bboxes = self._temp_text_bboxes
            if character_bboxes is None and hasattr(self, '_temp_character_bboxes'):
                character_bboxes = self._temp_character_bboxes
            if reference_embeddings is None and hasattr(self, '_temp_reference_embeddings'):
                reference_embeddings = self._temp_reference_embeddings
            
            # Debug info
            print(f"Modified forward called with:")
            print(f"ip_hidden_states: {'Present' if ip_hidden_states is not None else 'None'}")
            print(f"text_bboxes: {'Present' if text_bboxes is not None else 'None'}")
            print(f"character_bboxes: {'Present' if character_bboxes is not None else 'None'}")
            print(f"reference_embeddings: {'Present' if reference_embeddings is not None else 'None'}")

            batch_size = len(hidden_states)
            heights = [h.shape[-2] // self.config.patch_size for h in hidden_states]
            widths = [w.shape[-1] // self.config.patch_size for w in hidden_states]
            
            # When using direct void embedding, ensure other params are None
            if ip_hidden_states is not None:
                # Force these to be None
                text_bboxes = None
                character_bboxes = None
                reference_embeddings = None

            if ip_hidden_states is None and text_bboxes is None and character_bboxes is None and reference_embeddings is None:
                # Directly use the void embedding
                void_embed = self.ip_adapter.void_ip_embed.weight
                ip_hidden_states = void_embed.unsqueeze(0).expand(batch_size, 1, -1).to(hidden_states.device)
                ip_attention_mask = torch.ones((batch_size, 1), device=hidden_states.device, dtype=torch.bool)
            
            # For the original code path, if we're using bboxes
            if ip_hidden_states is None and (text_bboxes is not None or character_bboxes is not None or reference_embeddings is not None):
                # Let the IP adapter generate the embeddings
                print(f"Calling IP adapter with:")
                print(f"  text_bboxes: {text_bboxes}")
                print(f"  character_bboxes: {character_bboxes}")
                if reference_embeddings:
                    print(f"  reference_embeddings shapes: {[[emb.shape if hasattr(emb, 'shape') else 'N/A' for emb in batch] for batch in reference_embeddings]}")
                try:
                    ip_hidden_states, ip_attention_mask = self.ip_adapter(text_bboxes, character_bboxes, reference_embeddings, cfg_on_10_percent)
                    print(f"IP adapter succeeded, output shape: {ip_hidden_states.shape}")
                    
                    # Handle CFG: if batch_size > 1, duplicate IP adapter outputs
                    if batch_size > ip_hidden_states.shape[0]:
                        print(f"Duplicating IP adapter outputs for CFG: {ip_hidden_states.shape[0]} -> {batch_size}")
                        # Duplicate for CFG (typically batch_size=2 for negative+positive)
                        ip_hidden_states = ip_hidden_states.repeat(batch_size, 1, 1)
                        ip_attention_mask = ip_attention_mask.repeat(batch_size, 1)
                        print(f"After duplication: ip_hidden_states shape: {ip_hidden_states.shape}")
                        
                except Exception as e:
                    print(f"IP adapter failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise e
            
            # 1. Input
            
            hidden_states = [self.pos_embed(hs[None])[0] for hs in hidden_states]
            attention_mask = [torch.ones(x.shape[0]) for x in hidden_states]
            hidden_states = pad_sequence(hidden_states, batch_first=True)
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0).bool().to(hidden_states.device)
            original_attention_mask = attention_mask

            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
            
            # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

            if self.caption_projection is not None:
                encoder_hidden_states = self.caption_projection(encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

            # 2. Blocks
            for i, block in enumerate(self.transformer_blocks):
                print(f"Processing transformer block {i}, hidden_states shape: {hidden_states.shape}")
                if ip_hidden_states is not None:
                    print(f"  ip_hidden_states shape: {ip_hidden_states.shape}")
                
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    try:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            attention_mask,
                            encoder_hidden_states,
                            encoder_attention_mask,
                            ip_hidden_states,
                            ip_attention_mask,
                            timestep,
                            cross_attention_kwargs,
                            None,
                            **ckpt_kwargs,
                        )
                    except Exception as e:
                        print(f"Error in transformer block {i} with checkpoint: {e}")
                        raise e
                else:
                    try:
                        hidden_states = block(
                            hidden_states,
                            attention_mask=attention_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            ip_hidden_states=ip_hidden_states,
                            ip_attention_mask=ip_attention_mask,
                            timestep=timestep,
                            cross_attention_kwargs=cross_attention_kwargs,
                            class_labels=None,
                        )
                    except Exception as e:
                        print(f"Error in transformer block {i}: {e}")
                        print(f"  hidden_states shape: {hidden_states.shape}")
                        print(f"  ip_hidden_states shape: {ip_hidden_states.shape if ip_hidden_states is not None else 'None'}")
                        print(f"  encoder_hidden_states shape: {encoder_hidden_states.shape}")
                        import traceback
                        traceback.print_exc()
                        raise e
                
                # Only print for first few blocks to avoid spam
                if i < 3:
                    print(f"  Block {i} completed, output shape: {hidden_states.shape}")
                elif i == 3:
                    print(f"  (Suppressing further block output for brevity...)")

            # 3. Output
            shift, scale = (
                self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
            ).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

            # unpatchify
            outputs = []
            for idx, (height, width) in enumerate(zip(heights, widths)):
                _hidden_state = hidden_states[idx][original_attention_mask[idx]].reshape(
                    shape=(height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
                )
                _hidden_state = torch.einsum("hwpqc->chpwq", _hidden_state)
                outputs.append(_hidden_state.reshape(
                    shape=(self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
                ))
            
            if len(set([x.shape for x in outputs])) == 1:
                outputs = torch.stack(outputs)

            if not return_dict:
                return (outputs,)

            return Transformer2DModelOutput(sample=outputs)
        
        # Replace the transformer's forward method with our modified version
        self.pipe.transformer.forward = modified_forward.__get__(self.pipe.transformer, self.pipe.transformer.__class__)
        
        # Use MPS for Mac M-series chips
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.pipe = self.pipe.to(self.device)
        
        # Enable attention slicing for memory efficiency
        self.pipe.enable_attention_slicing()
        print("Model pipeline loaded successfully.")

    def init_upscaler(self):
        """Initialize the upscaler during model setup"""
        print("Initializing upscaler...")
        try:
            # Try to import and initialize RealESRGAN
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # Initialize RealESRGAN
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
            
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            print(f"Initializing upscaler on device: {device}")
            
            self.upsampler = RealESRGANer(
                device,
                scale=2,
                model_path=None,  # Will download automatically first time
                model=model,
                half=False if device == 'cpu' else True  # Half precision for GPU, full for CPU
            )
            
            print("RealESRGAN upscaler initialized successfully")
            self.has_upscaler = True
            
        except ImportError as e:
            print(f"Warning: RealESRGAN not available - {e}")
            print("Using standard PIL for upscaling. For better quality, install realesrgan and basicsr.")
            self.has_upscaler = False
            self.upsampler = None
        except Exception as e:
            print(f"Error initializing upscaler: {e}")
            self.has_upscaler = False
            self.upsampler = None

    def get_character_descriptions(self, character_name, max_desc=3):
        """Get curated descriptions for a specific character"""
        for character in self.character_data:
            if character["name"] == character_name:
                descriptions = character.get("descriptions", [])
                # Return a limited number of the most important descriptions
                return descriptions[:max_desc]
        return []
    
    def load_character_from_keepers(self, character_name):
        """Check if a character has an image in the keepers folder and load it"""
        # Path to character_output/character_images/keepers directory
        keepers_dir = Path(self.output_dir).parent / "character_output" / "character_images" / "keepers"
        
        # Check for character image in keepers folder
        character_image_path = keepers_dir / f"{character_name}.png"
        
        if character_image_path.exists():
            return str(character_image_path)
        return None
    
    def load_character_embeddings(self):
        """Load the character embeddings map"""
        embeddings_map_path = Path("character_output") / "character_embeddings" / "character_embeddings_map.json"
        
        if embeddings_map_path.exists():
            with open(embeddings_map_path, 'r') as f:
                self.character_embedding_map = json.load(f)
            print(f"Loaded {len(self.character_embedding_map)} character embeddings")
        else:
            print(f"Warning: Character embeddings map not found at {embeddings_map_path}")
            self.character_embedding_map = {}

    def get_character_embedding(self, character_name):
        """Get the embedding for a character"""
        if character_name in self.character_embedding_map:
            embedding_path = self.character_embedding_map[character_name]["embedding_path"]
            
            try:
                # Load the embedding tensor with the safe_globals context manager
                with torch.serialization.safe_globals([_reconstruct]):
                    embedding = torch.load(embedding_path, weights_only=False)
                    return embedding
            except Exception as e:
                print(f"Error loading embedding for {character_name}: {e}")
        
        return None

    def prepare_panel_references(self, panel_data):
        """Prepare character reference embeddings for a panel"""
        characters = list(panel_data['characters'])
        reference_embeddings = {}
        
        for character_name in characters:
            embedding = self.get_character_embedding(character_name)
            if embedding is not None:
                reference_embeddings[character_name] = embedding
                print(f"Using stored embedding for {character_name}")
        
        return reference_embeddings

    def get_character_reference_embedding(self, character_name):
        """Get embedding for a character, prioritizing images in the keepers folder"""
        # Check if character exists in keepers folder
        keeper_image_path = self.load_character_from_keepers(character_name)
        
        if keeper_image_path:
            # Here you would implement code to create an embedding from the image
            # This depends on your embedding method (CLIP, etc.)
            # Example pseudocode:
            # embedding = create_embedding_from_image(keeper_image_path)
            # return embedding
            print(f"Found keeper image for {character_name}: {keeper_image_path}")
            return keeper_image_path
        
        # If not in keepers, check if it exists in the character_embeddings
        elif character_name in self.character_embeddings:
            return self.character_embeddings[character_name]["image_path"]
        
        return None
    
    def create_panel_prompt(self, panel_data, use_character_descriptions=None):
        """Create a comprehensive prompt for a manga panel with enhanced character descriptions
        
        Args:
            panel_data: Panel data including characters, setting, etc.
            use_character_descriptions: If None, use hybrid approach (descriptions + embeddings).
                                      If True/False, force include/exclude descriptions.
        """
        setting = panel_data['setting']
        characters = list(panel_data['characters'])
        elements = panel_data['elements']
        
        # Extract action and dialogue text
        action_texts = []
        dialogue_texts = []
        
        for element in elements:
            if element['type'] == 'action':
                action_texts.append(element['text'])
            elif element['type'] == 'dialogue':
                dialogue_texts.append(f"{element['character']} says: {element['dialogue']}")
        
        # Base prompt elements - using narrative style for PixArt
        prompt_parts = [
            "A black and white manga panel with screen tone showing"
        ]
        
        # Add setting
        prompt_parts.append(setting)
        
        # Determine whether to include character descriptions
        if use_character_descriptions is None:
            # REVERT TO DESCRIPTIONS-FIRST APPROACH: Use descriptions when available, skip embeddings
            # This avoids interference between text and visual channels
            use_character_descriptions = True
        
        # Add characters with enhanced descriptions
        if use_character_descriptions:
            # Enhanced approach: include detailed character descriptions alongside embeddings
            for character_name in characters:
                char_descriptions = self.get_character_descriptions(character_name)
                has_embedding = self.get_character_embedding(character_name) is not None
                
                if char_descriptions:
                    # Create rich character description - this is the primary method for character consistency
                    char_prompt = f"{character_name} ({', '.join(char_descriptions)})"
                    prompt_parts.append(char_prompt)
                    print(f"Using enhanced description for {character_name}: {', '.join(char_descriptions)}")
                else:
                    # Fallback to just name if no descriptions available
                    prompt_parts.append(character_name)
                    print(f"Using name only for {character_name} (no description available)")
        else:
            # Fallback: just mention character names, rely on embeddings
            if characters:
                char_names = ", ".join(characters)
                prompt_parts.append(f"featuring {char_names}")
        
        # Add actions if available
        if action_texts:
            action_prompt = " ".join(action_texts)
            prompt_parts.append(action_prompt)
        
        # Create the final prompt in a more natural language style
        prompt = " ".join(prompt_parts)
        
        # Add manga-specific style enhancements
        style_enhancements = [
            "detailed lineart",
            "high contrast black and white",
            "professional manga illustration style",
            "sharp clean lines"
        ]
        prompt += f", {', '.join(style_enhancements)}"
        
        # For dialogue, add speech bubble instructions
        if dialogue_texts:
            prompt += f". Speech bubbles contain: {' '.join(dialogue_texts)}"
            
        return prompt
    
    def optimize_panel_dimensions(self, width, height):
        """
        Optimizes panel dimensions to work well with the image generation model.
        
        Args:
            width (int): Original panel width
            height (int): Original panel height
            
        Returns:
            tuple: (generation_width, generation_height, scale_factor)
        """
        # Maximum area the model can handle efficiently (512×512)
        max_area = 512 * 512
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Calculate dimensions that fit within model constraints
        if width * height > max_area:
            # Scale down proportionally to fit max area
            scale = (max_area / (width * height)) ** 0.5
            gen_width = int(width * scale)
            gen_height = int(height * scale)
        else:
            gen_width = width
            gen_height = height
        
        # Ensure dimensions are multiples of 8 (for stable diffusion models)
        gen_width = (gen_width // 8) * 8
        gen_height = (gen_height // 8) * 8
        
        # Ensure minimum dimensions
        gen_width = max(64, gen_width)
        gen_height = max(64, gen_height)
        
        # Calculate how much we're scaling for later upscaling
        scale_factor = (width / gen_width, height / gen_height)
        
        # Log the optimization for debugging
        print(f"Panel optimization: {width}×{height} → {gen_width}×{gen_height}")
        print(f"Aspect ratio maintained: {aspect_ratio:.3f} ≈ {gen_width/gen_height:.3f}")
        
        return gen_width, gen_height, scale_factor

    def upscale_panel_image(self, image, original_width, original_height):
        """
        Upscales the generated panel image back to the original dimensions.
        Uses pre-initialized upscaler if available.
        
        Args:
            image (PIL.Image): Generated image at model dimensions
            original_width (int): Original panel width
            original_height (int): Original panel height
            
        Returns:
            PIL.Image: Upscaled image at original dimensions
        """
        # If dimensions are already correct, no need to upscale
        if image.width == original_width and image.height == original_height:
            return image
            
        try:
            # Use the pre-initialized upscaler if available
            if self.has_upscaler and self.upsampler is not None:
                # Convert PIL image to numpy array
                img_array = np.array(image)
                
                # Process the image with RealESRGAN
                upscaled, _ = self.upsampler.enhance(img_array)
                upscaled_image = Image.fromarray(upscaled)
                
                # Resize to exact target dimensions
                final_image = upscaled_image.resize((original_width, original_height), Image.LANCZOS)
                print(f"Upscaled image from {image.width}x{image.height} to {final_image.width}x{final_image.height} using RealESRGAN")
                return final_image
                
        except Exception as e:
            print(f"Advanced upscaling failed: {e}. Falling back to standard resize.")
        
        # Fallback to standard PIL upscaling if RealESRGAN failed or isn't available
        print(f"Using standard PIL resize from {image.width}x{image.height} to {original_width}x{original_height}")
        return image.resize((original_width, original_height), Image.LANCZOS)
    
    def add_text_overlays(self, image, ip_params, panel_data, width, height):
        """
        Add white boxes with black text over the generated image to replace AI garbage text.
        """
        # Create a copy of the image to modify and ensure RGB mode
        overlaid_image = image.copy()
        if overlaid_image.mode != 'RGB':
            overlaid_image = overlaid_image.convert('RGB')
        draw = ImageDraw.Draw(overlaid_image)
        
        # Get text boxes from panel data
        text_boxes = panel_data.get('textBoxes', [])
        
        # Load font
        try:
            font_paths = [
                "/Users/iansears/mangaui/fonts/mangat.ttf",
                "/System/Library/Fonts/Arial.ttf",
                "arial.ttf"
            ]
            
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, 16)
                    break
                except:
                    continue
                    
            if font is None:
                font = ImageFont.load_default()
                
        except:
            font = ImageFont.load_default()
        
        # Add each text overlay
        for i, text_box in enumerate(text_boxes):
            # Get text content
            text = text_box.get('text', '').strip()
            
            # Get position and size (convert normalized coordinates to pixels)
            x_norm = text_box.get('x', 0)
            y_norm = text_box.get('y', 0)
            width_norm = text_box.get('width', 0.1)
            height_norm = text_box.get('height', 0.1)
            
            x = int(x_norm * width)
            y = int(y_norm * height)
            box_width = int(width_norm * width)
            box_height = int(height_norm * height)
            
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, width - box_width))
            y = max(0, min(y, height - box_height))
            box_width = min(box_width, width - x)
            box_height = min(box_height, height - y)
            
            # Draw white rectangle background (no border to blend with speech bubble)
            draw.rectangle(
                [x, y, x + box_width, y + box_height],
                fill='white'
            )
            
            # If there's text, add it
            if text:
                # Find the optimal font size that uses the full text box
                padding = 8
                available_width = box_width - padding
                available_height = box_height - padding
                
                optimal_size = self.find_optimal_font_size_only(text, available_width, available_height)
                
                # Load font
                try:
                    optimal_font = ImageFont.truetype("/Users/iansears/mangaui/fonts/mangat.ttf", optimal_size)
                except:
                    try:
                        optimal_font = ImageFont.truetype("arial.ttf", optimal_size)
                    except:
                        optimal_font = font
                
                # Word wrap text to fit in the box
                wrapped_text = self.wrap_text(text, optimal_font, available_width)
                
                # Calculate text position (centered in the box)
                try:
                    text_bbox = draw.textbbox((0, 0), wrapped_text, font=optimal_font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    text_x = x + (box_width - text_width) // 2
                    text_y = y + (box_height - text_height) // 2
                    
                    # Draw the text in black
                    draw.text((text_x, text_y), wrapped_text, fill=(0, 0, 0), font=optimal_font)
                except:
                    # Simple fallback
                    draw.text((x + 5, y + 5), text, fill=(0, 0, 0), font=font)
            
            print(f"Added text overlay {i+1}: '{text}' at ({x}, {y}) size ({box_width}x{box_height})")
        
        return overlaid_image
    
    def wrap_text(self, text, font, max_width):
        """
        Wrap text to fit within the specified width.
        
        Args:
            text (str): Text to wrap
            font: PIL font object
            max_width (int): Maximum width in pixels
            
        Returns:
            str: Wrapped text with newlines
        """
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Test if adding this word would exceed the width
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line)
            test_width = bbox[2] - bbox[0]
            
            if test_width <= max_width:
                current_line.append(word)
            else:
                # Start a new line
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word is too long, break it
                    lines.append(word)
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def find_optimal_font_size_only(self, text, max_width, max_height):
        """
        Find the largest font size (number) that fits the text within the given dimensions.
        Returns just the size, not the font object.
        """
        # Start with a reasonable range
        min_size = 8
        max_size = min(max_height, 72)  # Cap at 72pt or box height
        optimal_size = min_size
        
        # Binary search for the optimal font size
        while min_size <= max_size:
            test_size = (min_size + max_size) // 2
            
            # Try to load font at test size using the same method that worked
            try:
                test_font = ImageFont.truetype("/Users/iansears/mangaui/fonts/mangat.ttf", test_size)
            except:
                try:
                    test_font = ImageFont.truetype("arial.ttf", test_size)
                except:
                    test_font = ImageFont.load_default()
            
            # Wrap text and measure dimensions
            wrapped_text = self.wrap_text(text, test_font, max_width)
            
            # Get text dimensions
            try:
                # Create a temporary draw object to measure text
                temp_image = Image.new('RGB', (1, 1))
                temp_draw = ImageDraw.Draw(temp_image)
                text_bbox = temp_draw.textbbox((0, 0), wrapped_text, font=test_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                # Fallback measurement
                text_width = test_size * len(text) * 0.6  # Rough estimate
                text_height = test_size * wrapped_text.count('\n') + test_size
            
            # Check if text fits
            if text_width <= max_width and text_height <= max_height:
                optimal_size = test_size
                min_size = test_size + 1  # Try larger
            else:
                max_size = test_size - 1  # Try smaller
        
        return optimal_size
    
    def find_optimal_font_size(self, text, max_width, max_height):
        """
        Find the largest font size that fits the text within the given dimensions.
        
        Args:
            text (str): Text to fit
            max_width (int): Maximum width in pixels
            max_height (int): Maximum height in pixels
            
        Returns:
            PIL.ImageFont: Font object with optimal size
        """
        # Start with a reasonable range
        min_size = 8
        max_size = min(max_height, 72)  # Cap at 72pt or box height
        optimal_size = min_size
        
        # Binary search for the optimal font size
        while min_size <= max_size:
            test_size = (min_size + max_size) // 2
            
            # Try to load font at test size
            try:
                test_font = ImageFont.truetype("/Users/iansears/mangaui/fonts/mangat.ttf", test_size)
            except:
                try:
                    test_font = ImageFont.truetype("arial.ttf", test_size)
                except:
                    test_font = ImageFont.load_default()
            
            # Wrap text and measure dimensions
            wrapped_text = self.wrap_text(text, test_font, max_width)
            
            # Get text dimensions
            try:
                # Create a temporary draw object to measure text
                temp_image = Image.new('RGB', (1, 1))
                temp_draw = ImageDraw.Draw(temp_image)
                text_bbox = temp_draw.textbbox((0, 0), wrapped_text, font=test_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                # Fallback measurement
                text_width = test_size * len(text) * 0.6  # Rough estimate
                text_height = test_size * wrapped_text.count('\n') + test_size
            
            # Check if text fits
            if text_width <= max_width and text_height <= max_height:
                optimal_size = test_size
                min_size = test_size + 1  # Try larger
            else:
                max_size = test_size - 1  # Try smaller
        
        # Return font with optimal size
        try:
            return ImageFont.truetype("/Users/iansears/mangaui/fonts/mangat.ttf", optimal_size)
        except:
            try:
                return ImageFont.truetype("arial.ttf", optimal_size)
            except:
                return ImageFont.load_default()
    
    def prepare_ip_adapter_inputs(self, panel_data):
        """Prepare inputs for IPAdapter in the format expected by the model"""
        # Get characters in the panel
        characters = list(panel_data['characters'])
        elements = panel_data['elements']
        
        # Initialize lists for both text boxes and character boxes
        text_bboxes = []
        character_bboxes = []
        reference_embeddings = []
        
        # Check if there are explicit text boxes defined
        if 'textBoxes' in panel_data and panel_data['textBoxes'] and len(panel_data['textBoxes']) > 0:
            # Use the explicit text boxes from the panel data
            for idx, text_box in enumerate(panel_data['textBoxes']):
                # Convert relative coordinates from text_box to the format expected by the model
                text_bboxes.append([
                    text_box['x'], 
                    text_box['y'], 
                    text_box['x'] + text_box['width'], 
                    text_box['y'] + text_box['height']
                ])
        
        # Add character embeddings from keepers
        for character_name in characters:
            # Get embedding for this character
            embedding = self.get_character_embedding(character_name)
            
            if embedding is not None:
                # If character boxes are defined, use those positions
                if 'characterBoxes' in panel_data and panel_data['characterBoxes']:
                    for char_box in panel_data['characterBoxes']:
                        if char_box['character'] == character_name:
                            character_bboxes.append([
                                char_box['x'], 
                                char_box['y'], 
                                char_box['x'] + char_box['width'], 
                                char_box['y'] + char_box['height']
                            ])
                            reference_embeddings.append(embedding)
                            break
                    else:
                        # If no matching box, place character in center of panel
                        character_bboxes.append([0.2, 0.2, 0.8, 0.8])
                        reference_embeddings.append(embedding)
                else:
                    # Default placement if no character boxes defined
                    character_bboxes.append([0.2, 0.2, 0.8, 0.8])
                    reference_embeddings.append(embedding)
        
        # Always wrap in batch format for consistent API
        return [text_bboxes], [character_bboxes], [reference_embeddings]

    def generate_panel(self, panel_data, panel_index, seed=None, width=None, height=None, project_id='default', ip_params=None, negative_prompt=None):
        """
        Generate a manga panel based on panel data with proper dimension handling.
        
        Args:
            panel_data (dict): Panel data including characters, setting, etc.
            panel_index (int): Index of the panel
            seed (int, optional): Random seed for reproducibility
            width (int, optional): Requested panel width from UI
            height (int, optional): Requested panel height from UI
            project_id (str, optional): Project ID for organization
            ip_params (dict, optional): Parameters for the IPAdapter including text_bboxes, character_bboxes, and reference_embeddings
            negative_prompt (str, optional): Negative prompt for generation
        
        Returns:
            tuple: (output_path, panel_data) - Path to saved image and panel data
        """
        # Determine output path based on project_id
        if project_id != 'default':
            # UI-based generation - save to manga_projects
            output_base = Path("manga_projects") / project_id / "panels"
            output_base.mkdir(parents=True, exist_ok=True)
            output_path = output_base / f"panel_{panel_index:04d}.png"
        else:
            # Command-line/batch generation - save to self.panels_dir (manga_output/panels)
            self.panels_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.panels_dir / f"panel_{panel_index:04d}.png"

        # Create a specific prompt for this panel
        # When using IP adapter with character embeddings, exclude character descriptions from prompt
        prompt = self.create_panel_prompt(panel_data)
        
        # Standard negative prompt for manga
        if negative_prompt is None:
            negative_prompt = "deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands"
        
        # Set up generator for seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Use provided dimensions or default to 512x512 if none specified
        if width is None or height is None:
            width = 512
            height = 512
        
        # Optimize dimensions for the model
        gen_width, gen_height, scale_factor = self.optimize_panel_dimensions(width, height)
        
        print(f"Generating panel {panel_index}: {prompt[:100]}...")
        print(f"Panel dimensions: {width}x{height} (requested), {gen_width}x{gen_height} (generation)")
        print(f"Output path: {output_path}")
        
        # MINIMAL CHANGE: Store IP parameters on transformer before pipeline call
        transformer = self.pipe.transformer
        
        if ip_params is not None:
            text_bboxes = ip_params.get('text_bboxes', [])
            character_bboxes = ip_params.get('character_bboxes', [])
            reference_embeddings = ip_params.get('reference_embeddings', [])
            
            # Convert numpy arrays to PyTorch tensors for reference embeddings
            if reference_embeddings:
                converted_embeddings = []
                for emb in reference_embeddings:
                    if emb is not None:
                        if isinstance(emb, np.ndarray):
                            # Convert numpy array to PyTorch tensor
                            emb_tensor = torch.from_numpy(emb).to(
                                device=self.device,
                                dtype=torch.float16  # Match model dtype
                            )
                            # Flatten to remove extra dimensions - should be [768] not [1, 768]
                            emb_tensor = emb_tensor.flatten()
                            converted_embeddings.append(emb_tensor)
                        else:
                            # Ensure existing tensors have correct device and dtype
                            emb_tensor = emb.to(device=self.device, dtype=torch.float16)
                            # Flatten to remove extra dimensions - should be [768] not [1, 768]
                            emb_tensor = emb_tensor.flatten()
                            converted_embeddings.append(emb_tensor)
                    else:
                        converted_embeddings.append(None)
                reference_embeddings = converted_embeddings
            
            # Convert to batch format expected by IP adapter
            # Your IP adapter expects: [[batch1_items], [batch2_items], ...]
            # So wrap each list in another list for batch_size=1
            transformer._temp_text_bboxes = [text_bboxes]  # [[...]] format
            transformer._temp_character_bboxes = [character_bboxes]  # [[...]] format  
            transformer._temp_reference_embeddings = [reference_embeddings]  # [[...]] format
            
            # Report what we're using
            char_count = len(character_bboxes) if character_bboxes else 0
            text_count = len(text_bboxes) if text_bboxes else 0
            print(f"Using {char_count} character references and {text_count} text boxes")
            print(f"Stored parameters on transformer for IP adapter")
            print(f"  text_bboxes format: {len(transformer._temp_text_bboxes)} batches, {len(transformer._temp_text_bboxes[0]) if transformer._temp_text_bboxes else 0} items")
            print(f"  character_bboxes format: {len(transformer._temp_character_bboxes)} batches, {len(transformer._temp_character_bboxes[0]) if transformer._temp_character_bboxes else 0} items")
            print(f"  reference_embeddings format: {len(transformer._temp_reference_embeddings)} batches, {len(transformer._temp_reference_embeddings[0]) if transformer._temp_reference_embeddings else 0} items")
            if reference_embeddings:
                print(f"  reference_embeddings types: {[type(emb).__name__ if emb is not None else 'None' for emb in reference_embeddings]}")
                print(f"  reference_embeddings devices: {[emb.device if hasattr(emb, 'device') else 'N/A' for emb in reference_embeddings]}")
                print(f"  reference_embeddings shapes: {[emb.shape if hasattr(emb, 'shape') else 'N/A' for emb in reference_embeddings]}")
                print(f"  reference_embeddings dtypes: {[emb.dtype if hasattr(emb, 'dtype') else 'N/A' for emb in reference_embeddings]}")
                
                # PROOF: Show exactly which character boxes are linked to which embeddings
                print(f"🔗 CHARACTER EMBEDDING VERIFICATION:")
                for i, (bbox, emb) in enumerate(zip(character_bboxes, reference_embeddings)):
                    if emb is not None:
                        # Calculate a simple hash of the embedding for identification
                        emb_hash = hash(emb.flatten().sum().item()) % 10000
                        print(f"    Character box {i}: bbox={bbox} -> embedding_hash={emb_hash:04d}, shape={emb.shape}")
                    else:
                        print(f"    Character box {i}: bbox={bbox} -> NO EMBEDDING")
                print(f"✅ CONFIRMED: {char_count} character boxes are spatially linked to {len([e for e in reference_embeddings if e is not None])} embeddings")
        else:
            # Clear any existing stored parameters
            transformer._temp_text_bboxes = None
            transformer._temp_character_bboxes = None
            transformer._temp_reference_embeddings = None
        
        try:
            # Use the standard pipeline (which was working!)
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=gen_width,
                height=gen_height,
                generator=generator
            ).images[0]
            
            # Upscale to requested dimensions if necessary
            if gen_width != width or gen_height != height:
                image = self.upscale_panel_image(image, width, height)
                
        except Exception as e:
            print(f"Error during generation: {e}")
            print("Trying with smaller resolution...")
            # Fallback to lower resolution
            try:
                # Use a more conservative resolution
                gen_width = min(384, gen_width)
                gen_height = min(384, gen_height)
                
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    width=gen_width,
                    height=gen_height,
                    generator=generator
                ).images[0]
                
                # Upscale to requested dimensions
                image = self.upscale_panel_image(image, width, height)
                
            except Exception as e:
                print(f"Error during fallback generation: {e}")
                # Create a blank panel with error message as last resort
                image = Image.new('RGB', (width, height), color='white')
                draw = ImageDraw.Draw(image)
                draw.text((width//2-100, height//2), "Generation error", fill='black')
        
        finally:
            # ALWAYS clean up stored parameters
            if hasattr(transformer, '_temp_text_bboxes'):
                delattr(transformer, '_temp_text_bboxes')
            if hasattr(transformer, '_temp_character_bboxes'):
                delattr(transformer, '_temp_character_bboxes')
            if hasattr(transformer, '_temp_reference_embeddings'):
                delattr(transformer, '_temp_reference_embeddings')
        
        # Add text overlays if any text boxes are defined
        if panel_data.get('textBoxes'):
            print(f"Applying text overlays for {len(panel_data['textBoxes'])} text boxes")
            image = self.add_text_overlays(image, ip_params, panel_data, width, height)
        
        # Save the image
        image.save(output_path)
        
        # Save panel metadata for both command-line and UI generation
        if project_id == 'default':
            # Command-line generation - save to self.panels_dir
            panel_info_path = self.panels_dir / f"panel_{panel_index:04d}.json"
        else:
            # UI generation - save to project panels directory
            panel_info_path = output_path.parent / f"panel_{panel_index:04d}.json"
            
        with open(panel_info_path, 'w') as f:
            # Convert set to list for JSON serialization
            panel_data_json = panel_data.copy()
            panel_data_json['characters'] = list(panel_data_json['characters'])
            panel_data_json['prompt'] = prompt
            panel_data_json['seed'] = seed
            panel_data_json['width'] = width
            panel_data_json['height'] = height
            panel_data_json['gen_width'] = gen_width
            panel_data_json['gen_height'] = gen_height
            panel_data_json['project_id'] = project_id
            characters = list(panel_data['characters'])
            panel_data_json['characters_with_embeddings'] = [
                character for character in characters if self.get_character_embedding(character) is not None
            ]
            json.dump(panel_data_json, f, indent=2)
        
        return output_path, panel_data
    
    def generate_manga_from_panels(self, panel_prompts, start_panel=0, end_panel=None, use_fixed_seed=False):
        """Generate manga panels based on panel prompts"""
        panel_paths = []
        
        # Handle panel range
        if end_panel is None:
            end_panel = len(panel_prompts)
        else:
            end_panel = min(end_panel, len(panel_prompts))
        
        # Subset of panels to generate
        panels_to_generate = panel_prompts[start_panel:end_panel]
        
        for idx, panel_data in enumerate(tqdm(panels_to_generate, desc="Generating panels")):
            # Use a fixed seed for reproducibility if requested
            seed = 42 * (idx + 1) if use_fixed_seed else torch.randint(0, 2**32, (1,)).item()
            
            panel_path, data = self.generate_panel(panel_data, start_panel + idx, seed)
            panel_paths.append({
                'path': panel_path,
                'data': data
            })
        
        return panel_paths
    
    def create_manga_pages(self, panel_paths, panels_per_page=6):
        """Arrange panels into manga pages"""
        total_panels = len(panel_paths)
        total_pages = (total_panels + panels_per_page - 1) // panels_per_page
        
        pages = []
        
        for page_idx in range(total_pages):
            start_idx = page_idx * panels_per_page
            end_idx = min(start_idx + panels_per_page, total_panels)
            
            page_panels = panel_paths[start_idx:end_idx]
            page_path = self._create_page_layout(page_panels, page_idx)
            pages.append(page_path)
        
        return pages
    
    def _create_page_layout(self, page_panels, page_idx):
        """Create a page layout from panels"""
        # For simplicity, we'll use a basic grid layout
        # In a real implementation, you'd want a more sophisticated layout engine
        
        # Base page size (A4 proportions)
        page_width = 1654
        page_height = 2339
        
        page_image = Image.new('RGB', (page_width, page_height), (255, 255, 255))
        draw = ImageDraw.Draw(page_image)
        
        num_panels = len(page_panels)
        
        if num_panels <= 2:
            # Simple vertical stack
            panel_height = page_height // num_panels
            for i, panel_data in enumerate(page_panels):
                panel_img = Image.open(panel_data['path'])
                panel_img = panel_img.resize((page_width, panel_height), Image.LANCZOS)
                page_image.paste(panel_img, (0, i * panel_height))
        
        elif num_panels <= 4:
            # 2x2 grid
            panel_width = page_width // 2
            panel_height = page_height // 2
            for i, panel_data in enumerate(page_panels):
                row = i // 2
                col = i % 2
                panel_img = Image.open(panel_data['path'])
                panel_img = panel_img.resize((panel_width, panel_height), Image.LANCZOS)
                page_image.paste(panel_img, (col * panel_width, row * panel_height))
        
        else:
            # 3x2 grid
            panel_width = page_width // 2
            panel_height = page_height // 3
            for i, panel_data in enumerate(page_panels):
                if i >= 6:  # Maximum 6 panels per page
                    break
                row = i // 2
                col = i % 2
                panel_img = Image.open(panel_data['path'])
                panel_img = panel_img.resize((panel_width, panel_height), Image.LANCZOS)
                page_image.paste(panel_img, (col * panel_width, row * panel_height))
        
        # Add page number
        font_size = 36
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        draw.text((page_width - 100, page_height - 50), f"Page {page_idx + 1}", fill=(0, 0, 0), font=font)
        
        # Save the page
        output_path = self.pages_dir / f"page_{page_idx:03d}.png"
        page_image.save(output_path)
        
        print(f"Created page {page_idx + 1} with {num_panels} panels")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate manga from screenplay using Drawatoon")
    parser.add_argument("--model_path", type=str, default="./drawatoon-v1",
                        help="Path to the Drawatoon model")
    parser.add_argument("--screenplay", type=str, default="the-rat.txt",
                        help="Path to screenplay text file")
    parser.add_argument("--character_data", type=str, default="characters.json",
                        help="Path to character data JSON file")
    parser.add_argument("--character_embeddings", type=str, 
                        default="character_output/character_embeddings.json",
                        help="Path to character embeddings JSON file")
    parser.add_argument("--output_dir", type=str, default="manga_output",
                        help="Directory to save manga output")
    parser.add_argument("--start_scene", type=int, default=0,
                        help="Scene index to start generation from")
    parser.add_argument("--end_scene", type=int, default=None,
                        help="Scene index to end generation at (exclusive)")
    parser.add_argument("--start_panel", type=int, default=0,
                        help="Panel index to start generation from")
    parser.add_argument("--end_panel", type=int, default=None,
                        help="Panel index to end generation at (exclusive)")
    parser.add_argument("--panels_per_page", type=int, default=6,
                        help="Number of panels per manga page")
    parser.add_argument("--fixed_seed", action="store_true",
                        help="Use fixed seeds for reproducibility")
    parser.add_argument("--parse_only", action="store_true",
                        help="Only parse the screenplay without generating images")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse the screenplay
    print(f"Parsing screenplay: {args.screenplay}")
    parser = ScreenplayParser(args.screenplay, args.character_data)
    scenes = parser.parse()
    print(f"Found {len(scenes)} scenes in the screenplay")
    
    # Save the parsed scenes for reference
    scenes_path = Path(args.output_dir) / "parsed_scenes.json"
    with open(scenes_path, 'w') as f:
        json.dump(scenes, f, indent=2)
    print(f"Saved parsed scenes to {scenes_path}")
    
    # Generate panel prompts with dynamic panel estimation
    print("Generating panel prompts with dynamic panel allocation...")
    panel_prompts = parser.generate_panel_prompts(
        scenes, 
        start_scene=args.start_scene,
        end_scene=args.end_scene
    )
    print(f"Generated {len(panel_prompts)} panel prompts")
    
    # Save panel prompts for reference
    prompts_path = Path(args.output_dir) / "panel_prompts.json"
    
    # Convert sets to lists for JSON serialization
    serializable_prompts = []
    for prompt in panel_prompts:
        prompt_copy = prompt.copy()
        prompt_copy['characters'] = list(prompt_copy['characters'])
        serializable_prompts.append(prompt_copy)
    
    with open(prompts_path, 'w') as f:
        json.dump(serializable_prompts, f, indent=2)
    print(f"Saved panel prompts to {prompts_path}")
    
    if args.parse_only:
        print("Parsed screenplay and generated prompts. Exiting as requested.")
        return
    
    # Initialize manga generator and generate panels
    print("Initializing manga generator...")
    manga_generator = MangaGenerator(
        model_path=args.model_path,
        character_data_path=args.character_data,
        character_embedding_path=args.character_embeddings,
        output_dir=args.output_dir
    )
    
    # Generate panels
    print("Generating manga panels...")
    panel_paths = manga_generator.generate_manga_from_panels(
        panel_prompts,
        start_panel=args.start_panel,
        end_panel=args.end_panel,
        use_fixed_seed=args.fixed_seed
    )
    
    # Create manga pages
    print("Creating manga pages...")
    pages = manga_generator.create_manga_pages(panel_paths, args.panels_per_page)
    
    print(f"Generated {len(pages)} manga pages from screenplay.")
    print(f"Output saved to {args.output_dir}")
    
    # Create a summary file
    summary_path = Path(args.output_dir) / "manga_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Manga Generation Summary\n")
        f.write(f"======================\n\n")
        f.write(f"Screenplay: {args.screenplay}\n")
        f.write(f"Total scenes in screenplay: {len(scenes)}\n")
        f.write(f"Scenes processed: {args.start_scene} to {args.end_scene if args.end_scene else len(scenes)}\n")
        f.write(f"Total panels generated: {len(panel_paths)}\n")
        f.write(f"Total pages created: {len(pages)}\n")
        f.write(f"Panels per page: {args.panels_per_page}\n\n")
        f.write(f"Page listing:\n")
        for i, page in enumerate(pages):
            f.write(f"  Page {i+1}: {page}\n")
    
    print(f"Summary written to {summary_path}")

if __name__ == "__main__":
    main()