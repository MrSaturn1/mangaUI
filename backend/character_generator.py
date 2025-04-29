import os
import json
import torch
from pathlib import Path
from diffusers import PixArtSigmaPipeline
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import datetime
import shutil

# Character Generator class for creating and managing character designs
class CharacterGenerator:
    def __init__(self, model_path, character_data_path, output_dir="character_output"):
        # Load character data
        with open(character_data_path, 'r', encoding='utf-8') as f:
            self.character_data = json.load(f)
        
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.character_images_dir = self.output_dir / "character_images"
        self.character_images_dir.mkdir(parents=True, exist_ok=True)

        # Create keepers directory inside character_images_dir (not output_dir)
        self.keepers_dir = self.character_images_dir / "keepers"
        self.keepers_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the model pipeline
        self.init_pipeline()
        
        # Dictionary to track which characters have been generated
        self.generated_characters = {}
        
        # Load embedding map if it exists
        self.embedding_map_path = self.output_dir / "character_embeddings.json"
        if self.embedding_map_path.exists():
            with open(self.embedding_map_path, 'r') as f:
                self.generated_characters = json.load(f)
    
    def init_pipeline(self):
        """Initialize the Drawatoon pipeline"""
        print("Loading model pipeline...")
        
        # Use standard pipeline
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16
        )
        
        # We need a more fundamental patch that overrides the actual forward method
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
                ip_hidden_states, ip_attention_mask = self.ip_adapter(text_bboxes, character_bboxes, reference_embeddings, cfg_on_10_percent)
            
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
            for block in self.transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
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
                else:
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

            from diffusers.models.modeling_outputs import Transformer2DModelOutput
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
    
    def get_important_characters(self, min_importance=4):
        """Get a list of important characters based on their description count"""
        important_characters = []
        
        for character in self.character_data:
            if character.get("character_type") == "NAMED":
                descriptions = character.get("descriptions", [])
                character_name = character.get("name")
                
                # Consider only characters with enough descriptions
                if len(descriptions) >= min_importance:
                    important_characters.append({
                        "name": character_name,
                        "descriptions": descriptions,
                        "importance": len(descriptions)
                    })
        
        # Sort by importance (description count)
        important_characters.sort(key=lambda x: x["importance"], reverse=True)
        return important_characters
    
    def get_all_characters(self):
        """Get a list of all named characters regardless of description count"""
        all_characters = []
        
        for character in self.character_data:
            descriptions = character.get("descriptions", [])
            character_name = character.get("name")
            
            # Add all named characters
            all_characters.append({
                "name": character_name,
                "descriptions": descriptions,
                "importance": len(descriptions)
            })

        return all_characters
    
    def prepare_inputs_for_character_generation(self):
        """Prepare inputs for the IPAdapter that will use the void embedding"""
        # Create one empty list per batch - this triggers void embedding path
        text_bboxes = [[]]  # One empty list
        character_bboxes = [[]]  # One empty list 
        reference_embeddings = [[]]  # One empty list
        return text_bboxes, character_bboxes, reference_embeddings
    
    def prepare_ip_adapter_inputs(self):
        """Create IP hidden states and attention mask using void embedding for character generation"""
        # Access the IPAdapter to get dimensions right
        ip_adapter = self.pipe.transformer.ip_adapter
        
        # Get the void embedding
        void_embed = ip_adapter.void_ip_embed.weight
        
        # Create IP hidden states using the void embedding (batch size of 1)
        batch_size = 1
        ip_hidden_states = void_embed.unsqueeze(0).expand(batch_size, 1, -1).to(self.device)
        
        # Create attention mask (all ones)
        ip_attention_mask = torch.ones(
            (batch_size, 1), 
            device=self.device, 
            dtype=torch.bool
        )
        
        print(f"Created IP hidden states with shape: {ip_hidden_states.shape}")
        print(f"Created IP attention mask with shape: {ip_attention_mask.shape}")
        
        return ip_hidden_states, ip_attention_mask
    
    def generate_character(self, character_name, seed=None, regenerate=False):
        """Generate a character image based on character descriptions"""
        # Get character descriptions
        character_info = None
        for character in self.character_data:
            if character["name"] == character_name:
                descriptions = character.get("descriptions", [])
                character_info = {
                    "name": character_name,
                    "descriptions": descriptions,
                    "importance_context": character.get("importance_context", "")
                }
                break
        
        if not character_info:
            print(f"No character data found for: {character_name}")
            return None
        
        descriptions = character_info["descriptions"]
        
        if not descriptions:
            print(f"No descriptions found for character: {character_name}")
            return None
        
        # Select the most relevant descriptions (up to 5)
        selected_descriptions = descriptions[:5]
        
        # Create a prompt for the character
        prompt = f"manga style character portrait, black and white panel with screen tone, clear detailed face of {character_name}, " + ", ".join(selected_descriptions)
        negative_prompt = "deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands"
        
        # Use the provided seed if available
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Using seed: {seed}")
        
        # Create IP hidden states directly from the void embedding
        ip_adapter = self.pipe.transformer.ip_adapter
        void_embed = ip_adapter.void_ip_embed.weight
        batch_size = 1
        ip_hidden_states = void_embed.unsqueeze(0).expand(batch_size, 1, -1).to(self.device)
        ip_attention_mask = torch.ones((batch_size, 1), device=self.device, dtype=torch.bool)
        
        # Generate the image
        print(f"Generating character: {character_name}")
        
        # Generate the image with direct IP hidden states
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
            ip_hidden_states=None,
            ip_attention_mask=None,
            # Explicitly set these to None
            text_bboxes=None,
            character_bboxes=None,
            reference_embeddings=None,
            cfg_on_10_percent=False
        ).images[0]
        
        # Save the image
        output_path = self.character_images_dir / f"{character_name}.png"
        image.save(output_path)
        
        # Create character card with details
        card_path = self.create_character_card(character_name, character_info, output_path)
        
        # Save character info for embedding
        if regenerate and character_name in self.generated_characters:
            print(f"Regenerated character: {character_name}")
            # If regenerating, update the seed
            self.generated_characters[character_name]["seed"] = seed
        else:
            # Save new character info
            self.generated_characters[character_name] = {
                "name": character_name,
                "image_path": str(output_path),
                "card_path": str(card_path),
                "seed": seed
            }
            print(f"Generated new character: {character_name}")
        
        # Save the updated embedding map
        with open(self.embedding_map_path, 'w') as f:
            json.dump(self.generated_characters, f, indent=2)
        
        return output_path
    
    def create_character_card(self, character_name, character_info, image_path):
        """Create a character card with image and descriptions"""
        # Load image
        char_img = Image.open(image_path)
        
        # Create a larger canvas for the card
        card_width = 800
        card_height = 1000
        card = Image.new('RGB', (card_width, card_height), (255, 255, 255))
        
        # Resize character image to fit the card
        img_width = card_width - 40  # Margins on both sides
        img_height = int(char_img.height * (img_width / char_img.width))
        char_img_resized = char_img.resize((img_width, img_height), Image.LANCZOS)
        
        # Paste character image
        card.paste(char_img_resized, (20, 20))
        
        # Add text descriptions
        draw = ImageDraw.Draw(card)
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except:
            # Fallback to default font if Arial is not available
            font = ImageFont.load_default()
        
        # Add character name
        draw.text((20, img_height + 40), f"Character: {character_name}", fill=(0, 0, 0), font=font)
        
        # Add descriptions
        y_offset = img_height + 80
        descriptions = character_info["descriptions"][:8]  # Limit to 8 descriptions
        for i, desc in enumerate(descriptions):
            desc_text = f"{i+1}. {desc}"
            draw.text((30, y_offset), desc_text, fill=(0, 0, 0), font=font)
            y_offset += 30
        
        # Save the character card
        card_path = self.character_images_dir / f"{character_name}_card.png"
        card.save(card_path)
        return card_path
    
    def regenerate_character(self, character_name, seed=None):
        """Regenerate a character image with an optional new seed"""
        return self.generate_character(character_name, seed, regenerate=True)
    
    def generate_all_characters(self, min_importance=0):
        """Generate images for all characters"""
        characters = self.get_all_characters()
        print(f"Found {len(characters)} named characters.")
        
        for i, character in enumerate(tqdm(characters, desc="Generating characters")):
            print(f"Generating {i+1}/{len(characters)}: {character['name']}")
            # Use a random seed for initial generation
            seed = torch.randint(0, 2**32, (1,)).item()
            self.generate_character(character['name'], seed)
        
        print(f"Generated {len(characters)} characters.")
        print(f"Character images saved to: {self.character_images_dir}")

    def regenerate_non_keepers(self):
        """
        Regenerate all characters that are not present in the 'keepers' folder
        and save them in a timestamped directory.
        """
        # Create a timestamped directory for this batch of generations
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamped_dir = self.output_dir / f"generation_{timestamp}"
        timestamped_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all characters
        all_characters = self.get_all_characters()
        
        # Get list of character names that exist in the keepers folder
        keeper_characters = set()
        if self.keepers_dir.exists():
            for file in os.listdir(self.keepers_dir):
                if file.endswith('.png') and not file.endswith('_card.png'):
                    # Extract character name from filename
                    character_name = os.path.splitext(file)[0]
                    keeper_characters.add(character_name)
        
        # Filter characters that need regeneration (not in keepers)
        characters_to_regenerate = [
            char for char in all_characters 
            if char['name'] not in keeper_characters
        ]
        
        print(f"Found {len(keeper_characters)} characters in keepers folder.")
        print(f"Will regenerate {len(characters_to_regenerate)} characters.")
        
        # Regenerate characters and save to timestamped folder
        for i, character in enumerate(tqdm(characters_to_regenerate, desc="Regenerating characters")):
            print(f"Regenerating {i+1}/{len(characters_to_regenerate)}: {character['name']}")
            
            # Use a random seed for regeneration
            seed = torch.randint(0, 2**32, (1,)).item()
            
            # Generate the character
            image_path = self.generate_character(character['name'], seed)
            
            if image_path:
                # Copy the image and card to timestamped directory
                char_filename = os.path.basename(image_path)
                card_filename = f"{os.path.splitext(char_filename)[0]}_card.png"
                
                # Copy image
                shutil.copy2(
                    image_path, 
                    timestamped_dir / char_filename
                )
                
                # Copy card if it exists
                card_path = self.character_images_dir / card_filename
                if card_path.exists():
                    shutil.copy2(
                        card_path,
                        timestamped_dir / card_filename
                    )
        
        print(f"Regenerated {len(characters_to_regenerate)} characters.")
        print(f"New character images saved to: {timestamped_dir}")
        return timestamped_dir

def main():
    parser = argparse.ArgumentParser(description="Generate manga characters using Drawatoon")
    parser.add_argument("--model_path", type=str, default="./drawatoon-v1",
                        help="Path to the Drawatoon model")
    parser.add_argument("--character_data", type=str, default="charactersNL.json",
                        help="Path to character data JSON file")
    parser.add_argument("--output_dir", type=str, default="character_output",
                        help="Directory to save character images")
    parser.add_argument("--character", type=str, default=None,
                        help="Generate a specific character (by name)")
    parser.add_argument("--regenerate", type=str, default=None,
                        help="Regenerate a specific character (by name)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for generation")
    parser.add_argument("--min_importance", type=int, default=4,
                        help="Minimum number of descriptions for a character to be considered important")
    parser.add_argument("--all_characters", action="store_true",
                        help="Generate all characters, even those with few descriptions")
    parser.add_argument("--regenerate_non_keepers", action="store_true",
                        help="Regenerate all characters not present in the 'keepers' folder")
    
    args = parser.parse_args()
    
    generator = CharacterGenerator(
        model_path=args.model_path,
        character_data_path=args.character_data,
        output_dir=args.output_dir
    )
    
    if args.regenerate_non_keepers:
        # Regenerate all characters not in the keepers folder
        print("Regenerating all characters not in the keepers folder")
        generator.regenerate_non_keepers()
    elif args.regenerate:
        # Regenerate a specific character
        print(f"Regenerating character: {args.regenerate}")
        generator.regenerate_character(args.regenerate, args.seed)
    elif args.character:
        # Generate a specific character
        print(f"Generating character: {args.character}")
        generator.generate_character(args.character, args.seed)
    else:
        # Generate characters based on importance or all
        min_importance = 0 if args.all_characters else args.min_importance
        generator.generate_all_characters(min_importance)

if __name__ == "__main__":
    main()
