# mangaui/backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import json
import os
import sys
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path

# Add the current directory to the path so we can import manga_generator and character_generator
sys.path.append('.')

# Import your classes
from manga_generator import MangaGenerator, ScreenplayParser
from character_generator import CharacterGenerator

app = Flask(__name__)
CORS(app)

# Global variables for model instances
manga_generator = None
character_generator = None

# Default paths - assume drawatoon is in the current directory
DEFAULT_MODEL_PATH = './drawatoon-v1'
DEFAULT_CHARACTER_DATA_PATH = './characters.json'
DEFAULT_CHARACTER_EMBEDDING_PATH = './character_output/character_embeddings/character_embeddings.json'
DEFAULT_OUTPUT_DIR = './manga_output'

@app.route('/api/init', methods=['POST'])
def initialize_models():
    """Initialize the model pipelines"""
    global manga_generator, character_generator
    
    data = request.json
    model_path = data.get('model_path', DEFAULT_MODEL_PATH)
    character_data_path = data.get('character_data_path', DEFAULT_CHARACTER_DATA_PATH)
    character_embedding_path = data.get('character_embedding_path', DEFAULT_CHARACTER_EMBEDDING_PATH)
    output_dir = data.get('output_dir', DEFAULT_OUTPUT_DIR)
    
    try:
        # Initialize the manga generator if not already initialized
        if manga_generator is None:
            manga_generator = MangaGenerator(
                model_path=model_path,
                character_data_path=character_data_path,
                character_embedding_path=character_embedding_path,
                output_dir=output_dir
            )
        
        # Initialize the character generator if not already initialized
        if character_generator is None:
            character_generator = CharacterGenerator(
                model_path=model_path,
                character_data_path=character_data_path,
                output_dir="character_output"
            )
        
        return jsonify({'status': 'success', 'message': 'Models initialized successfully'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/get_characters', methods=['GET'])
def get_characters():
    """Get all available characters and their embeddings"""
    global manga_generator, character_generator
    
    if manga_generator is None:
        return jsonify({'status': 'error', 'message': 'Models not initialized'}), 400
    
    try:
        # Load character data
        with open(manga_generator.character_data_path, 'r', encoding='utf-8') as f:
            characters_data = json.load(f)
        
        # Get embeddings map
        embedding_map = {}
        embedding_map_path = os.path.join('character_output', 'character_embeddings', 'character_embeddings_map.json')
        if os.path.exists(embedding_map_path):
            with open(embedding_map_path, 'r') as f:
                embedding_map = json.load(f)
        
        # Prepare response data
        characters = []
        for char in characters_data:
            name = char.get('name', '')
            if not name:
                continue
                
            has_embedding = name in embedding_map
            
            # Try to get image data if available
            image_data = None
            keeper_path = os.path.join('character_output', 'character_images', 'keepers', f"{name}.png")
            
            if os.path.exists(keeper_path):
                with open(keeper_path, 'rb') as img_file:
                    image_bytes = img_file.read()
                    image_data = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
            
            characters.append({
                'name': name,
                'descriptions': char.get('descriptions', []),
                'hasEmbedding': has_embedding,
                'imageData': image_data
            })
                
        return jsonify({
            'status': 'success', 
            'characters': characters
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/generate_character', methods=['POST'])
def generate_character():
    """Generate a character with the given name and seed"""
    global character_generator
    
    if character_generator is None:
        return jsonify({'status': 'error', 'message': 'Models not initialized'}), 400
    
    data = request.json
    character_name = data.get('name')
    seed = data.get('seed')
    regenerate = data.get('regenerate', False)
    
    if not character_name:
        return jsonify({'status': 'error', 'message': 'Character name is required'}), 400
    
    try:
        # Generate or regenerate the character
        if regenerate:
            output_path = character_generator.regenerate_character(character_name, seed)
        else:
            output_path = character_generator.generate_character(character_name, seed)
        
        # Read the output image
        with open(output_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'imageData': f"data:image/png;base64,{img_data}",
            'name': character_name,
            'seed': seed
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/save_to_keepers', methods=['POST'])
def save_to_keepers():
    """Save a character to the keepers folder"""
    data = request.json
    character_name = data.get('name')
    
    if not character_name:
        return jsonify({'status': 'error', 'message': 'Character name is required'}), 400
    
    try:
        # Check if the character image exists
        character_path = os.path.join('character_output', 'character_images', f"{character_name}.png")
        keeper_path = os.path.join('character_output', 'character_images', 'keepers', f"{character_name}.png")
        
        if not os.path.exists(character_path):
            return jsonify({'status': 'error', 'message': f'Character image not found: {character_name}'}), 404
        
        # Create the keepers directory if it doesn't exist
        os.makedirs(os.path.dirname(keeper_path), exist_ok=True)
        
        # Copy the character image to the keepers folder
        from shutil import copyfile
        copyfile(character_path, keeper_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Character {character_name} saved to keepers'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/generate_panel', methods=['POST'])
def generate_panel():
    """Generate a manga panel with custom positions and prompts"""
    global manga_generator
    
    if manga_generator is None:
        return jsonify({'status': 'error', 'message': 'Models not initialized'}), 400
    
    data = request.json
    prompt = data.get('prompt', '')
    setting = data.get('setting', '')
    
    # Parse the positioning data
    dialogue_positions = data.get('dialoguePositions', [])
    character_positions = data.get('characterPositions', [])
    character_names = data.get('characterNames', [])
    panel_index = data.get('panelIndex', 0)
    seed = data.get('seed')
    
    try:
        # Create the custom panel data structure
        panel_data = {
            'setting': setting,
            'elements': [],
            'characters': set(character_names),
            'scene_index': data.get('sceneIndex', 0)
        }
        
        # Add dialogue elements
        for dialogue in data.get('dialogues', []):
            panel_data['elements'].append({
                'type': 'dialogue',
                'character': dialogue.get('character', ''),
                'dialogue': [dialogue.get('text', '')],
                'characters_present': [dialogue.get('character', '')]
            })
        
        # Add action elements
        for action in data.get('actions', []):
            panel_data['elements'].append({
                'type': 'action',
                'text': action.get('text', ''),
                'characters_present': character_names
            })
        
        # Create a custom prepare_ip_adapter_inputs function
        def custom_prepare_ip_adapter_inputs(panel_data):
            # Initialize lists for both text boxes and character boxes
            text_bboxes = []
            character_bboxes = []
            reference_embeddings = []
            
            # Add dialogue positions
            for pos in dialogue_positions:
                text_bboxes.append([
                    pos.get('x', 0), 
                    pos.get('y', 0), 
                    pos.get('x', 0) + pos.get('width', 0.3), 
                    pos.get('y', 0) + pos.get('height', 0.2)
                ])
            
            # Add character positions and get embeddings
            for i, (name, pos) in enumerate(zip(character_names, character_positions)):
                if pos:  # Only add if position is defined
                    character_bboxes.append([
                        pos.get('x', 0), 
                        pos.get('y', 0), 
                        pos.get('x', 0) + pos.get('width', 0.6), 
                        pos.get('y', 0) + pos.get('height', 0.6)
                    ])
                    
                    # Get character embedding
                    embedding = manga_generator.get_character_embedding(name)
                    if embedding is not None:
                        reference_embeddings.append(embedding)
            
            # Always wrap in batch format for consistent API
            return [text_bboxes], [character_bboxes], [reference_embeddings]
        
        # Save the original method
        original_prepare_ip_adapter_inputs = manga_generator.prepare_ip_adapter_inputs
        
        # Override the method
        manga_generator.prepare_ip_adapter_inputs = custom_prepare_ip_adapter_inputs
        
        # If a custom prompt is not provided, generate one from panel data
        if not prompt:
            prompt = manga_generator.create_panel_prompt(panel_data)
        
        # Generate the panel
        output_path, panel_data = manga_generator.generate_panel(panel_data, panel_index, seed)
        
        # Restore the original method
        manga_generator.prepare_ip_adapter_inputs = original_prepare_ip_adapter_inputs
        
        # Read the output image
        with open(output_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'imageData': f"data:image/png;base64,{img_data}",
            'panelIndex': panel_index,
            'prompt': prompt
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/create_page', methods=['POST'])
def create_page():
    """Create a manga page from multiple panels"""
    global manga_generator
    
    if manga_generator is None:
        return jsonify({'status': 'error', 'message': 'Models not initialized'}), 400
    
    data = request.json
    panel_indices = data.get('panelIndices', [])
    layout = data.get('layout', 'grid')  # grid, vertical, custom
    page_index = data.get('pageIndex', 0)
    
    try:
        # Create panel paths list in the format expected by MangaGenerator
        panel_paths = []
        for idx in panel_indices:
            panel_path = manga_generator.panels_dir / f"panel_{idx:04d}.png"
            if panel_path.exists():
                # Load panel data if available
                panel_data_path = manga_generator.panels_dir / f"panel_{idx:04d}.json"
                panel_data = {}
                if panel_data_path.exists():
                    with open(panel_data_path, 'r') as f:
                        panel_data = json.load(f)
                
                panel_paths.append({
                    'path': panel_path,
                    'data': panel_data
                })
        
        if not panel_paths:
            return jsonify({'status': 'error', 'message': 'No valid panels found'}), 400
        
        # Generate the page
        page_path = manga_generator._create_page_layout(panel_paths, page_index)
        
        # Read the output page
        with open(page_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'imageData': f"data:image/png;base64,{img_data}",
            'pageIndex': page_index
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/parse_screenplay', methods=['POST'])
def parse_screenplay():
    """Parse a screenplay file and return scene information"""
    data = request.json
    screenplay_path = data.get('screenplay_path', './the-rat.txt')
    character_data_path = data.get('character_data_path', './characters.json')
    
    if not os.path.exists(screenplay_path):
        return jsonify({'status': 'error', 'message': 'Screenplay file not found'}), 400
    
    if not os.path.exists(character_data_path):
        return jsonify({'status': 'error', 'message': 'Character data file not found'}), 400
    
    try:
        parser = ScreenplayParser(screenplay_path, character_data_path)
        scenes = parser.parse()
        
        # Prepare response data (convert sets to lists)
        serialized_scenes = []
        for i, scene in enumerate(scenes):
            # Make a deep copy of the scene to avoid modifying the original
            scene_copy = {
                'location': scene['location'],
                'time': scene['time'],
                'interior_exterior': scene['interior_exterior'],
                'estimated_panels': scene['estimated_panels'],
                'elements': [],
                'index': i
            }
            
            for element in scene['elements']:
                element_copy = element.copy()
                if element['type'] == 'dialogue':
                    # Combine dialogue lines for the frontend
                    element_copy['dialogue_text'] = ' '.join(element['dialogue'])
                scene_copy['elements'].append(element_copy)
            
            serialized_scenes.append(scene_copy)
        
        return jsonify({
            'status': 'success',
            'scenes': serialized_scenes
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/get_generated_panels', methods=['GET'])
def get_generated_panels():
    """Get all generated panels"""
    global manga_generator
    
    if manga_generator is None:
        return jsonify({'status': 'error', 'message': 'Models not initialized'}), 400
    
    try:
        panels = []
        panels_dir = manga_generator.panels_dir
        
        # Limit to first 50 panels to avoid overloading
        for i, panel_file in enumerate(sorted(panels_dir.glob('panel_*.png'))):
            if i >= 50:  # Limit to 50 panels
                break
                
            panel_index = int(panel_file.stem.split('_')[1])
            
            # Load panel data if available
            panel_data_path = panels_dir / f"{panel_file.stem}.json"
            panel_data = {}
            if panel_data_path.exists():
                with open(panel_data_path, 'r') as f:
                    panel_data = json.load(f)
            
            # Create thumbnail
            with open(panel_file, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            panels.append({
                'index': panel_index,
                'imageData': f"data:image/png;base64,{img_data}",
                'data': panel_data
            })
            
        return jsonify({
            'status': 'success',
            'panels': panels
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)