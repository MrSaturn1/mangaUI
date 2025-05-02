# mangaui/backend/app.py
from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
import torch
import json
import os
import sys
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
import threading
import queue
import time
import random
import traceback

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

initialization_status = {
    "is_initializing": False,
    "is_initialized": False,
    "progress": 0,
    "message": "Idle",
    "error": None
}

# Default paths - assume drawatoon is in the current directory
DEFAULT_MODEL_PATH = './drawatoon-v1'
DEFAULT_CHARACTER_DATA_PATH = './characters.json'
DEFAULT_CHARACTER_EMBEDDING_PATH = './character_output/character_embeddings/character_embeddings.json'
DEFAULT_OUTPUT_DIR = './manga_output'

def initialize_models_thread(model_path, character_data_path, character_embedding_path, output_dir):
    """Background thread function to initialize models"""
    global manga_generator, character_generator, initialization_status
    
    try:
        initialization_status["is_initializing"] = True
        initialization_status["message"] = "Loading character data..."
        initialization_status["progress"] = 10
        
        # Initialize manga generator
        initialization_status["message"] = "Initializing manga generator..."
        initialization_status["progress"] = 20
        
        manga_generator = MangaGenerator(
            model_path=model_path,
            character_data_path=character_data_path,
            character_embedding_path=character_embedding_path,
            output_dir=output_dir
        )
        
        # Store in app config
        app.config['MANGA_GENERATOR'] = manga_generator
        
        initialization_status["message"] = "Initializing character generator..."
        initialization_status["progress"] = 70
        
        # Initialize character generator
        character_generator = CharacterGenerator(
            model_path=model_path,
            character_data_path=character_data_path,
            output_dir="character_output"
        )
        
        initialization_status["is_initialized"] = True
        initialization_status["is_initializing"] = False
        initialization_status["message"] = "Models initialized successfully"
        initialization_status["progress"] = 100
        
    except Exception as e:
        import traceback
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        initialization_status["is_initializing"] = False
        initialization_status["message"] = "Initialization failed"
        initialization_status["error"] = f"{error_message}\n\n{error_traceback}"
        initialization_status["progress"] = 0
        
        print(f"Error initializing models: {error_message}")
        traceback.print_exc()

@app.route('/api/init', methods=['POST'])
def initialize_models():
    """Initialize the model pipelines in a background thread"""
    global initialization_status
    
    # If already initializing or initialized, just return the status
    if initialization_status["is_initializing"]:
        return jsonify({
            'status': 'in_progress', 
            'message': initialization_status["message"],
            'progress': initialization_status["progress"]
        })
    
    if initialization_status["is_initialized"]:
        return jsonify({
            'status': 'success', 
            'message': 'Models already initialized',
            'progress': 100
        })
    
    # Get initialization parameters
    data = request.json
    model_path = data.get('model_path', DEFAULT_MODEL_PATH)
    character_data_path = data.get('character_data_path', DEFAULT_CHARACTER_DATA_PATH)
    character_embedding_path = data.get('character_embedding_path', DEFAULT_CHARACTER_EMBEDDING_PATH)
    output_dir = data.get('output_dir', DEFAULT_OUTPUT_DIR)
    
    # Start initialization in a background thread
    thread = threading.Thread(
        target=initialize_models_thread,
        args=(model_path, character_data_path, character_embedding_path, output_dir)
    )
    thread.daemon = True  # Thread will exit when main thread exits
    thread.start()
    
    return jsonify({
        'status': 'in_progress',
        'message': 'Model initialization started in background',
        'progress': 5
    })

@app.route('/api/init/status', methods=['GET'])
def get_initialization_status():
    """Get the current status of model initialization"""
    global initialization_status
    
    status = 'success' if initialization_status["is_initialized"] else \
             'in_progress' if initialization_status["is_initializing"] else \
             'error' if initialization_status["error"] else 'idle'
    
    return jsonify({
        'status': status,
        'is_initialized': initialization_status["is_initialized"],
        'is_initializing': initialization_status["is_initializing"],
        'message': initialization_status["message"],
        'progress': initialization_status["progress"],
        'error': initialization_status["error"]
    })

# Add this helper function for other API endpoints
def ensure_models_initialized():
    """Check if models are initialized and return appropriate response if not"""
    global initialization_status
    
    if not initialization_status["is_initialized"]:
        if initialization_status["is_initializing"]:
            return {
                'status': 'in_progress',
                'message': 'Models are still initializing. Please try again later.',
                'progress': initialization_status["progress"]
            }, 202  # Accepted, but processing
        else:
            return {
                'status': 'error',
                'message': 'Models are not initialized. Please initialize models first.',
                'error': initialization_status["error"] if initialization_status["error"] else None
            }, 400  # Bad request
    
    return None

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

# Request queue for panel generation
panel_request_queue = queue.Queue()
panel_response_map = {}  # Map of request_id to response

# Process to handle the panel generation queue
def panel_request_worker():
    global panel_request_queue, panel_response_map, initialization_status
    
    while True:
        # Wait until the model is initialized
        while not initialization_status["is_initialized"]:
            time.sleep(1)
        
        try:
            # Get next request from queue if available
            try:
                request_data = panel_request_queue.get(block=False)
                request_id = request_data.get('request_id')
                panel_data = request_data.get('panel_data')
                
                print(f"Processing queued request {request_id}")
                
                try:
                    # Process the panel generation
                    manga_generator = current_app.config.get('MANGA_GENERATOR')
                    
                    if manga_generator is None:
                        panel_response_map[request_id] = {
                            'status': 'error',
                            'message': 'Manga generator not initialized'
                        }
                        continue
                    
                    # Extract parameters
                    panel_index = panel_data.get('panelIndex', 0)
                    seed = panel_data.get('seed', random.randint(0, 1000000))
                    width = int(panel_data.get('width', 512))
                    height = int(panel_data.get('height', 512))
                    
                    # Generate the panel
                    output_path, updated_panel_data = manga_generator.generate_panel(
                        panel_data=panel_data.get('panel_data'),
                        panel_index=panel_index,
                        seed=seed,
                        width=width,
                        height=height
                    )
                    
                    # Convert image to base64 for sending to frontend
                    with open(output_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    # Save result to response map
                    panel_response_map[request_id] = {
                        'status': 'success',
                        'imageData': f'data:image/png;base64,{img_data}',
                        'prompt': panel_data.get('prompt') or manga_generator.create_panel_prompt(panel_data.get('panel_data')),
                        'panelIndex': panel_index,
                        'seed': seed,
                        'width': width,
                        'height': height
                    }
                    
                except Exception as e:
                    print(f"Error processing queued request {request_id}: {e}")
                    traceback.print_exc()
                    panel_response_map[request_id] = {
                        'status': 'error',
                        'message': str(e)
                    }
                
                finally:
                    # Mark task as done
                    panel_request_queue.task_done()
                
            except queue.Empty:
                # No requests in queue, just sleep a bit
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error in panel request worker: {e}")
            traceback.print_exc()
            time.sleep(1)  # Avoid tight loop on error

# Start the panel request worker thread
panel_worker_thread = threading.Thread(target=panel_request_worker)
panel_worker_thread.daemon = True
panel_worker_thread.start()

@app.route('/api/generate_panel', methods=['POST'])
def generate_panel():
    """API endpoint to generate a panel image or queue it for generation"""
    try:
        # Get JSON data from request
        data = request.json
        
        # Generate a unique request ID
        request_id = f"req_{time.time()}_{random.randint(1000, 9999)}"
        
        # If models are initialized, process directly; otherwise queue
        if initialization_status["is_initialized"]:
            # Extract panel details
            prompt = data.get('prompt', '')
            setting = data.get('setting', '')
            characterNames = data.get('characterNames', [])
            dialogues = data.get('dialogues', [])
            actions = data.get('actions', [])
            characterPositions = data.get('characterPositions', [])
            dialoguePositions = data.get('dialoguePositions', [])
            panelIndex = data.get('panelIndex', 0)
            seed = data.get('seed', random.randint(0, 1000000))
            
            # Extract panel dimensions from the request
            panel_width = int(data.get('width', 0))
            panel_height = int(data.get('height', 0))
            
            # If dimensions are not provided or are invalid, use reasonable defaults
            if panel_width <= 0 or panel_height <= 0:
                print("Warning: Invalid panel dimensions received from frontend. Using defaults.")
                panel_width = 512
                panel_height = 512
            
            # Create a panel data object similar to what the MangaGenerator expects
            panel_data = {
                'setting': setting,
                'characters': set(characterNames),
                'elements': [],
                'scene_index': 0  # This might not be relevant for UI-generated panels
            }
            
            # Add dialogue elements
            for i, dialogue in enumerate(dialogues):
                if dialogue['character'] and dialogue['text']:
                    dialogue_element = {
                        'type': 'dialogue',
                        'character': dialogue['character'],
                        'dialogue': dialogue['text'],
                        'characters_present': [dialogue['character']]
                    }
                    panel_data['elements'].append(dialogue_element)
            
            # Add action elements
            for action in actions:
                if action['text']:
                    action_element = {
                        'type': 'action',
                        'text': action['text'],
                        'characters_present': characterNames
                    }
                    panel_data['elements'].append(action_element)
            
            # If no elements were added but we have a prompt, create a generic action
            if not panel_data['elements'] and prompt:
                panel_data['elements'].append({
                    'type': 'action',
                    'text': prompt,
                    'characters_present': characterNames
                })
            
            # Generate the panel using MangaGenerator
            manga_generator = current_app.config['MANGA_GENERATOR']
            
            output_path, updated_panel_data = manga_generator.generate_panel(
                panel_data=panel_data,
                panel_index=panelIndex,
                seed=seed,
                width=panel_width,
                height=panel_height
            )
            
            # Convert image to base64 for sending to frontend
            with open(output_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Return the image data and updated panel information
            return jsonify({
                'status': 'success',
                'imageData': f'data:image/png;base64,{img_data}',
                'prompt': prompt or manga_generator.create_panel_prompt(panel_data),
                'panelIndex': panelIndex,
                'seed': seed,
                'width': panel_width,
                'height': panel_height,
                'request_id': request_id
            })
            
        else:
            # Models not initialized yet, queue the request
            panel_request = {
                'request_id': request_id,
                'panel_data': {
                    'prompt': data.get('prompt', ''),
                    'setting': data.get('setting', ''),
                    'characterNames': data.get('characterNames', []),
                    'dialogues': data.get('dialogues', []),
                    'actions': data.get('actions', []),
                    'characterPositions': data.get('characterPositions', []),
                    'dialoguePositions': data.get('dialoguePositions', []),
                    'panelIndex': data.get('panelIndex', 0),
                    'seed': data.get('seed', random.randint(0, 1000000)),
                    'width': int(data.get('width', 512)),
                    'height': int(data.get('height', 512)),
                    'panel_data': {
                        'setting': data.get('setting', ''),
                        'characters': set(data.get('characterNames', [])),
                        'elements': [],
                        'scene_index': 0
                    }
                }
            }
            
            # Add dialogue elements
            for i, dialogue in enumerate(data.get('dialogues', [])):
                if dialogue.get('character') and dialogue.get('text'):
                    dialogue_element = {
                        'type': 'dialogue',
                        'character': dialogue['character'],
                        'dialogue': dialogue['text'],
                        'characters_present': [dialogue['character']]
                    }
                    panel_request['panel_data']['panel_data']['elements'].append(dialogue_element)
            
            # Add action elements
            for action in data.get('actions', []):
                if action.get('text'):
                    action_element = {
                        'type': 'action',
                        'text': action['text'],
                        'characters_present': data.get('characterNames', [])
                    }
                    panel_request['panel_data']['panel_data']['elements'].append(action_element)
            
            # If no elements were added but we have a prompt, create a generic action
            if not panel_request['panel_data']['panel_data']['elements'] and data.get('prompt'):
                panel_request['panel_data']['panel_data']['elements'].append({
                    'type': 'action',
                    'text': data.get('prompt'),
                    'characters_present': data.get('characterNames', [])
                })
            
            # Add to queue
            panel_request_queue.put(panel_request)
            
            # Return status - the request is queued
            initialization_progress = initialization_status["progress"]
            return jsonify({
                'status': 'queued',
                'message': 'Panel generation queued. Models still initializing.',
                'request_id': request_id,
                'progress': initialization_progress,
                'eta_seconds': max(5, int((100 - initialization_progress) * 0.5))
            })
            
    except Exception as e:
        print(f"Error handling generate_panel request: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@app.route('/api/check_panel_status', methods=['GET'])
def check_panel_status():
    """Check if a queued panel generation is complete"""
    request_id = request.args.get('request_id')
    
    if not request_id:
        return jsonify({
            'status': 'error',
            'message': 'No request ID provided'
        }), 400
    
    # Check if we have a response for this request
    if request_id in panel_response_map:
        # Get response and remove from map
        response = panel_response_map.pop(request_id)
        return jsonify(response)
    
    # Still processing
    return jsonify({
        'status': 'processing',
        'message': 'Panel is still being generated',
        'progress': initialization_status["progress"]
    })

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
    
@app.route('/api/status', methods=['GET'])
def check_status():
    """Check if models are initialized"""
    global initialization_status
    
    return jsonify({
        'status': 'success',
        'initialized': initialization_status["is_initialized"],
        'initializing': initialization_status["is_initializing"],
        'message': initialization_status["message"],
        'progress': initialization_status["progress"]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)