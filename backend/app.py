# mangaui/backend/app.py
from flask import Flask, request, jsonify, current_app, send_file
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
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

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
                    project_id = panel_data.get('projectId', 'default')
                    panel_index = panel_data.get('panelIndex', 0)
                    seed = panel_data.get('seed', random.randint(0, 1000000))
                    width = int(panel_data.get('width', 512))
                    height = int(panel_data.get('height', 512))

                    # Create project directories if they don't exist
                    project_output_dir = Path("manga_projects") / project_id
                    project_panels_dir = project_output_dir / "panels"
                    project_panels_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate the panel
                    output_path, updated_panel_data = manga_generator.generate_panel(
                        panel_data=panel_data.get('panel_data'),
                        panel_index=panel_index,
                        seed=seed,
                        width=width,
                        height=height,
                        project_id=project_id  # Pass project ID
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
                        'height': height,
                        'projectId': project_id
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

        # Get project ID or use 'default' if not provided
        project_id = data.get('projectId', 'default')
        
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
            characterBoxes = data.get('characterBoxes', [])
            textBoxes = data.get('textBoxes', [])
            
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
                'scene_index': 0,  # This might not be relevant for UI-generated panels
                'textBoxes': textBoxes,  # Add text boxes data
                'characterBoxes': characterBoxes  # Add character boxes data
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
            
            # Create project directories if they don't exist
            project_output_dir = Path("manga_projects") / project_id
            project_panels_dir = project_output_dir / "panels"
            project_panels_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate panel and save to project directory
            output_path, updated_panel_data = manga_generator.generate_panel(
                panel_data=panel_data,
                panel_index=panelIndex,
                seed=seed,
                width=panel_width,
                height=panel_height,
                project_id=project_id  # Pass project ID to generate_panel
            )
            
            # Convert image to base64 for sending to frontend
            with open(output_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Return the image data and updated panel information
            return jsonify({
                'status': 'success',
                'imageData': f'data:image/png;base64,{img_data}',
                'imagePath': f'/api/images/{project_id}/panels/panel_{panelIndex:04d}.png',
                'prompt': prompt or manga_generator.create_panel_prompt(panel_data),
                'panelIndex': panelIndex,
                'seed': seed,
                'width': panel_width,
                'height': panel_height,
                'request_id': request_id,
                'projectId': project_id
            })
            
        else:
            # Models not initialized yet, queue the request
            panel_request = {
                'request_id': request_id,
                'panel_data': {
                    'projectId': project_id,
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
    project_id = data.get('projectId', 'default')
    
    try:
        # Create project directories if they don't exist
        project_output_dir = Path("manga_projects") / project_id
        project_panels_dir = project_output_dir / "panels"
        project_pages_dir = project_output_dir / "pages"
        project_pages_dir.mkdir(parents=True, exist_ok=True)
        
        # Create panel paths list in the format expected by MangaGenerator
        panel_paths = []
        for idx in panel_indices:
            panel_path = project_panels_dir / f"panel_{idx:04d}.png"
            if panel_path.exists():
                # Load panel data if available
                panel_data_path = project_panels_dir / f"panel_{idx:04d}.json"
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
        
        # Generate the page and save to project directory
        page_path = project_pages_dir / f"page_{page_index:03d}.png"
        # Call a modified version of _create_page_layout that saves to project directory
        # or modify the function to accept a custom output path
        page_path = manga_generator._create_page_layout(panel_paths, page_index, output_path=page_path)
        
        # Read the output page
        with open(page_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'imageData': f"data:image/png;base64,{img_data}",
            'pageIndex': page_index,
            'projectId': project_id
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
    """Get all generated panels for a specific project"""
    global manga_generator
    
    if manga_generator is None:
        return jsonify({'status': 'error', 'message': 'Models not initialized'}), 400
    
    project_id = request.args.get('projectId', 'default')
    
    try:
        panels = []
        project_panels_dir = Path("manga_projects") / project_id / "panels"
        
        if not project_panels_dir.exists():
            return jsonify({
                'status': 'success',
                'panels': []
            })
        
        # Limit to first 50 panels to avoid overloading
        for i, panel_file in enumerate(sorted(project_panels_dir.glob('panel_*.png'))):
            if i >= 50:  # Limit to 50 panels
                break
                
            panel_index = int(panel_file.stem.split('_')[1])
            
            # Load panel data if available
            panel_data_path = project_panels_dir / f"{panel_file.stem}.json"
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
                'data': panel_data,
                'projectId': project_id
            })
            
        return jsonify({
            'status': 'success',
            'panels': panels
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/images/<project_id>/<filename>', methods=['GET'])
def get_image(project_id, filename):
    """Retrieve an image by its path"""
    try:
        image_path = Path("manga_projects") / project_id / "panels" / filename
        
        if not image_path.exists():
            # Check if it's in pages folder
            image_path = Path("manga_projects") / project_id / "pages" / filename
            if not image_path.exists():
                return jsonify({
                    'status': 'error',
                    'message': 'Image not found'
                }), 404
        
        return send_file(image_path.as_posix(), mimetype='image/png')
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
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

# PROJECT MANAGEMENT ROUTES

@app.route('/api/projects', methods=['GET'])
def get_projects():
    """Get all manga projects"""
    try:
        projects_dir = Path("manga_projects")
        projects_dir.mkdir(exist_ok=True)
        
        projects_file = projects_dir / "projects.json"
        
        if projects_file.exists():
            with open(projects_file, 'r') as f:
                projects = json.load(f)
        else:
            projects = []
            
        return jsonify({
            'status': 'success',
            'projects': projects
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/projects', methods=['POST'])
def create_project():
    """Create a new manga project"""
    try:
        data = request.json
        name = data.get('name')
        
        if not name:
            return jsonify({
                'status': 'error',
                'message': 'Project name is required'
            }), 400
            
        projects_dir = Path("manga_projects")
        projects_dir.mkdir(exist_ok=True)
        
        projects_file = projects_dir / "projects.json"
        
        if projects_file.exists():
            with open(projects_file, 'r') as f:
                projects = json.load(f)
        else:
            projects = []
            
        # Create new project
        project_id = f"project_{int(time.time())}"
        new_project = {
            'id': project_id,
            'name': name,
            'pages': 0,
            'lastModified': datetime.datetime.now().isoformat()
        }
        
        projects.append(new_project)
        
        # Save projects metadata
        with open(projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
            
        # Create project directory
        project_dir = projects_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        return jsonify({
            'status': 'success',
            'project': new_project
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get a specific project and its pages"""
    try:
        projects_dir = Path("manga_projects")
        projects_file = projects_dir / "projects.json"
        
        if not projects_file.exists():
            return jsonify({
                'status': 'error',
                'message': 'No projects found'
            }), 404
            
        with open(projects_file, 'r') as f:
            projects = json.load(f)
            
        project = next((p for p in projects if p['id'] == project_id), None)
        
        if not project:
            return jsonify({
                'status': 'error',
                'message': f'Project {project_id} not found'
            }), 404
            
        # Load project pages
        project_dir = projects_dir / project_id
        pages_file = project_dir / "pages.json"
        
        if pages_file.exists():
            with open(pages_file, 'r') as f:
                pages = json.load(f)
        else:
            pages = []
            
        return jsonify({
            'status': 'success',
            'project': project,
            'pages': pages
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/projects/<project_id>', methods=['PUT'])
def update_project(project_id):
    """Update a project's pages"""
    try:
        data = request.json
        pages = data.get('pages', [])
        
        projects_dir = Path("manga_projects")
        projects_file = projects_dir / "projects.json"
        
        if not projects_file.exists():
            return jsonify({
                'status': 'error',
                'message': 'No projects found'
            }), 404
            
        with open(projects_file, 'r') as f:
            projects = json.load(f)
            
        project_index = next((i for i, p in enumerate(projects) if p['id'] == project_id), None)
        
        if project_index is None:
            return jsonify({
                'status': 'error',
                'message': f'Project {project_id} not found'
            }), 404
            
        # Update project metadata
        projects[project_index]['pages'] = len(pages)
        projects[project_index]['lastModified'] = datetime.datetime.now().isoformat()
        
        # Save updated projects metadata
        with open(projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
            
        # Save project pages
        project_dir = projects_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        pages_file = project_dir / "pages.json"
        with open(pages_file, 'w') as f:
            json.dump(pages, f, indent=2)
            
        return jsonify({
            'status': 'success',
            'project': projects[project_index]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project"""
    try:
        projects_dir = Path("manga_projects")
        projects_file = projects_dir / "projects.json"
        
        if not projects_file.exists():
            return jsonify({
                'status': 'error',
                'message': 'No projects found'
            }), 404
            
        with open(projects_file, 'r') as f:
            projects = json.load(f)
            
        project_index = next((i for i, p in enumerate(projects) if p['id'] == project_id), None)
        
        if project_index is None:
            return jsonify({
                'status': 'error',
                'message': f'Project {project_id} not found'
            }), 404
            
        # Remove project from list
        deleted_project = projects.pop(project_index)
        
        # Save updated projects metadata
        with open(projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
            
        # Delete project directory
        project_dir = projects_dir / project_id
        if project_dir.exists():
            import shutil
            shutil.rmtree(project_dir)
            
        return jsonify({
            'status': 'success',
            'message': f'Project {deleted_project["name"]} deleted successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@app.route('/api/export/pdf', methods=['POST'])
def export_pdf():
    """Generate a PDF from manga pages using standard manga dimensions (5" × 7.5")"""
    try:
        data = request.json
        project_id = data.get('projectId', 'default')
        project_name = data.get('projectName', 'Untitled')
        page_images = data.get('pages', [])
        quality = data.get('quality', 'normal')
        
        if not page_images:
            return jsonify({
                'status': 'error',
                'message': 'No page images provided'
            }), 400
        
        # Create exports directory if it doesn't exist
        exports_dir = Path("manga_projects") / project_id / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{project_name.replace(' ', '_')}_{timestamp}.pdf"
        pdf_path = exports_dir / filename
        
        # Import PDF library
        from reportlab.lib.pagesizes import inch
        from reportlab.pdfgen import canvas
        from PIL import Image
        import io
        
        # Standard manga dimensions: 5" × 7.5"
        manga_width = 5 * inch
        manga_height = 7.5 * inch
        manga_size = (manga_width, manga_height)
        
        # Determine quality settings
        dpi = 300 if quality == 'high' else 150
        
        # Create PDF
        c = canvas.Canvas(pdf_path.as_posix(), pagesize=manga_size)
        
        for page_data in page_images:
            page_index = page_data.get('pageIndex', 0)
            data_url = page_data.get('dataURL')
            
            if not data_url:
                continue
            
            # Extract base64 data from data URL
            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            
            # Load image with PIL
            img = Image.open(io.BytesIO(binary_data))
            
            # Calculate scaling to fit manga page with small margin
            margin = 0.125 * inch  # 1/8 inch margin
            content_width = manga_width - (2 * margin)
            content_height = manga_height - (2 * margin)
            
            # Maintain aspect ratio
            img_ratio = img.width / img.height
            page_ratio = content_width / content_height
            
            if img_ratio > page_ratio:  # Image is wider than the page ratio
                scaled_width = content_width
                scaled_height = content_width / img_ratio
            else:  # Image is taller than the page ratio
                scaled_height = content_height
                scaled_width = content_height * img_ratio
            
            # Calculate position to center the image
            x_position = margin + (content_width - scaled_width) / 2
            y_position = manga_height - margin - scaled_height  # PDF origin is at bottom left
            
            # Add image to PDF
            c.drawImage(
                ImageReader(img),
                x_position,
                y_position,
                width=scaled_width,
                height=scaled_height
            )
            
            # Move to next page
            c.showPage()
        
        # Save PDF
        c.save()
        
        # Return download URL
        return jsonify({
            'status': 'success',
            'downloadUrl': f'/api/downloads/{project_id}/{filename}'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@app.route('/api/downloads/<project_id>/<filename>', methods=['GET'])
def download_file(project_id, filename):
    """Download a generated file (PDF, etc.)"""
    try:
        file_path = Path("manga_projects") / project_id / "exports" / filename
        
        if not file_path.exists():
            return jsonify({
                'status': 'error',
                'message': 'File not found'
            }), 404
        
        return send_file(
            file_path.as_posix(),
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf' if filename.endswith('.pdf') else 'application/octet-stream'
        )
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)