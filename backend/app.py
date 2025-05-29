# mangaui/backend/app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response, File, UploadFile, Query, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Set
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
import uvicorn
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from contextlib import asynccontextmanager
import numpy as np

# Add the current directory to the path so we can import manga_generator and character_generator
sys.path.append('.')

# Import your classes
from manga_generator import MangaGenerator, ScreenplayParser
from character_generator import CharacterGenerator

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the worker thread at startup
    thread = threading.Thread(target=panel_request_worker, daemon=True)
    thread.start()
    yield
    # Cleanup can go here (if needed)

app = FastAPI(
    title="MangaUI API", 
    description="Backend API for MangaUI - Generate manga panels and characters",
    lifespan=lifespan  # Add the lifespan parameter here
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

# Request queue for panel generation
panel_request_queue = queue.Queue()
panel_response_map = {}  # Map of request_id to response

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)  # Convert sets to lists
        return super().default(obj)

# Pydantic models for request/response data
class InitRequest(BaseModel):
    model_path: Optional[str] = DEFAULT_MODEL_PATH
    character_data_path: Optional[str] = DEFAULT_CHARACTER_DATA_PATH
    character_embedding_path: Optional[str] = DEFAULT_CHARACTER_EMBEDDING_PATH
    output_dir: Optional[str] = DEFAULT_OUTPUT_DIR

class StatusResponse(BaseModel):
    status: str
    is_initialized: bool
    is_initializing: bool
    message: str
    progress: int
    error: Optional[str] = None

class GenerateCharacterRequest(BaseModel):
    name: str
    seed: Optional[int] = None
    regenerate: Optional[bool] = False

class SaveToKeepersRequest(BaseModel):
    name: str

class DialogueItem(BaseModel):
    character: str
    text: str

class ActionItem(BaseModel):
    text: str

class Position(BaseModel):
    x: float
    y: float
    width: float
    height: float

class TextBox(BaseModel):
    text: str
    x: float
    y: float
    width: float
    height: float

class CharacterBox(BaseModel):
    character: str
    x: float
    y: float
    width: float
    height: float
    color: str

class PanelGenerateRequest(BaseModel):
    projectId: Optional[str] = "default"
    prompt: Optional[str] = ""
    setting: Optional[str] = ""
    characterNames: List[str] = []
    dialogues: List[DialogueItem] = []
    actions: List[ActionItem] = []
    characterBoxes: Optional[List[CharacterBox]] = []
    textBoxes: Optional[List[TextBox]] = []
    panelIndex: Optional[int] = 0
    seed: Optional[int] = None
    width: Optional[int] = 512
    height: Optional[int] = 512

class CreateProjectRequest(BaseModel):
    name: str

class UpdateProjectRequest(BaseModel):
    pages: List[Dict[str, Any]] = []

class CreatePageRequest(BaseModel):
    panelIndices: List[int] = []
    layout: Optional[str] = "grid"
    pageIndex: Optional[int] = 0
    projectId: Optional[str] = "default"

class ParseScreenplayRequest(BaseModel):
    screenplay_path: Optional[str] = './the-rat.txt'
    character_data_path: Optional[str] = './characters.json'

class ExportPdfRequest(BaseModel):
    projectId: Optional[str] = "default"
    projectName: Optional[str] = "Untitled"
    pages: List[Dict[str, Any]] = []
    quality: Optional[str] = "normal"

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
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        initialization_status["is_initializing"] = False
        initialization_status["message"] = "Initialization failed"
        initialization_status["error"] = f"{error_message}\n\n{error_traceback}"
        initialization_status["progress"] = 0
        
        print(f"Error initializing models: {error_message}")
        traceback.print_exc()

@app.post("/api/init", response_model=Dict[str, Any])
async def initialize_models(request: InitRequest):
    """Initialize the model pipelines in a background thread"""
    global initialization_status
    
    # If already initializing or initialized, just return the status
    if initialization_status["is_initializing"]:
        return {
            'status': 'in_progress', 
            'message': initialization_status["message"],
            'progress': initialization_status["progress"]
        }
    
    if initialization_status["is_initialized"]:
        return {
            'status': 'success', 
            'message': 'Models already initialized',
            'progress': 100
        }
    
    # Start initialization in a background thread
    thread = threading.Thread(
        target=initialize_models_thread,
        args=(request.model_path, request.character_data_path, request.character_embedding_path, request.output_dir)
    )
    thread.daemon = True  # Thread will exit when main thread exits
    thread.start()
    
    return {
        'status': 'in_progress',
        'message': 'Model initialization started in background',
        'progress': 5
    }

@app.get("/api/init/status", response_model=StatusResponse)
async def get_initialization_status():
    """Get the current status of model initialization"""
    global initialization_status
    
    status = 'success' if initialization_status["is_initialized"] else \
             'in_progress' if initialization_status["is_initializing"] else \
             'error' if initialization_status["error"] else 'idle'
    
    return {
        'status': status,
        'is_initialized': initialization_status["is_initialized"],
        'is_initializing': initialization_status["is_initializing"],
        'message': initialization_status["message"],
        'progress': initialization_status["progress"],
        'error': initialization_status["error"]
    }

# Helper function for other API endpoints
def ensure_models_initialized():
    """Check if models are initialized and return error if not"""
    global initialization_status
    
    if not initialization_status["is_initialized"]:
        if initialization_status["is_initializing"]:
            raise HTTPException(
                status_code=202,  # Accepted, but processing
                detail={
                    'status': 'in_progress',
                    'message': 'Models are still initializing. Please try again later.',
                    'progress': initialization_status["progress"]
                }
            )
        else:
            raise HTTPException(
                status_code=400,  # Bad request
                detail={
                    'status': 'error',
                    'message': 'Models are not initialized. Please initialize models first.',
                    'error': initialization_status["error"] if initialization_status["error"] else None
                }
            )

@app.get("/api/get_characters", response_model=Dict[str, Any])
async def get_characters():
    """Get all available characters and their embeddings"""
    global manga_generator, character_generator
    
    if manga_generator is None:
        raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Models not initialized'})
    
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
                
        return {
            'status': 'success', 
            'characters': characters
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

@app.post("/api/generate_character", response_model=Dict[str, Any])
async def generate_character(request: GenerateCharacterRequest):
    """Generate a character with the given name and seed"""
    global character_generator
    
    if character_generator is None:
        raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Models not initialized'})
    
    character_name = request.name
    seed = request.seed
    regenerate = request.regenerate
    
    if not character_name:
        raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Character name is required'})
    
    try:
        # Generate or regenerate the character
        if regenerate:
            output_path = character_generator.regenerate_character(character_name, seed)
        else:
            output_path = character_generator.generate_character(character_name, seed)
        
        # Read the output image
        with open(output_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return {
            'status': 'success',
            'imageData': f"data:image/png;base64,{img_data}",
            'name': character_name,
            'seed': seed
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

@app.post("/api/save_to_keepers", response_model=Dict[str, Any])
async def save_to_keepers(request: SaveToKeepersRequest):
    """Save a character to the keepers folder"""
    character_name = request.name
    
    if not character_name:
        raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Character name is required'})
    
    try:
        # Check if the character image exists
        character_path = os.path.join('character_output', 'character_images', f"{character_name}.png")
        keeper_path = os.path.join('character_output', 'character_images', 'keepers', f"{character_name}.png")
        
        if not os.path.exists(character_path):
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': f'Character image not found: {character_name}'})
        
        # Create the keepers directory if it doesn't exist
        os.makedirs(os.path.dirname(keeper_path), exist_ok=True)
        
        # Copy the character image to the keepers folder
        from shutil import copyfile
        copyfile(character_path, keeper_path)
        
        return {
            'status': 'success',
            'message': f'Character {character_name} saved to keepers'
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

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
                    
                    # Get text_bboxes and character_bboxes
                    text_bboxes = []
                    if panel_data.get('textBoxes'):
                        for box in panel_data.get('textBoxes'):
                            text_bboxes.append([
                                box['x'], 
                                box['y'], 
                                box['x'] + box['width'], 
                                box['y'] + box['height']
                            ])
                    
                    character_bboxes = []
                    reference_embeddings = []
                    
                    if panel_data.get('characterBoxes'):
                        for box in panel_data.get('characterBoxes'):
                            character_bboxes.append([
                                box['x'], 
                                box['y'], 
                                box['x'] + box['width'], 
                                box['y'] + box['height']
                            ])
                            
                            # Get the character embedding if available
                            character_name = box['character']
                            embedding = manga_generator.get_character_embedding(character_name)
                            if embedding is not None:
                                reference_embeddings.append(embedding)
                            else:
                                print(f"Warning: No embedding found for character {character_name}")
                                reference_embeddings.append(None)
                    
                    # Create IP parameters
                    ip_params = {
                        'text_bboxes': text_bboxes,
                        'character_bboxes': character_bboxes,
                        'reference_embeddings': reference_embeddings
                    }
                    
                    # Generate the panel
                    output_path, updated_panel_data = manga_generator.generate_panel(
                        panel_data=panel_data.get('panel_data'),
                        panel_index=panel_index,
                        seed=seed,
                        width=width,
                        height=height,
                        project_id=project_id,
                        ip_params=ip_params
                    )
                    
                    # Save panel metadata (using custom JSON encoder for NumPy arrays)
                    panel_json_path = project_panels_dir / f"panel_{panel_index:04d}.json"
                    with open(panel_json_path, 'w') as f:
                        json.dump(panel_data.get('panel_data'), f, indent=2, cls=NumpyEncoder)
                    
                    # Convert image to base64 for sending to frontend
                    with open(output_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    # Save result to response map
                    panel_response_map[request_id] = {
                        'status': 'success',
                        'imageData': f'data:image/png;base64,{img_data}',
                        'imagePath': f'/api/images/{project_id}/panels/panel_{panel_index:04d}.png',
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

@app.post("/api/generate_panel", response_model=Dict[str, Any])
async def generate_panel(request: PanelGenerateRequest):
    """API endpoint to generate a panel image or queue it for generation"""
    try:
        # Get project ID or use 'default' if not provided
        project_id = request.projectId
        
        # Log the incoming data for debugging
        print(f"Received data for panel generation, project ID: {project_id}")
        
        # Create panel_data for the manga generator in the correct format
        # The model expects text_bboxes and character_bboxes as lists of [x1, y1, x2, y2]
        text_bboxes = []
        if request.textBoxes:
            for box in request.textBoxes:
                # Convert from {x, y, width, height} to [x1, y1, x2, y2]
                text_bboxes.append([
                    box.x, 
                    box.y, 
                    box.x + box.width, 
                    box.y + box.height
                ])
                
        character_bboxes = []
        reference_embeddings = []
        
        if request.characterBoxes:
            for box in request.characterBoxes:
                # Convert from {x, y, width, height} to [x1, y1, x2, y2]
                character_bboxes.append([
                    box.x, 
                    box.y, 
                    box.x + box.width, 
                    box.y + box.height
                ])
                
                # Get the character embedding if available
                character_name = box.character
                if manga_generator:
                    embedding = manga_generator.get_character_embedding(character_name)
                    if embedding is not None:
                        reference_embeddings.append(embedding)
                    else:
                        print(f"Warning: No embedding found for character {character_name}")
                        # Add a placeholder embedding (None) so the indices match
                        reference_embeddings.append(None)
        
        # Create a panel data object for the MangaGenerator
        panel_data = {
            'setting': request.setting,
            'characters': set(request.characterNames),
            'elements': [],
            'scene_index': 0,
            'textBoxes': [box.dict() for box in request.textBoxes] if request.textBoxes else [],
            'characterBoxes': [box.dict() for box in request.characterBoxes] if request.characterBoxes else []
        }
        
        # Add dialogue elements
        for dialogue in request.dialogues:
            if dialogue.character and dialogue.text:
                dialogue_element = {
                    'type': 'dialogue',
                    'character': dialogue.character,
                    'dialogue': dialogue.text,
                    'characters_present': [dialogue.character]
                }
                panel_data['elements'].append(dialogue_element)
        
        # Add action elements
        for action in request.actions:
            if action.text:
                action_element = {
                    'type': 'action',
                    'text': action.text,
                    'characters_present': request.characterNames
                }
                panel_data['elements'].append(action_element)
        
        # If no elements were added but we have a prompt, create a generic action
        if not panel_data['elements'] and request.prompt:
            panel_data['elements'].append({
                'type': 'action',
                'text': request.prompt,
                'characters_present': request.characterNames
            })
            
        # Process the panel generation
        if initialization_status["is_initialized"]:
            # Extract panel details
            panel_index = request.panelIndex
            seed = request.seed if request.seed is not None else random.randint(0, 1000000)
            
            # Extract panel dimensions from the request
            panel_width = request.width
            panel_height = request.height
            
            # Create project directories if they don't exist
            project_output_dir = Path("manga_projects") / project_id
            project_panels_dir = project_output_dir / "panels"
            project_panels_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate the panel using MangaGenerator
            if not manga_generator:
                raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Manga generator not initialized'})
            
            # Create a JSON-serializable version of panel_data for saving to file
            panel_data_json_safe = panel_data.copy()
            
            # Create parameters to pass to transformer
            ip_params = {
                'text_bboxes': text_bboxes,
                'character_bboxes': character_bboxes,
                'reference_embeddings': reference_embeddings
            }
            
            # Generate panel and save to project directory
            output_path, updated_panel_data = manga_generator.generate_panel(
                panel_data=panel_data,
                panel_index=panel_index,
                seed=seed,
                width=panel_width,
                height=panel_height,
                project_id=project_id,
                ip_params=ip_params  # Pass the IP adapter parameters separately
            )
            
            # Save panel metadata (using custom JSON encoder for NumPy arrays)
            panel_json_path = project_panels_dir / f"panel_{panel_index:04d}.json"
            with open(panel_json_path, 'w') as f:
                json.dump(panel_data_json_safe, f, indent=2, cls=NumpyEncoder)
            
            # Return response with panel data and project ID
            with open(output_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            return {
                'status': 'success',
                'imageData': f'data:image/png;base64,{img_data}',
                'imagePath': f'/api/images/{project_id}/panels/panel_{panel_index:04d}.png',
                'prompt': request.prompt or manga_generator.create_panel_prompt(panel_data),
                'panelIndex': panel_index,
                'seed': seed,
                'width': panel_width,
                'height': panel_height,
                'projectId': project_id
            }
        else:
            # Queue the request if models not initialized
            # Generate a unique request ID
            request_id = f"req_{time.time()}_{random.randint(1000, 9999)}"
            
            # For queued requests, we need to ensure all data is JSON serializable
            # The actual embeddings will be retrieved when the request is processed
            character_names = [box.character for box in request.characterBoxes] if request.characterBoxes else []
            
            panel_request = {
                'request_id': request_id,
                'panel_data': {
                    'projectId': project_id,
                    'prompt': request.prompt,
                    'setting': request.setting,
                    'characterNames': request.characterNames,
                    'dialogues': [d.dict() for d in request.dialogues],
                    'actions': [a.dict() for a in request.actions],
                    'textBoxes': [t.dict() for t in request.textBoxes] if request.textBoxes else [],
                    'characterBoxes': [c.dict() for c in request.characterBoxes] if request.characterBoxes else [],
                    'panelIndex': request.panelIndex,
                    'seed': seed,
                    'width': panel_width,
                    'height': panel_height,
                    'panel_data': panel_data,
                }
            }
            
            # Add to queue
            panel_request_queue.put(panel_request)
            
            # Return status - the request is queued
            initialization_progress = initialization_status["progress"]
            return {
                'status': 'queued',
                'message': 'Panel generation queued. Models still initializing.',
                'request_id': request_id,
                'progress': initialization_progress,
                'projectId': project_id,
                'eta_seconds': max(5, int((100 - initialization_progress) * 0.5))
            }
            
    except Exception as e:
        print(f"Error handling generate_panel request: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})
    
@app.get("/api/check_panel_status", response_model=Dict[str, Any])
async def check_panel_status(request_id: str = Query(...)):
    """Check if a queued panel generation is complete"""
    if not request_id:
        raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'No request ID provided'})
    
    # Check if we have a response for this request
    if request_id in panel_response_map:
        # Get response and remove from map
        response = panel_response_map.pop(request_id)
        return response
    
    # Still processing
    return {
        'status': 'processing',
        'message': 'Panel is still being generated',
        'progress': initialization_status["progress"]
    }

@app.post("/api/create_page", response_model=Dict[str, Any])
async def create_page(request: CreatePageRequest):
    """Create a manga page from multiple panels"""
    global manga_generator
    
    if manga_generator is None:
        raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Models not initialized'})
    
    panel_indices = request.panelIndices
    layout = request.layout
    page_index = request.pageIndex
    project_id = request.projectId
    
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
            raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'No valid panels found'})
        
        # Generate the page and save to project directory
        page_path = project_pages_dir / f"page_{page_index:03d}.png"
        # Call a modified version of _create_page_layout that saves to project directory
        # or modify the function to accept a custom output path
        page_path = manga_generator._create_page_layout(panel_paths, page_index, output_path=page_path)
        
        # Read the output page
        with open(page_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return {
            'status': 'success',
            'imageData': f"data:image/png;base64,{img_data}",
            'pageIndex': page_index,
            'projectId': project_id
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

@app.post("/api/parse_screenplay", response_model=Dict[str, Any])
async def parse_screenplay(request: ParseScreenplayRequest):
    """Parse a screenplay file and return scene information"""
    screenplay_path = request.screenplay_path
    character_data_path = request.character_data_path
    
    if not os.path.exists(screenplay_path):
        raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Screenplay file not found'})
    
    if not os.path.exists(character_data_path):
        raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Character data file not found'})
    
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
        
        return {
            'status': 'success',
            'scenes': serialized_scenes
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

@app.get("/api/get_generated_panels", response_model=Dict[str, Any])
async def get_generated_panels(projectId: str = Query(default="default")):
    """Get all generated panels for a specific project"""
    global manga_generator
    
    if manga_generator is None:
        raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Models not initialized'})
    
    try:
        panels = []
        project_panels_dir = Path("manga_projects") / projectId / "panels"
        
        if not project_panels_dir.exists():
            return {
                'status': 'success',
                'panels': []
            }
        
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
                'projectId': projectId
            })
            
        return {
            'status': 'success',
            'panels': panels
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

@app.get("/api/images/{project_id}/{filename}")
async def get_image(project_id: str, filename: str):
    """Retrieve an image by its path"""
    try:
        image_path = Path("manga_projects") / project_id / "panels" / filename
        
        if not image_path.exists():
            # Check if it's in pages folder
            image_path = Path("manga_projects") / project_id / "pages" / filename
            if not image_path.exists():
                raise HTTPException(status_code=404, detail={'status': 'error', 'message': 'Image not found'})
        
        return FileResponse(image_path.as_posix(), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})
    
@app.get("/api/status", response_model=Dict[str, Any])
async def check_status():
    """Check if models are initialized"""
    global initialization_status
    
    return {
        'status': 'success',
        'initialized': initialization_status["is_initialized"],
        'initializing': initialization_status["is_initializing"],
        'message': initialization_status["message"],
        'progress': initialization_status["progress"]
    }

# PROJECT MANAGEMENT ROUTES

@app.get("/api/projects", response_model=Dict[str, Any])
async def get_projects():
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
            
        return {
            'status': 'success',
            'projects': projects
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

@app.post("/api/projects", response_model=Dict[str, Any])
async def create_project(request: CreateProjectRequest):
    """Create a new manga project"""
    try:
        name = request.name
        
        if not name:
            raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Project name is required'})
            
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
            'pages': 1,  # Start with 1 page instead of 0
            'lastModified': datetime.datetime.now().isoformat()
        }
        
        projects.append(new_project)
        
        # Save projects metadata
        with open(projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
            
        # Create project directory
        project_dir = projects_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        # Initialize with default page
        default_pages = [{
            'id': 'page-1',
            'panels': []
        }]
        
        pages_file = project_dir / "pages.json"
        with open(pages_file, 'w') as f:
            json.dump(default_pages, f, indent=2)
            
        return {
            'status': 'success',
            'project': new_project
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

@app.get("/api/projects/{project_id}", response_model=Dict[str, Any])
async def get_project(project_id: str):
    """Get a specific project and its pages"""
    try:
        projects_dir = Path("manga_projects")
        projects_file = projects_dir / "projects.json"
        
        if not projects_file.exists():
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': 'No projects found'})
            
        with open(projects_file, 'r') as f:
            projects = json.load(f)
            
        project = next((p for p in projects if p['id'] == project_id), None)
        
        if not project:
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': f'Project {project_id} not found'})
            
        # Load project pages
        project_dir = projects_dir / project_id
        pages_file = project_dir / "pages.json"
        
        if pages_file.exists():
            with open(pages_file, 'r') as f:
                pages = json.load(f)
        else:
            # Initialize with empty page if no pages exist
            pages = [{
                'id': 'page-1',
                'panels': []
            }]
            
        return {
            'status': 'success',
            'project': project,
            'pages': pages
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

@app.put("/api/projects/{project_id}", response_model=Dict[str, Any])
async def update_project(project_id: str, request: UpdateProjectRequest):
    """Update a project's pages"""
    try:
        pages = request.pages
        
        projects_dir = Path("manga_projects")
        projects_file = projects_dir / "projects.json"
        
        if not projects_file.exists():
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': 'No projects found'})
            
        with open(projects_file, 'r') as f:
            projects = json.load(f)
            
        project_index = next((i for i, p in enumerate(projects) if p['id'] == project_id), None)
        
        if project_index is None:
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': f'Project {project_id} not found'})
            
        # Update project metadata
        projects[project_index]['pages'] = len(pages)
        projects[project_index]['lastModified'] = datetime.datetime.now().isoformat()
        
        # Save updated projects metadata
        with open(projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
            
        # Save project pages with panel data
        project_dir = projects_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        # Process pages to ensure panel data is preserved
        processed_pages = []
        for page in pages:
            processed_page = {
                'id': page.get('id'),
                'panels': []
            }
            
            # Process each panel
            for panel in page.get('panels', []):
                # Keep all panel data but ensure JSON serializable
                processed_panel = {
                    'id': panel.get('id'),
                    'x': panel.get('x'),
                    'y': panel.get('y'),
                    'width': panel.get('width'),
                    'height': panel.get('height'),
                    'imagePath': panel.get('imagePath'),
                    'prompt': panel.get('prompt'),
                    'setting': panel.get('setting'),
                    'seed': panel.get('seed'),
                    'panelIndex': panel.get('panelIndex'),
                    'characterNames': panel.get('characterNames', []),
                    'characterBoxes': panel.get('characterBoxes', []),
                    'textBoxes': panel.get('textBoxes', []),
                    'dialogues': panel.get('dialogues', []),
                    'actions': panel.get('actions', [])
                }
                processed_page['panels'].append(processed_panel)
            
            processed_pages.append(processed_page)
        
        # Save pages to file
        pages_file = project_dir / "pages.json"
        with open(pages_file, 'w') as f:
            json.dump(processed_pages, f, indent=2, cls=NumpyEncoder)
            
        return {
            'status': 'success',
            'project': projects[project_index],
            'message': f'Saved {len(pages)} pages with {sum(len(p["panels"]) for p in processed_pages)} panels'
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

@app.delete("/api/projects/{project_id}", response_model=Dict[str, Any])
async def delete_project(project_id: str):
    """Delete a project"""
    try:
        projects_dir = Path("manga_projects")
        projects_file = projects_dir / "projects.json"
        
        if not projects_file.exists():
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': 'No projects found'})
            
        with open(projects_file, 'r') as f:
            projects = json.load(f)
            
        project_index = next((i for i, p in enumerate(projects) if p['id'] == project_id), None)
        
        if project_index is None:
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': f'Project {project_id} not found'})
            
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
            
        return {
            'status': 'success',
            'message': f'Project {deleted_project["name"]} deleted successfully'
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})
    
@app.post("/api/export/pdf", response_model=Dict[str, Any])
async def export_pdf(request: ExportPdfRequest):
    """Generate a PDF from manga pages using standard manga dimensions (5" × 7.5")"""
    try:
        project_id = request.projectId
        project_name = request.projectName
        page_images = request.pages
        quality = request.quality
        
        if not page_images:
            raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'No page images provided'})
        
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
        return {
            'status': 'success',
            'downloadUrl': f'/api/downloads/{project_id}/{filename}'
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})
    
@app.get("/api/downloads/{project_id}/{filename}")
async def download_file(project_id: str, filename: str):
    """Download a generated file (PDF, etc.)"""
    try:
        file_path = Path("manga_projects") / project_id / "exports" / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': 'File not found'})
        
        media_type = "application/pdf" if filename.endswith('.pdf') else "application/octet-stream"
        return FileResponse(
            path=file_path.as_posix(),
            filename=filename,
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)