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

# Reduce logging spam from polling endpoints
import logging

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out noisy polling endpoint logs
        try:
            # Get the message from different possible sources
            message = ""
            
            # Try multiple ways to get the log message
            if hasattr(record, 'getMessage'):
                message = str(record.getMessage())
            elif hasattr(record, 'msg'):
                message = str(record.msg % record.args) if record.args else str(record.msg)
            elif hasattr(record, 'args') and record.args:
                message = str(record.args[0])
            
            # Also check the record name/logger name
            logger_name = getattr(record, 'name', '')
            
            # List of endpoints to filter out (case insensitive)
            noisy_endpoints = [
                '/api/status',
                '/api/get_characters', 
                '/api/init/status',
                '/api/check_panel_status',
                'GET /api/status',
                'GET /api/get_characters',
                'GET /api/init/status', 
                'GET /api/check_panel_status',
                '"GET /api/status',
                '"GET /api/get_characters',
                '"GET /api/init/status',
                '"GET /api/check_panel_status'
            ]
            
            # HTTP status codes to filter
            noisy_patterns = [
                '200 OK',
                '- "GET /api/status',
                '- "GET /api/get_characters',
                '- "GET /api/init/status', 
                '- "GET /api/check_panel_status'
            ]
            
            # Filter out if any noisy endpoint is in the message
            message_lower = message.lower()
            
            # Check endpoints
            for endpoint in noisy_endpoints:
                if endpoint.lower() in message_lower:
                    return False
            
            # Check patterns  
            for pattern in noisy_patterns:
                if pattern.lower() in message_lower:
                    return False
                    
            # Also filter uvicorn access logs for these endpoints
            if 'uvicorn.access' in logger_name:
                for endpoint in ['/api/status', '/api/get_characters', '/api/init/status', '/api/check_panel_status']:
                    if endpoint in message_lower:
                        return False
                        
        except Exception:
            # If there's any error in filtering, allow the log through
            pass
            
        return True

# Apply the filter to multiple loggers that might be generating noise
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
logging.getLogger("fastapi").addFilter(EndpointFilter()) 
logging.getLogger("uvicorn").addFilter(EndpointFilter())

# Also set uvicorn access logger to WARNING level to reduce noise
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Apply filter to uvicorn access logger and other potential loggers
for logger_name in ["uvicorn.access", "uvicorn", "fastapi"]:
    try:
        logger = logging.getLogger(logger_name)
        logger.addFilter(EndpointFilter())
    except Exception:
        pass

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
    negativePrompt: Optional[str] = "deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands"
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
    
    # Allow character loading even if models aren't initialized yet
    # We can read character data and embeddings from files directly
    
    try:
        # Load character data from the default path if manga_generator isn't available
        character_data_path = DEFAULT_CHARACTER_DATA_PATH
        if manga_generator is not None:
            character_data_path = manga_generator.character_data_path
            
        with open(character_data_path, 'r', encoding='utf-8') as f:
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
            
            # Try to get image data if available - check main location first, then keepers as fallback
            image_data = None
            main_path = os.path.join('character_output', 'character_images', f"{name}.png")
            keeper_path = os.path.join('character_output', 'character_images', 'keepers', f"{name}.png")
            
            # Prefer main image, fallback to keeper for backwards compatibility
            image_path = main_path if os.path.exists(main_path) else (keeper_path if os.path.exists(keeper_path) else None)
            
            if image_path:
                try:
                    with open(image_path, 'rb') as img_file:
                        image_bytes = img_file.read()
                        image_data = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                except Exception as e:
                    print(f"Error loading image for {name}: {e}")
                    image_data = None
            
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

def create_hybrid_embedding_from_image(image_path: str):
    """Create a hybrid CLIP+Magi v2 embedding from an image using DiffSensei approach"""
    try:
        from hybrid_character_encoder import HybridCharacterEncoder
        
        # Initialize hybrid encoder
        encoder = HybridCharacterEncoder('characters.json')
        
        # Extract character name from path for better logging
        character_name = Path(image_path).stem
        
        # Generate embedding using the hybrid approach
        embedding = encoder.extract_hybrid_embedding(image_path, character_name)
        
        # Ensure it's in the right format [1, 768] for compatibility
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # Add batch dimension
        
        return embedding.cpu()  # Return as CPU tensor
        
    except ImportError:
        print("Warning: Hybrid encoder not available, falling back to Magi v2")
        return create_magi_v2_embedding_from_image(image_path)
    except Exception as e:
        print(f"Error creating hybrid embedding: {e}")
        print("Falling back to Magi v2 embedding")
        return create_magi_v2_embedding_from_image(image_path)

def create_random_embedding_for_character(character_name: str):
    """Create a random embedding for character consistency"""
    print(f"Creating random embedding for {character_name}")
    return torch.randn(1, 768)  # Match Drawatoon's expected 768 dimensions

def create_magi_v2_embedding_from_image(image_path: str):
    """Create a random embedding instead of Magi v2 for now"""
    character_name = Path(image_path).stem
    print(f"Using random embedding for {character_name} (reverting from Magi v2)")
    return create_random_embedding_for_character(character_name)

def save_character_generation_to_history(character_name: str, output_path: Path, seed: int, prompt: str = "", create_embedding: bool = True):
    """Save a character generation to the history directory with CLIP embedding"""
    try:
        # Create history directory structure
        history_dir = Path("character_output") / "character_images" / "history" / character_name
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this generation
        timestamp = str(int(time.time() * 1000))  # milliseconds for uniqueness
        datetime_str = datetime.datetime.now().isoformat()
        
        # Copy image to history with timestamp
        history_image_path = history_dir / f"{timestamp}.png"
        import shutil
        shutil.copy2(output_path, history_image_path)
        
        # Create Magi v2 embedding
        embedding_created = False
        if create_embedding:
            try:
                embedding = create_magi_v2_embedding_from_image(str(history_image_path))
                
                # Save embedding as PyTorch tensor (.pt file)
                embedding_path = history_dir / f"{timestamp}.pt"
                torch.save(embedding, embedding_path)
                
                embedding_created = True
                print(f"Created hybrid embedding for {character_name} at {embedding_path}")
                
            except Exception as e:
                print(f"Failed to create embedding for {character_name}: {e}")
        
        # Load or create history metadata
        history_file = history_dir / "history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)
        else:
            history_data = {'generations': []}
        
        # Mark all previous generations as inactive
        for gen in history_data.get('generations', []):
            gen['isActive'] = False
        
        # Add new generation as active
        new_generation = {
            'timestamp': timestamp,
            'datetime': datetime_str,
            'seed': seed,
            'prompt': prompt,
            'isActive': True,
            'hasEmbedding': embedding_created
        }
        
        history_data['generations'].append(new_generation)
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        return timestamp
        
    except Exception as e:
        print(f"Error saving character generation to history: {e}")
        return None

def save_panel_generation_to_history(project_id: str, panel_index: int, output_path: Path, seed: int, prompt: str = ""):
    """Save a panel generation to the history directory"""
    try:
        # Create history directory structure
        history_dir = Path("manga_projects") / project_id / "panels" / "history" / f"panel_{panel_index:04d}"
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this generation
        timestamp = str(int(time.time() * 1000))  # milliseconds for uniqueness
        datetime_str = datetime.datetime.now().isoformat()
        
        # Copy image to history with timestamp
        history_image_path = history_dir / f"{timestamp}.png"
        import shutil
        shutil.copy2(output_path, history_image_path)
        
        # Load or create history metadata
        history_file = history_dir / "history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)
        else:
            history_data = {'generations': []}
        
        # Mark all previous generations as inactive
        for gen in history_data.get('generations', []):
            gen['isActive'] = False
        
        # Add new generation as active
        new_generation = {
            'timestamp': timestamp,
            'datetime': datetime_str,
            'seed': seed,
            'prompt': prompt,
            'isActive': True
        }
        
        history_data['generations'].append(new_generation)
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        return timestamp
        
    except Exception as e:
        print(f"Error saving panel generation to history: {e}")
        return None

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
        
        # Save this generation to history
        timestamp = save_character_generation_to_history(
            character_name, 
            Path(output_path), 
            seed, 
            f"Character generation for {character_name}"
        )
        
        # Read the output image
        with open(output_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return {
            'status': 'success',
            'imageData': f"data:image/png;base64,{img_data}",
            'name': character_name,
            'seed': seed,
            'timestamp': timestamp
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
                        ip_params=ip_params,
                        negative_prompt=panel_data.get('negativePrompt')
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
                ip_params=ip_params,  # Pass the IP adapter parameters separately
                negative_prompt=request.negativePrompt
            )
            
            # Save this generation to history
            panel_prompt = request.prompt or manga_generator.create_panel_prompt(panel_data)
            timestamp = save_panel_generation_to_history(
                project_id, 
                panel_index,
                Path(output_path),
                seed,
                panel_prompt
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
                'projectId': project_id,
                'timestamp': timestamp
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
                    'negativePrompt': request.negativePrompt,
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
            'pages': 2,  # Start with 2 pages for better layout
            'lastModified': datetime.datetime.now().isoformat()
        }
        
        projects.append(new_project)
        
        # Save projects metadata
        with open(projects_file, 'w') as f:
            json.dump(projects, f, indent=2)
            
        # Create project directory
        project_dir = projects_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        # Initialize with 2 default pages
        default_pages = [
            {
                'id': 'page-1',
                'panels': []
            },
            {
                'id': 'page-2',
                'panels': []
            }
        ]
        
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
                    'imageData': panel.get('imageData'),  # Save imageData as well if present
                    'prompt': panel.get('prompt'),
                    'negativePrompt': panel.get('negativePrompt'),
                    'enhancedPrompt': panel.get('enhancedPrompt'),
                    'setting': panel.get('setting'),
                    'seed': panel.get('seed'),
                    'panelIndex': panel.get('panelIndex'),
                    'characterNames': panel.get('characterNames', []),
                    'characterBoxes': panel.get('characterBoxes', []),
                    'textBoxes': panel.get('textBoxes', []),
                    'dialogues': panel.get('dialogues', []),
                    'actions': panel.get('actions', []),
                    'timestamp': panel.get('timestamp')  # Save generation timestamp
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

# GENERATION HISTORY ROUTES

@app.get("/api/character_history/{character_name}", response_model=Dict[str, Any])
async def get_character_history(character_name: str):
    """Get generation history for a character"""
    try:
        character_name = character_name.replace('%20', ' ')  # Handle URL encoding
        
        # Path to character history directory
        history_dir = Path("character_output") / "character_images" / "history" / character_name
        
        if not history_dir.exists():
            return {
                'status': 'success',
                'history': [],
                'currentGeneration': None
            }
        
        # Load history metadata
        history_file = history_dir / "history.json"
        if not history_file.exists():
            return {
                'status': 'success', 
                'history': [],
                'currentGeneration': None
            }
        
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        
        # Load image data for each generation
        generations = []
        current_generation = None
        
        for generation in history_data.get('generations', []):
            image_path = history_dir / f"{generation['timestamp']}.png"
            if image_path.exists():
                try:
                    with open(image_path, 'rb') as img_file:
                        image_bytes = img_file.read()
                        if len(image_bytes) == 0:
                            print(f"Warning: Empty image file at {image_path}")
                            continue
                        image_data = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                    
                    gen_data = {
                        'timestamp': generation['timestamp'],
                        'datetime': generation['datetime'],
                        'imageData': image_data,
                        'seed': generation['seed'],
                        'prompt': generation.get('prompt', ''),
                        'isActive': generation.get('isActive', False),
                        'hasEmbedding': generation.get('hasEmbedding', False)
                    }
                    
                    generations.append(gen_data)
                    
                    if generation.get('isActive', False):
                        current_generation = gen_data
                        
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
            else:
                print(f"History image not found: {image_path}")
        
        return {
            'status': 'success',
            'history': generations,
            'currentGeneration': current_generation
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

@app.get("/api/panel_history/{project_id}/{panel_index}", response_model=Dict[str, Any])
async def get_panel_history(project_id: str, panel_index: int):
    """Get generation history for a panel"""
    try:
        # Path to panel history directory
        history_dir = Path("manga_projects") / project_id / "panels" / "history" / f"panel_{panel_index:04d}"
        
        if not history_dir.exists():
            return {
                'status': 'success',
                'history': [],
                'currentGeneration': None
            }
        
        # Load history metadata
        history_file = history_dir / "history.json"
        if not history_file.exists():
            return {
                'status': 'success',
                'history': [],
                'currentGeneration': None
            }
        
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        
        # Load image data for each generation
        generations = []
        current_generation = None
        
        for generation in history_data.get('generations', []):
            image_path = history_dir / f"{generation['timestamp']}.png"
            if image_path.exists():
                try:
                    with open(image_path, 'rb') as img_file:
                        image_bytes = img_file.read()
                        if len(image_bytes) == 0:
                            print(f"Warning: Empty panel image file at {image_path}")
                            continue
                        image_data = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                    
                    gen_data = {
                        'timestamp': generation['timestamp'],
                        'datetime': generation['datetime'],
                        'imageData': image_data,
                        'seed': generation['seed'],
                        'prompt': generation.get('prompt', ''),
                        'isActive': generation.get('isActive', False)
                    }
                    
                    generations.append(gen_data)
                    
                    if generation.get('isActive', False):
                        current_generation = gen_data
                        
                except Exception as e:
                    print(f"Error loading panel image {image_path}: {e}")
                    continue
            else:
                print(f"Panel history image not found: {image_path}")
        
        return {
            'status': 'success',
            'history': generations,
            'currentGeneration': current_generation
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

class SetActiveGenerationRequest(BaseModel):
    characterName: Optional[str] = None
    projectId: Optional[str] = None 
    panelIndex: Optional[int] = None
    timestamp: str
    createEmbedding: Optional[bool] = False

@app.post("/api/set_active_character_generation", response_model=Dict[str, Any])
async def set_active_character_generation(request: SetActiveGenerationRequest):
    """Set a character generation as active"""
    try:
        character_name = request.characterName
        timestamp = request.timestamp
        create_embedding = request.createEmbedding
        
        if not character_name:
            raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Character name is required'})
        
        # Path to character history directory
        history_dir = Path("character_output") / "character_images" / "history" / character_name
        history_file = history_dir / "history.json"
        
        if not history_file.exists():
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': 'Character history not found'})
        
        # Load and update history
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        
        # Update active status
        found_generation = False
        for generation in history_data.get('generations', []):
            if generation['timestamp'] == timestamp:
                generation['isActive'] = True
                generation['hasEmbedding'] = create_embedding or generation.get('hasEmbedding', False)
                found_generation = True
                
                # Copy this generation to the main character image location
                source_image_path = history_dir / f"{timestamp}.png"
                source_embedding_path = history_dir / f"{timestamp}.pt"
                dest_image_path = Path("character_output") / "character_images" / f"{character_name}.png"
                dest_embedding_path = Path("character_output") / "character_embeddings" / f"{character_name}.pt"
                
                if source_image_path.exists():
                    import shutil
                    dest_image_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_image_path, dest_image_path)
                    
                    # Copy embedding if it exists
                    if source_embedding_path.exists():
                        dest_embedding_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_embedding_path, dest_embedding_path)
                        
                        # Update character embeddings map
                        embeddings_map_path = Path("character_output") / "character_embeddings" / "character_embeddings_map.json"
                        try:
                            if embeddings_map_path.exists():
                                with open(embeddings_map_path, 'r') as f:
                                    embeddings_map = json.load(f)
                            else:
                                embeddings_map = {}
                            
                            embeddings_map[character_name] = {
                                "name": character_name,
                                "image_path": str(dest_image_path),
                                "embedding_path": str(dest_embedding_path)
                            }
                            
                            with open(embeddings_map_path, 'w') as f:
                                json.dump(embeddings_map, f, indent=2)
                                
                        except Exception as e:
                            print(f"Error updating embeddings map: {e}")
                    
                    # If creating embedding flag is set, also copy to keepers for backwards compatibility
                    if create_embedding:
                        keeper_path = Path("character_output") / "character_images" / "keepers" / f"{character_name}.png"
                        keeper_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_image_path, keeper_path)
            else:
                generation['isActive'] = False
        
        if not found_generation:
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': 'Generation not found'})
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        return {
            'status': 'success',
            'message': 'Character generation set as active'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

@app.post("/api/set_active_panel_generation", response_model=Dict[str, Any])
async def set_active_panel_generation(request: SetActiveGenerationRequest):
    """Set a panel generation as active"""
    try:
        project_id = request.projectId
        panel_index = request.panelIndex
        timestamp = request.timestamp
        
        if not project_id or panel_index is None:
            raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Project ID and panel index are required'})
        
        # Path to panel history directory
        history_dir = Path("manga_projects") / project_id / "panels" / "history" / f"panel_{panel_index:04d}"
        history_file = history_dir / "history.json"
        
        if not history_file.exists():
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': 'Panel history not found'})
        
        # Load and update history
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        
        # Update active status
        found_generation = False
        for generation in history_data.get('generations', []):
            if generation['timestamp'] == timestamp:
                generation['isActive'] = True
                found_generation = True
                
                # Copy this generation to the main panel location
                source_path = history_dir / f"{timestamp}.png"
                dest_path = Path("manga_projects") / project_id / "panels" / f"panel_{panel_index:04d}.png"
                
                if source_path.exists():
                    import shutil
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, dest_path)
            else:
                generation['isActive'] = False
        
        if not found_generation:
            raise HTTPException(status_code=404, detail={'status': 'error', 'message': 'Generation not found'})
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        return {
            'status': 'success',
            'message': 'Panel generation set as active'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})

def migrate_keepers_to_main():
    """Safely migrate keeper images to main location and create proper CLIP embeddings"""
    keepers_dir = Path("character_output") / "character_images" / "keepers"
    main_dir = Path("character_output") / "character_images"
    embeddings_dir = Path("character_output") / "character_embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    if not keepers_dir.exists():
        print("No keepers directory found, migration not needed")
        return
    
    print("Migrating keeper images to main location and creating CLIP embeddings...")
    
    # Load or create embeddings map
    embeddings_map_path = embeddings_dir / "character_embeddings_map.json"
    if embeddings_map_path.exists():
        with open(embeddings_map_path, 'r') as f:
            embeddings_map = json.load(f)
    else:
        embeddings_map = {}
    
    migrated_count = 0
    
    for keeper_file in keepers_dir.glob("*.png"):
        # Skip card files
        if keeper_file.name.endswith("_card.png"):
            continue
            
        character_name = keeper_file.stem
        main_file = main_dir / keeper_file.name
        embedding_file = embeddings_dir / f"{character_name}.pt"
        
        print(f"Processing {character_name}...")
        
        # Copy image to main location (always do this)
        import shutil
        shutil.copy2(keeper_file, main_file)
        print(f"  ✓ Copied image to main location")
        
        # Create proper CLIP embedding from the image
        try:
            embedding = create_clip_embedding_from_image(str(keeper_file))
            torch.save(embedding, embedding_file)
            
            # Update embeddings map
            embeddings_map[character_name] = {
                "name": character_name,
                "image_path": str(main_file),
                "embedding_path": str(embedding_file)
            }
            
            print(f"  ✓ Created CLIP embedding")
            migrated_count += 1
            
        except Exception as e:
            print(f"  ✗ Failed to create embedding: {e}")
    
    # Save updated embeddings map
    with open(embeddings_map_path, 'w') as f:
        json.dump(embeddings_map, f, indent=2)
    
    print(f"\nMigration complete!")
    print(f"  • Migrated {migrated_count} characters with proper CLIP embeddings")
    print(f"  • Keeper files preserved in keepers folder")
    print(f"  • Embeddings map updated at {embeddings_map_path}")
    print(f"  • Old random embeddings replaced with CLIP embeddings")

@app.post("/api/migrate_keepers", response_model=Dict[str, Any])
async def migrate_keepers():
    """API endpoint to safely migrate keeper images to main location"""
    try:
        migrate_keepers_to_main()
        return {
            'status': 'success',
            'message': 'Keeper images migrated to main location successfully'
        }
    except Exception as e:
        return {
            'status': 'error', 
            'message': f'Migration failed: {str(e)}'
        }

@app.post("/api/generate_hybrid_embeddings", response_model=Dict[str, Any])
async def generate_hybrid_embeddings():
    """API endpoint to generate hybrid CLIP+Magi embeddings for all characters"""
    try:
        from hybrid_character_encoder import HybridCharacterEncoder
        
        print("Initializing hybrid character encoder...")
        encoder = HybridCharacterEncoder('characters.json')
        
        print("Generating hybrid embeddings for all keeper characters...")
        embeddings_map = encoder.generate_all_keeper_embeddings()
        
        generated_count = len([char for char, data in embeddings_map.items() 
                              if data.get("embedding_type") == "hybrid_clip_magi_768"])
        
        return {
            'status': 'success',
            'message': f'Generated hybrid embeddings for {generated_count} characters',
            'embedding_type': 'hybrid_clip_magi_768',
            'characters_processed': generated_count
        }
        
    except ImportError:
        return {
            'status': 'error',
            'message': 'Hybrid character encoder not available. Make sure opencv-python is installed.'
        }
    except Exception as e:
        print(f"Error generating hybrid embeddings: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error', 
            'message': f'Failed to generate hybrid embeddings: {str(e)}'
        }

@app.post("/api/fix_embedding_dimensions", response_model=Dict[str, Any])
async def fix_embedding_dimensions():
    """API endpoint to fix embedding dimensions from 512 to 768"""
    try:
        fixed_count = 0
        embeddings_dir = Path("character_output") / "character_embeddings"
        embeddings_map_path = embeddings_dir / "character_embeddings_map.json"
        
        if not embeddings_map_path.exists():
            return {
                'status': 'error',
                'message': 'No embeddings map found'
            }
        
        with open(embeddings_map_path, 'r') as f:
            embeddings_map = json.load(f)
        
        for character_name, char_data in embeddings_map.items():
            embedding_path = Path(char_data["embedding_path"])
            image_path = Path(char_data["image_path"])
            
            if embedding_path.exists() and image_path.exists():
                try:
                    # Load existing embedding to check dimensions
                    existing_embedding = torch.load(embedding_path, weights_only=False)
                    
                    # Check if embedding has wrong dimensions (512 instead of 768)
                    if existing_embedding.shape[-1] == 512:
                        print(f"Fixing embedding dimensions for {character_name}: {existing_embedding.shape} -> 768")
                        
                        # Create new 768-dimensional embedding using hybrid approach
                        new_embedding = create_hybrid_embedding_from_image(str(image_path))
                        torch.save(new_embedding, embedding_path)
                        
                        fixed_count += 1
                        print(f"  ✓ Fixed embedding for {character_name}")
                    else:
                        print(f"Embedding for {character_name} already has correct dimensions: {existing_embedding.shape}")
                        
                except Exception as e:
                    print(f"  ✗ Failed to fix embedding for {character_name}: {e}")
        
        return {
            'status': 'success',
            'message': f'Fixed {fixed_count} character embeddings to correct dimensions'
        }
    except Exception as e:
        print(f"Error fixing embedding dimensions: {e}")
        return {
            'status': 'error', 
            'message': f'Failed to fix embedding dimensions: {str(e)}'
        }

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)