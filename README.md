# MangaUI

Full-stack UI for generating manga-style comics using the Drawatoon AI model locally.
Made this to make my life easier generating a manga out of a screenplay I wrote, maybe this will be in shape at some point to help other people.
Pages are generated at 5x7.5 inches, standard manga page size.
Drawatoon can only generate 512x512 pixels, though the generation does not have to be a perfect square. 
I've included an upscaler as part of the pipeline so we can generate panels at any size we like.

## Overview

This project consists of:
- **Frontend**: A Next.js application for editing manga panels and pages
- **Backend**: A Flask API that interfaces with the Drawatoon model to generate manga-style images (instructions on downloading drawatoon below)

## Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.8+
- GPU with CUDA support (recommended for faster generation but I got this working on my M3 Max)

## Setup Instructions

### 1. Clone the repository

```bash
git clone git@github.com:MrSaturn1/mangaUI.git
cd mangaUI
```

### 2. Download the Drawatoon model
This application requires the Drawatoon model from Hugging Face:

Download the model from [fumeisama/drawatoon-v1](https://huggingface.co/fumeisama/drawatoon-v1)
Place the entire model folder in the following location: backend/drawatoon-v1/

The folder structure should look like:

backend/
└── drawatoon-v1/
    ├── model_index.json
    ├── README.md
    ├── scheduler/
    ├── text_encoder/
    ├── tokenizer/
    ├── transformer/
    └── vae/

### 3. Set up the Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The backend server will start at http://localhost:8000.

### 4. Set up the Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend development server will start at http://localhost:3000. Paste that into your browser and you're good to go.
Sometimes there are CORS issues, for now I've found that opening a new private browsing instance and launching the frontend in there resolves it.

### Features

Character generation and management
Consistent characters across scenes
Panel editing and composition
Scene generation with text prompts
Page composition and export

### Character Generation Guidelines

#### Single Character Scenes
When you have just one character in a panel:

Simply adding the character to the character list should be sufficient
The character will appear with their proper style from the embedding
They'll be placed in a default center position (covering roughly the middle 60% of the panel)
This is convenient for quick panel creation when precise positioning isn't critical

#### Multiple Character Scenes
When you have multiple characters in a panel:

It's essential to draw character boxes for each character
This gives you precise control over where each character appears
Without boxes, multiple characters might blend together in the center or be positioned unpredictably
Character boxes also help with proper composition, especially for dialogue scenes where characters need to face each other

#### Character Generation Best Practices

For simple single-character shots: Just select the character from the list
For any scene with interaction: Draw character boxes
For scenes where character positioning is important (even with one character): Draw character boxes
For wide establishing shots with characters in specific places: Always draw character boxes

### Text Bubble Generation Guidelines

#### Size Considerations:

Text boxes should be sized proportionally to the amount of text they'll contain
Leave about 20-30% extra space around the text to account for speech bubble styling
Too small text boxes may result in illegible or compressed text
Too large text boxes might dominate the panel unnecessarily


#### Positioning Best Practices:

Position text boxes near the speaking character when possible
For dialogue, place boxes in the upper portion of the panel when possible (following manga reading convention)
Avoid placing text boxes that cover important visual elements or character faces
Consider the reading flow (typically right-to-left in manga) when positioning multiple text boxes


#### When to Use Text Boxes:

Always create text boxes for dialogue that should appear in speech bubbles
Use text boxes for narration or scene descriptions that should appear in the panel
Empty panels (with no text) don't need text boxes at all


#### Multiple Text Boxes:

For complex dialogue scenes, draw separate text boxes for each speaker
Position sequential dialogue in a logical reading order
Maintain consistent spacing between multiple text boxes


#### Technical Notes:

The model associates dialogue with text boxes in the order they're created
Each text box requires corresponding dialogue content
Text boxes without dialogue may result in empty or random text bubbles
The model may generate stylized speech bubbles based on the dialogue content (questions, exclamations, etc.)

### License

MIT License

### Acknowledgements

Drawatoon Model by fumeisama
