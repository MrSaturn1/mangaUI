# MangaUI

Full-stack UI for generating manga-style comics using the Drawatoon AI model locally.
Made this to make my life easier generating a manga out of a screenplay I wrote, maybe this will be in shape at some point to help other people.
Pages are generated at 5x7.5 inches, standard manga page size.

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

The backend server will start at http://localhost:5000.

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

### License

MIT License

### Acknowledgements

Drawatoon Model by fumeisama
