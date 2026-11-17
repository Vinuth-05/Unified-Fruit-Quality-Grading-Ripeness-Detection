# Unified Fruit Quality Grading and Ripeness Detection

## Model
Vision Transformer (ViT)

## Fruit Types
- Apple
- Orange
- Pear

## Ripeness Classes
- Unripe
- Ripe
- Overripe

## Total Classes
9

## Dataset
The dataset contains training, validation and testing images
organized into nine fruit-ripeness classes.

## Trained Model
The trained Vision Transformer model is stored in:

Best_ViT_Model/

Files:
- config.json
- model.safetensors
- preprocessor_config.json

## Backend
FastAPI is used as the local backend interface.

Location:
Review01_App/backend/

Main files:
- main.py
- inference.py

## Frontend
React + Vite

Location:
Review01_App/frontend/

## Prediction Workflow

Image Upload
      ↓
React Frontend
      ↓
FastAPI /predict
      ↓
Image Preprocessing
      ↓
Trained ViT Model
      ↓
Prediction
      ↓
Fruit Type + Ripeness + Confidence
      ↓
React Frontend

## Running the Application

### Backend

cd Review01_App/backend

python -m uvicorn main:app --reload

Backend:
http://127.0.0.1:8000

### Frontend

cd Review01_App/frontend

npm.cmd run dev

Frontend:
http://localhost:5173

## Important
FastAPI is used only as the local communication layer
between the React frontend and our trained machine-learning model.

The trained ViT model is stored and executed locally.
No external AI prediction API is used.