import torch
from fastapi import Depends, HTTPException
import os
import sys

# Store loaded resources at module level
_model = None
_vocab = None
_preprocessor = None

# Path constants
MODEL_PATH = "app/core/models/artifacts/textCNN_best_model.pt"
VOCAB_PATH = "app/core/models/vocab/vocab_textCNN.pth"

def initialize_resources():
    """Initialize all resources on application startup"""
    global _model, _vocab, _preprocessor
    
    from app.api.src.model.textcnn import load_model
    from app.api.src.preprocessing.preprocessing import TextPreprocessor
    
    try:
        try:
            # Try with weights_only=False first
            _vocab = torch.load(VOCAB_PATH, weights_only=False, map_location='cpu')
        except Exception as e:
            print(f"First vocab loading attempt failed: {e}")
            
            # Try pickle_module approach
            import pickle
            _vocab = torch.load(VOCAB_PATH, map_location='cpu', pickle_module=pickle)
        
        print(f"Vocabulary loaded successfully with {len(_vocab)} entries")
        _model = load_model(MODEL_PATH, vocab_size=len(_vocab), embed_dim=100, numfilters=100, numclasses=2)
        _preprocessor = TextPreprocessor()
        print("Model and vocab loaded successfully")
    except Exception as e:
        print(f"Error loading model or vocab: {e}")
        raise e

def get_model():
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return _model

def get_vocab():
    if _vocab is None:
        raise HTTPException(status_code=500, detail="Vocabulary not loaded")
    return _vocab

def get_preprocessor():
    if _preprocessor is None:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded")
    return _preprocessor