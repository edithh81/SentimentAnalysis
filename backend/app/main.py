from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.service import initialize_resources
import torch
import os
MODEL_PATH = "models/artifacts/model.pt"
VOCAB_PATH = "models/vocab/vocab.pth"
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
async def startup_event():
    initialize_resources()
    
@app.get("/")
async def root():
    return {"message": "TextCNN backend"}
app.include_router(router, prefix="/api")

    






    
