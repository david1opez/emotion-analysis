from fastapi import FastAPI
import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

LABEL_COLUMNS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval","disgust",
    "embarrassment","excitement","fear","gratitude","grief","joy","love",
    "nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"
]

class TextInput(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    emotion: str
    probability: float

class GoEmotionsPredictor:
    def __init__(self, model_path: str = None):
        """Initialize the emotion predictor with the trained model."""
        if model_path is None:
            # Use absolute path to the model directory
            model_path = os.path.join(os.path.dirname(__file__), "roberta-goemotions")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the trained model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def get_top_emotions(self, text: str, top_k: int = 5) -> List[Dict[str, float]]:
        """Get the top K emotions for a given text."""
        # Tokenize the input
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=64,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
        # Create results dictionary
        predictions = {}
        for i, emotion in enumerate(LABEL_COLUMNS):
            predictions[emotion] = float(probabilities[i])
        
        # Sort emotions by probability and return top K
        sorted_emotions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return [{"emotion": emotion, "probability": prob} for emotion, prob in sorted_emotions[:top_k]]

app = FastAPI(title="GoEmotions API", description="Emotion prediction API using RoBERTa")

predictor = GoEmotionsPredictor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=List[EmotionResponse])
async def predict_emotions(input_data: TextInput):
    """
    Predict the top 5 emotions for the given text.
    """
    top_emotions = predictor.get_top_emotions(input_data.text, top_k=5)
    return top_emotions

@app.get("/")
async def root():
    return {"message": "GoEmotions API is running"}