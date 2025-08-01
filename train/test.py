import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict
import time
import os

LABEL_COLUMNS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval","disgust",
    "embarrassment","excitement","fear","gratitude","grief","joy","love",
    "nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"
]

class GoEmotionsPredictor:
    def __init__(self, model_path: str = None):
        """Initialize the emotion predictor with the trained model."""
        if model_path is None:
            import os
            # Use absolute path to the model directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(os.path.dirname(current_dir), "api", "roberta-goemotions")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path}")
        print(f"Using device: {self.device}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def predict_single(self, text: str) -> Dict[str, float]:
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=64,
            return_tensors="pt"
        )
        
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
        results = {}
        for i, emotion in enumerate(LABEL_COLUMNS):
            results[emotion] = float(probabilities[i])
            
        return results
    
    def get_top_emotions(self, text: str, top_k: int = 3) -> List[tuple]:
        """Get the top K emotions for a given text."""
        predictions = self.predict_single(text)
        
        # Sort emotions by probability
        sorted_emotions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_emotions[:top_k]

def interactive_test():    
    try:
        predictor = GoEmotionsPredictor()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Enter text to analyze emotions (type 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        print(f"\nAnalyzing: \"{user_input}\"")
        start_time = time.time()
        
        # Get top emotions
        top_emotions = predictor.get_top_emotions(user_input, top_k=5)
        
        print("\nTop 5 emotions:")
        for emotion, prob in top_emotions:
            print(f"  {emotion}: {prob:.3f}")
        
        end_time = time.time()
        print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds.")

def main():
    interactive_test()

if __name__ == "__main__":
    main()
