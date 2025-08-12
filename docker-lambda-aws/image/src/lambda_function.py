import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer

EMOTION_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval","disgust",
    "embarrassment","excitement","fear","gratitude","grief","joy","love",
    "nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"
]

tokenizer = Tokenizer.from_file("/opt/model/tokenizer/tokenizer.json")
session = ort.InferenceSession("roberta_goe.onnx", providers=['CPUExecutionProvider'], 
    sess_options=ort.SessionOptions())
session.set_providers(['CPUExecutionProvider'], [{'arena_extend_strategy': 'kSameAsRequested'}])

def get_top_emotions_onnx(text, top_k=5):
    enc = tokenizer.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    logits = session.run(["logits"], inputs)[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    top_idx = np.argsort(probs[0])[::-1][:top_k]
    return [{"emotion": EMOTION_LABELS[i], "confidence": float(probs[0][i])} for i in top_idx]

def lambda_handler(event, context):
    text = event["queryStringParameters"]["text"]
    if not text:
        return {
            "statusCode": 400,
            "body": {
                "error": "Text input is required.",
                "recieved": event
            }
        }
    try:
        top_emotions = get_top_emotions_onnx(text)
        return {
            "statusCode": 200,
            "body": {
                "emotions": top_emotions
            }
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"An error occurred: {str(e)}"
        }