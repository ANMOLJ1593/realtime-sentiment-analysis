# from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests

# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Hugging Face Inference Config
# -----------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN environment variable not set")

HF_MODEL_URL = (
    "https://api-inference.huggingface.co/models/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

# -----------------------------
# Request Schema
# -----------------------------
class SentimentRequest(BaseModel):
    text: str

# -----------------------------
# Inference Function
# -----------------------------
def get_sentiment(text: str):
    payload = {"inputs": text}

    response = requests.post(
        HF_MODEL_URL,
        headers=HEADERS,
        json=payload,
        timeout=30
    )

    response.raise_for_status()
    return response.json()

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API (HF Inference)"}

@app.post("/predict")
def sentiment_analysis(request: SentimentRequest):
    return get_sentiment(request.text)
