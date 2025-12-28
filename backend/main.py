from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import time

# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://realtime-sentiment-analysis-frontend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Hugging Face Inference Router (NEW)
# -----------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN environment variable not set")

HF_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
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
    start_time = time.time()

    response = requests.post(
        HF_MODEL_URL,
        headers=HEADERS,
        json={"inputs": text},
        timeout=30
    )

    if response.status_code != 200:
        return {
            "error": "Hugging Face inference failed",
            "status_code": response.status_code,
            "details": response.text
        }

    data = response.json()

    # Expected HF response: list of predictions
    if isinstance(data, list) and len(data) > 0:
        return {
            "label": data[0]["label"],
            "score": data[0]["score"],
            "time_taken": int((time.time() - start_time) * 1000)
        }

    return {
        "error": "Unexpected response format",
        "raw": data
    }

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API (HF Router)"}

@app.post("/predict")
def sentiment_analysis(request: SentimentRequest):
    return get_sentiment(request.text)
