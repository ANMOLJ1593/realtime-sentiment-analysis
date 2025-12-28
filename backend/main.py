from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://realtime-sentiment-analysis-frontend.onrender.com"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class SentimentRequest(BaseModel):
    text: str

def get_sentiment(text: str):
    start_time = time.time()

    response = requests.post(
        HF_MODEL_URL,
        headers=HEADERS,
        json={"inputs": text},
        timeout=30,
    )

    if response.status_code != 200:
        try:
            details = response.json()
        except Exception:
            details = response.text

        return {
            "error": "Hugging Face inference failed",
            "status_code": response.status_code,
            "details": details,
        }

    data = response.json()

    # unwrap nested HF responses
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], list):
            data = data[0]

        if len(data) > 0:
            return {
                "label": data[0]["label"],
                "score": data[0]["score"],
                "time_taken": int((time.time() - start_time) * 1000),
            }

    return {"error": "Unexpected HF response", "raw": data}

@app.get("/")
def root():
    return {"message": "HF Router Sentiment API running"}

@app.post("/predict")
def predict(req: SentimentRequest):
    return get_sentiment(req.text)
