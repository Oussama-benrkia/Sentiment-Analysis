"""
FastAPI web app.

Start:
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
from predict import load_model, predict

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"], _state["vocab"] = load_model()
    print(f"Model loaded on {config.DEVICE}. Ready.")
    yield
    _state.clear()


app = FastAPI(title="Sentiment Analysis", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment:  str
    emoji:      str
    confidence: float


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(body: PredictRequest):
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")
    label, emoji, confidence = predict(body.text, _state["model"], _state["vocab"])
    return PredictResponse(sentiment=label, emoji=emoji,
                           confidence=round(confidence * 100, 1))


@app.get("/health")
async def health():
    return {"status": "ok", "device": config.DEVICE, "num_classes": config.NUM_CLASSES}
