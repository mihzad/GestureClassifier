import io
import os

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.params import Body
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import numpy as np

from inference import Inferencer



@asynccontextmanager
async def lifespan(app: FastAPI):

    app.ml_inferencer = Inferencer()
    yield
    del app.ml_inferencer


app = FastAPI(lifespan=lifespan)

origins = os.getenv("GATEWAY_ALLOWED_ORIGINS").split(",,,")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionsItem(BaseModel):
    predictions: list[str]


@app.post('/predict', response_model=PredictionsItem)
async def recognize_letter(vid: UploadFile=File(
    title="video.npy for prediction",
    description="""
    A numpy array file (.npy) of (B,C,T,H,W) RGB-video (C=3) tensor with time dim T = 16 frames.
    Batch size(B), Height(H), Width(W) can be arbitrary. It is allowed to omit (B) dim.\n
    Values - float32 scaled to [0,1].
    Returns list of predicted letters with length=(B), one per video in batch.
    """,
)):

    if not vid.filename.endswith(".npy"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file extension. Only .npy files are accepted."
        )
    
    video_bytes = await vid.read()

    try:
        video_np = np.load(io.BytesIO(video_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"The npy file is corrupted. Error: {str(e)}"
        )
    if len(video_np.shape) < 4 or len(video_np.shape) > 5:
        raise HTTPException(
            status_code=422,
            detail=f"number of np array dimentions should be either 4 or 5, got: {len(video_np.shape)}"
        )
    elif len(video_np.shape) == 4:
        video_np = video_np[np.newaxis, :]

    try:
        predictions = await app.ml_inferencer.run_model(video_np)
    except RuntimeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Error processing the tensor. Error: {str(e)}"
            )

    return PredictionsItem(predictions = predictions)