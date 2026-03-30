# main.py
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse
import logging
import os
import joblib
import tensorflow as tf
import keras
from keras.models import load_model
from keras import layers

from app.config import settings
from app.routers import detection, health
from app.routers import sensor


@keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length,
            output_dim=d_model
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "d_model": self.d_model,
            }
        )
        return config


TRANSFORMER_MODEL_PATH = os.path.join(
    "ml_models", "transformer", "train3_transformer_autoencoder.keras"
)
SCALER_PATH = os.path.join(
    "ml_models", "transformer", "train3_transformer_scaler.pkl"
)

ANOMALY_THRESHOLD = 0.0011916750946368263
SEQUENCE_LENGTH = 20
NUM_FEATURES = 56


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(settings.MEDIA_ROOT)), name="media")

app.include_router(health.router, tags=["Health"])
app.include_router(detection.router, prefix="/api/v1", tags=["Detection"])
app.include_router(sensor.router, prefix="/api/v1", tags=["Sensor"])


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    from app.services.yolo_service import YOLOService

    try:
        YOLOService.get_model()
        print(f"[Startup] YOLOv8 model loaded successfully from {settings.YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"[Startup] Warning: Could not load YOLO model: {e}")

    try:
        app.state.transformer_model = load_model(
            TRANSFORMER_MODEL_PATH,
            custom_objects={"PositionalEmbedding": PositionalEmbedding},
            compile=False,
        )
        print(f"[Startup] Transformer model loaded successfully from {TRANSFORMER_MODEL_PATH}")
    except Exception as e:
        app.state.transformer_model = None
        print(f"[Startup] Warning: Could not load transformer model: {e}")

    try:
        app.state.scaler = joblib.load(SCALER_PATH)
        print(f"[Startup] Scaler loaded successfully from {SCALER_PATH}")
    except Exception as e:
        app.state.scaler = None
        print(f"[Startup] Warning: Could not load scaler: {e}")

    app.state.anomaly_threshold = ANOMALY_THRESHOLD
    app.state.sequence_length = SEQUENCE_LENGTH
    app.state.num_features = NUM_FEATURES


@app.get("/")
async def root(request: Request):
    """Landing page for UI"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": settings.APP_NAME},
    )


@app.get("/upload")
async def upload_page(request: Request):
    """Upload page for pipeline leak detection"""
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={},
    )
@app.get("/sensor")
async def sensor_page(request: Request):
    """Sensor analysis page"""
    return templates.TemplateResponse(
        request=request,
        name="sensor.html",
        context={"title": "Sensor Analysis"},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception during request")
    return PlainTextResponse(str(exc), status_code=500)