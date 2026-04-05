# main.py
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import logging

from app.config import settings
from app.routers import detection, health
from app.routers import sensor, unified


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
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(settings.MEDIA_ROOT)), name="media")

app.include_router(health.router, tags=["Health"])
app.include_router(detection.router, prefix="/api/v1", tags=["Detection"])
app.include_router(sensor.router, prefix="/api/v1", tags=["Sensor"])
app.include_router(unified.router, prefix="/api/v1", tags=["Unified"])


@app.on_event("startup")
async def startup_event():
    """Initialize app state and optionally warm-load models."""
    from app.services.yolo_service import YOLOService
    from app.services.transformer_service import TransformerService

    app.state.yolo_model = None
    app.state.transformer_model = None
    app.state.scaler = None
    app.state.anomaly_threshold = settings.ANOMALY_THRESHOLD
    app.state.sequence_length = settings.SEQUENCE_LENGTH
    app.state.num_features = settings.NUM_FEATURES

    if not settings.PRELOAD_MODELS:
        print("[Startup] Skipping model preload; models will be loaded on first inference request.")
        return

    try:
        app.state.yolo_model = YOLOService.get_model()
        print(f"[Startup] YOLOv8 model loaded successfully from {settings.YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"[Startup] Warning: Could not load YOLO model: {e}")

    try:
        app.state.transformer_model = TransformerService.get_model()
    except Exception as e:
        print(f"[Startup] Warning: Could not load transformer model: {e}")

    try:
        app.state.scaler = TransformerService.get_scaler()
    except Exception as e:
        print(f"[Startup] Warning: Could not load scaler: {e}")


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


@app.get("/dashboard")
async def dashboard_page(request: Request):
    """Unified risk dashboard page"""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"title": "Unified Risk Dashboard"},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception during request")
    detail = str(exc) if settings.DEBUG else "An unexpected internal error occurred."
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": detail,
        },
    )
