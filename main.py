# main.py
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse
import logging

from app.config import settings
from app.routers import detection, health

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Jinja templates
templates = Jinja2Templates(directory="templates")

# Mount frontend assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/media", StaticFiles(directory=str(settings.MEDIA_ROOT)), name="media")

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(detection.router, prefix="/api/v1", tags=["Detection"])


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    from app.services.yolo_service import YOLOService
    
    try:
        YOLOService.get_model()
        print(f"[Startup] YOLOv8 model loaded successfully from {settings.YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"[Startup] Warning: Could not load YOLO model: {e}")


@app.get("/", response_class=None)
async def root(request: Request):
    """Landing page for UI"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": settings.APP_NAME},
    )


@app.get("/upload", response_class=None)
async def upload_page(request: Request):
    """Upload page for pipeline leak detection"""
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception during request")
    return PlainTextResponse(str(exc), status_code=500)

