# app/routers/health.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str = ""


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are ready"""
    from app.services.yolo_service import YOLOService
    from app.config import settings
    
    try:
        model = YOLOService.get_model()
        model_loaded = model is not None
        model_path = str(settings.YOLO_MODEL_PATH)
    except Exception as e:
        model_loaded = False
        model_path = str(e)
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=model_path
    )