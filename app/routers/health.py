# app/routers/health.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict

router = APIRouter()


class ComponentHealth(BaseModel):
    ready: bool
    detail: str


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, ComponentHealth]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and inference stack are ready."""
    from app.services.yolo_service import YOLOService
    from app.services.transformer_service import TransformerService
    from app.config import settings

    components = {}

    try:
        YOLOService.get_model()
        components["yolo"] = ComponentHealth(
            ready=True,
            detail=str(settings.YOLO_MODEL_PATH),
        )
    except Exception as e:
        components["yolo"] = ComponentHealth(ready=False, detail=str(e))

    try:
        TransformerService.get_model()
        components["transformer"] = ComponentHealth(
            ready=True,
            detail=str(settings.TRANSFORMER_MODEL_PATH),
        )
    except Exception as e:
        components["transformer"] = ComponentHealth(ready=False, detail=str(e))

    try:
        TransformerService.get_scaler()
        components["scaler"] = ComponentHealth(
            ready=True,
            detail=str(settings.SCALER_PATH),
        )
    except Exception as e:
        components["scaler"] = ComponentHealth(ready=False, detail=str(e))

    overall_status = "healthy" if all(item.ready for item in components.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.MODEL_VERSION,
        components=components,
    )
