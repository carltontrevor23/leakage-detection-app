# app/routers/detection.py
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image

from app.config import settings
from app.models.detection import PredictionResponse, Detection, BoundingBox, ErrorResponse
from app.services.yolo_service import YOLOService
from app.utils.file_handling import validate_image, save_upload_file

router = APIRouter()


def validate_image_file(file: UploadFile) -> bool:
    """
    Validate uploaded file is an allowed image type.
    Returns True if valid, raises HTTPException if not.
    """
    # Check content type
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_IMAGE_TYPES)}"
        )
    
    # Check file size (read a chunk to avoid loading entire file)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB} MB"
        )
    
    return True


@router.post(
    "/detect",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def detect_leaks(
    image: UploadFile = File(..., description="Pipeline image to analyze"),
    confidence_threshold: float = Form(0.25, ge=0.1, le=1.0, description="Minimum confidence for detections")
):
    """
    Detect pipeline leaks in uploaded image using YOLOv8.
    
    Returns annotated image URL and detection details.
    """
    # Validate uploaded file
    validate_image_file(image)
    
    # Generate unique ID for this inspection
    inspection_id = uuid.uuid4().hex[:8]
    
    try:
        # Save uploaded file
        upload_path = save_upload_file(
            file=image,
            upload_dir=settings.UPLOAD_DIR,
            prefix=inspection_id
        )
        
        # Run YOLOv8 inference
        prediction = YOLOService.predict(
            image_path=str(upload_path),
            conf_threshold=confidence_threshold
        )
        
        # Build response URL
        result_image_url = f"/media/{prediction['result_image_path']}"
        
        # Convert predictions to proper Pydantic model objects
        detections = [
            Detection(
                label=d['label'],
                confidence=d['confidence'],
                bbox=BoundingBox(**d['bbox'])
            )
            for d in prediction['detections']
        ]

        response_model = PredictionResponse(
            inspection_id=inspection_id,
            detection_count=prediction['detection_count'],
            detections=detections,
            result_image_url=result_image_url,
            created_at=datetime.now()
        )

        # Return raw dict to avoid bad hashing when FastAPI serializes nested models
        return response_model.dict()

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )