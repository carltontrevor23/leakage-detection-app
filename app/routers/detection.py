# app/routers/detection.py
import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.config import settings
from app.models.detection import (
    PredictionResponse,
    Detection,
    BoundingBox,
    ErrorResponse,
    MultimodalPredictionResponse,
    SensorAnomalyResponse,
    MultimodalFusion,
)
from app.services.multimodal_service import MultimodalService
from app.services.yolo_service import YOLOService
from app.utils.file_handling import validate_image, save_upload_file

router = APIRouter()

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
    validate_image(
        file=image,
        allowed_types=settings.ALLOWED_IMAGE_TYPES,
        max_size_mb=settings.MAX_UPLOAD_SIZE_MB,
    )
    
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


@router.post(
    "/inspect",
    response_model=MultimodalPredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def inspect_pipeline(
    image: UploadFile = File(..., description="Pipeline image to analyze"),
    sensor_sequence: str = Form(..., description="Sensor sequence as JSON array with shape 20x56"),
    confidence_threshold: float = Form(0.25, ge=0.1, le=1.0, description="Minimum confidence for image detections")
):
    """
    Run a fused inspection using both image evidence and sensor sequence evidence.
    """
    validate_image(
        file=image,
        allowed_types=settings.ALLOWED_IMAGE_TYPES,
        max_size_mb=settings.MAX_UPLOAD_SIZE_MB,
    )

    inspection_id = uuid.uuid4().hex[:8]

    try:
        sequence = MultimodalService.parse_sequence(sensor_sequence)
        upload_path = save_upload_file(
            file=image,
            upload_dir=settings.UPLOAD_DIR,
            prefix=inspection_id,
        )

        prediction = MultimodalService.predict(
            image_path=str(upload_path),
            sensor_sequence=sequence,
            conf_threshold=confidence_threshold,
        )

        image_prediction = prediction["image_prediction"]
        sensor_prediction = prediction["sensor_prediction"]
        fusion_prediction = prediction["fusion"]

        detections = [
            Detection(
                label=d["label"],
                confidence=d["confidence"],
                bbox=BoundingBox(**d["bbox"]),
            )
            for d in image_prediction["detections"]
        ]

        response_model = MultimodalPredictionResponse(
            inspection_id=inspection_id,
            image_detection_count=image_prediction["detection_count"],
            image_result_image_url=f"/media/{image_prediction['result_image_path']}",
            image_detections=detections,
            sensor_analysis=SensorAnomalyResponse(**sensor_prediction),
            fusion=MultimodalFusion(**fusion_prediction),
            created_at=datetime.now(),
        )

        return response_model.dict()

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inspection failed: {str(e)}"
        )
