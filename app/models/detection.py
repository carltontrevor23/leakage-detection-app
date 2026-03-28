# app/models/detection.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    """Single detection result"""
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox


class PredictionResponse(BaseModel):
    """Response for prediction endpoint"""
    inspection_id: str
    detection_count: int
    detections: List[Detection]
    result_image_url: str
    created_at: datetime


class ErrorResponse(BaseModel):
    """Error response format"""
    error: str
    detail: Optional[str] = None