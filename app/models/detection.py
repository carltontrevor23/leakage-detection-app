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


class SensorAnomalyResponse(BaseModel):
    """Response for sensor anomaly prediction"""
    reconstruction_error: float
    threshold: float
    is_anomaly: bool
    risk_level: str
    status: str
    message: str


class MultimodalFusion(BaseModel):
    """Combined output from image and sensor evidence"""
    combined_risk_score: float = Field(..., ge=0.0, le=1.0)
    overall_status: str
    decision: str
    fusion_method: str
    image_evidence_present: bool
    sensor_anomaly_present: bool


class MultimodalPredictionResponse(BaseModel):
    """Response for multimodal inspection endpoint"""
    inspection_id: str
    image_detection_count: int
    image_result_image_url: str
    image_detections: List[Detection]
    sensor_analysis: SensorAnomalyResponse
    fusion: MultimodalFusion
    created_at: datetime
