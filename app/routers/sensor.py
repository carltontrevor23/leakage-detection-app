# app/routers/sensor.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from typing import List

from app.services.transformer_service import TransformerService

router = APIRouter()


class SensorDataRequest(BaseModel):
    """Request schema for transformer sensor data"""
    sequence: List[List[float]] = Field(
        ...,
        description="A 20x56 sequence of sensor values"
    )

    @model_validator(mode="after")
    def validate_sequence_shape(self):
        if len(self.sequence) != 20:
            raise ValueError("Input must contain exactly 20 time steps")

        for row in self.sequence:
            if len(row) != 56:
                raise ValueError("Each time step must contain exactly 56 features")

        return self


class SensorPredictionResponse(BaseModel):
    """Response schema for sensor prediction"""
    reconstruction_error: float
    threshold: float
    is_anomaly: bool
    risk_level: str


@router.post(
    "/sensor/predict",
    response_model=SensorPredictionResponse,
    summary="Predict leaks from sensor data"
)
async def predict_from_sensors(data: SensorDataRequest):
    """
    Analyze sensor sequence data to detect potential leaks.
    Uses Transformer autoencoder model.
    """
    try:
        prediction = TransformerService.predict(data.sequence)
        return SensorPredictionResponse(**prediction)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )