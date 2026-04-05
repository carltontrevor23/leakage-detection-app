# app/routers/sensor.py

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from typing import List

from app.config import settings
from app.models.detection import SensorAnomalyResponse
from app.services.transformer_service import TransformerService

router = APIRouter()


class SensorDataRequest(BaseModel):
    """Request schema for transformer sensor data"""
    sequence: List[List[float]] = Field(
        ...,
        description="A 20x56 sequence of sensor values",
        example=[[1.0] * settings.NUM_FEATURES for _ in range(settings.SEQUENCE_LENGTH)]
    )

    @model_validator(mode="after")
    def validate_sequence_shape(self):
        if len(self.sequence) != settings.SEQUENCE_LENGTH:
            raise ValueError(f"Input must contain exactly {settings.SEQUENCE_LENGTH} time steps")

        for row in self.sequence:
            if len(row) != settings.NUM_FEATURES:
                raise ValueError(f"Each time step must contain exactly {settings.NUM_FEATURES} features")

        return self


@router.post(
    "/sensor/predict",
    response_model=SensorAnomalyResponse,
    summary="Detect anomalies from sensor data"
)
async def predict_from_sensors(data: SensorDataRequest):
    """
    Analyze sensor sequence data to detect anomalies.
    Uses Transformer autoencoder model.
    """
    try:
        prediction = TransformerService.predict(data.sequence)
        return SensorAnomalyResponse(**prediction)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
