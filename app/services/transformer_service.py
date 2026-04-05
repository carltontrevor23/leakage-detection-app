from typing import Dict, Any, List
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

from app.config import settings


@register_keras_serializable(package="Custom")
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length,
            output_dim=d_model,
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "d_model": self.d_model,
            }
        )
        return config


class TransformerService:
    """
    Service for Transformer autoencoder inference on sensor sequence data.
    """

    _model = None
    _scaler = None

    @classmethod
    def get_model(cls):
        """Load transformer model once."""
        if cls._model is None:
            if not settings.TRANSFORMER_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Transformer model file not found at: {settings.TRANSFORMER_MODEL_PATH}"
                )

            try:
                cls._model = load_model(
                    str(settings.TRANSFORMER_MODEL_PATH),
                    custom_objects={"PositionalEmbedding": PositionalEmbedding},
                    compile=False,
                    safe_mode=False,
                )
                print(f"[Startup] Transformer model loaded successfully from {settings.TRANSFORMER_MODEL_PATH}")
            except Exception as e:
                raise RuntimeError(f"Failed to load transformer model: {e}")

        return cls._model

    @classmethod
    def get_scaler(cls):
        """Load scaler once."""
        if cls._scaler is None:
            if not settings.SCALER_PATH.exists():
                raise FileNotFoundError(
                    f"Scaler file not found at: {settings.SCALER_PATH}"
                )

            cls._scaler = joblib.load(settings.SCALER_PATH)
            print(f"[Startup] Scaler loaded successfully from {settings.SCALER_PATH}")

        return cls._scaler

    @classmethod
    def get_risk_level(cls, mse: float) -> str:
        """
        Convert reconstruction error into a human-readable risk level.
        """
        if mse <= settings.ANOMALY_THRESHOLD:
            return "low"
        elif mse <= settings.ANOMALY_THRESHOLD * 3:
            return "medium"
        else:
            return "high"

    @classmethod
    def get_status_message(cls, risk_level: str, is_anomaly: bool) -> Dict[str, str]:
        """
        Build user-friendly status and message for the prediction response.
        """
        if not is_anomaly:
            return {
                "status": "Normal operation",
                "message": "Sensor pattern is within the expected operating range.",
            }

        if risk_level == "high":
            return {
                "status": "Potential leak detected",
                "message": "Sensor pattern deviates significantly from normal operating conditions.",
            }
        elif risk_level == "medium":
            return {
                "status": "Suspicious sensor activity",
                "message": "Sensor pattern shows unusual behavior and should be investigated.",
            }
        else:
            return {
                "status": "Slight anomaly detected",
                "message": "Sensor pattern is slightly abnormal and should be monitored.",
            }

    @classmethod
    def validate_sequence_shape(cls, sensor_data: List[List[float]]) -> np.ndarray:
        x = np.array(sensor_data, dtype=np.float32)
        expected_shape = (settings.SEQUENCE_LENGTH, settings.NUM_FEATURES)

        if x.shape != expected_shape:
            raise ValueError(
                f"Expected input shape {expected_shape}, but got {x.shape}"
            )

        return x

    @classmethod
    def predict(cls, sensor_data: List[List[float]]) -> Dict[str, Any]:
        """
        Run inference on sequence sensor data.

        Args:
            sensor_data: 20x56 list of sensor readings

        Returns:
            Dict with anomaly prediction results
        """
        model = cls.get_model()
        scaler = cls.get_scaler()

        x = cls.validate_sequence_shape(sensor_data)

        x_2d = x.reshape(-1, settings.NUM_FEATURES)
        x_scaled_2d = scaler.transform(x_2d)
        x_scaled = x_scaled_2d.reshape(1, settings.SEQUENCE_LENGTH, settings.NUM_FEATURES)

        reconstruction = model.predict(x_scaled, verbose=0)
        mse = np.mean(np.square(x_scaled - reconstruction))
        is_anomaly = mse > settings.ANOMALY_THRESHOLD
        risk_level = cls.get_risk_level(float(mse))
        status_info = cls.get_status_message(risk_level, bool(is_anomaly))

        return {
            "reconstruction_error": float(mse),
            "threshold": float(settings.ANOMALY_THRESHOLD),
            "is_anomaly": bool(is_anomaly),
            "risk_level": risk_level,
            "status": status_info["status"],
            "message": status_info["message"],
        }
