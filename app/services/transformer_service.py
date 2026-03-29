from typing import Dict, Any, List
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable


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

    MODEL_PATH = os.path.join(
        "ml_models", "transformer", "train3_transformer_autoencoder.keras"
    )
    SCALER_PATH = os.path.join(
        "ml_models", "transformer", "train3_transformer_scaler.pkl"
    )

    THRESHOLD = 0.0011916750946368263
    SEQUENCE_LENGTH = 20
    NUM_FEATURES = 56

    @classmethod
    def get_model(cls):
        """Load transformer model once."""
        if cls._model is None:
            if not os.path.exists(cls.MODEL_PATH):
                raise FileNotFoundError(
                    f"Transformer model file not found at: {cls.MODEL_PATH}"
                )

            try:
                cls._model = load_model(
                    cls.MODEL_PATH,
                    custom_objects={"PositionalEmbedding": PositionalEmbedding},
                    compile=False,
                    safe_mode=False,
                )
                print(f"[Startup] Transformer model loaded successfully from {cls.MODEL_PATH}")
            except Exception as e:
                raise RuntimeError(f"Failed to load transformer model: {e}")

        return cls._model

    @classmethod
    def get_scaler(cls):
        """Load scaler once."""
        if cls._scaler is None:
            if not os.path.exists(cls.SCALER_PATH):
                raise FileNotFoundError(
                    f"Scaler file not found at: {cls.SCALER_PATH}"
                )

            cls._scaler = joblib.load(cls.SCALER_PATH)
            print(f"[Startup] Scaler loaded successfully from {cls.SCALER_PATH}")

        return cls._scaler

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

        x = np.array(sensor_data, dtype=np.float32)

        if x.shape != (cls.SEQUENCE_LENGTH, cls.NUM_FEATURES):
            raise ValueError(
                f"Expected input shape ({cls.SEQUENCE_LENGTH}, {cls.NUM_FEATURES}), "
                f"but got {x.shape}"
            )

        x_2d = x.reshape(-1, cls.NUM_FEATURES)
        x_scaled_2d = scaler.transform(x_2d)
        x_scaled = x_scaled_2d.reshape(1, cls.SEQUENCE_LENGTH, cls.NUM_FEATURES)

        reconstruction = model.predict(x_scaled, verbose=0)
        mse = np.mean(np.square(x_scaled - reconstruction))
        is_anomaly = mse > cls.THRESHOLD

        return {
            "reconstruction_error": float(mse),
            "threshold": float(cls.THRESHOLD),
            "is_anomaly": bool(is_anomaly),
            "risk_level": "high" if is_anomaly else "low",
        }