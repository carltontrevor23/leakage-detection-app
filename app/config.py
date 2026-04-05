# app/config.py
from pathlib import Path
from typing import Any
from pydantic_settings import BaseSettings
from pydantic import field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings"""
    
    # FastAPI settings
    APP_NAME: str = "Pipeline Leak Detector"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    CORS_ALLOW_ORIGINS: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    
    # File upload settings
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/webp"]
    
    # Model paths
    YOLO_MODEL_PATH: Path = BASE_DIR / "ml_models" / "yolo" / "best.pt"
    TRANSFORMER_MODEL_PATH: Path = BASE_DIR / "ml_models" / "transformer" / "train3_transformer_autoencoder.keras"
    SCALER_PATH: Path = BASE_DIR / "ml_models" / "transformer" / "train3_transformer_scaler.pkl"
    MODEL_VERSION: str = "2026.04"
    ANOMALY_THRESHOLD: float = 0.0011916750946368263
    SEQUENCE_LENGTH: int = 20
    NUM_FEATURES: int = 56
    
    # Media directories
    MEDIA_ROOT: Path = BASE_DIR / "media"
    UPLOAD_DIR: Path = MEDIA_ROOT / "uploads"
    RESULT_DIR: Path = MEDIA_ROOT / "results"
    
    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> bool:
        """Accept boolean env values plus common mode strings."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on", "development", "dev", "debug"}:
                return True
            if normalized in {"false", "0", "no", "off", "release", "prod", "production"}:
                return False
        return bool(value)

    @field_validator("CORS_ALLOW_ORIGINS", mode="before")
    @classmethod
    def parse_cors_allow_origins(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Create global settings instance
settings = Settings()

# Create directories if they don't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULT_DIR.mkdir(parents=True, exist_ok=True)
