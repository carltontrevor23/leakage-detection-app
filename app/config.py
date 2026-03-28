# app/config.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings
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
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # File upload settings
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/webp"]
    
    # Model paths
    YOLO_MODEL_PATH: Path = BASE_DIR / "ml_models" / "yolo" / "best.pt"
    TRANSFORMER_MODEL_PATH: Path = BASE_DIR / "ml_models" / "transformer"
    
    # Media directories
    MEDIA_ROOT: Path = BASE_DIR / "media"
    UPLOAD_DIR: Path = MEDIA_ROOT / "uploads"
    RESULT_DIR: Path = MEDIA_ROOT / "results"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()

# Create directories if they don't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULT_DIR.mkdir(parents=True, exist_ok=True)