# app/utils/file_handling.py
import shutil
import uuid
from pathlib import Path
from typing import Union
from fastapi import UploadFile, HTTPException, status
from PIL import Image
import os


def validate_image(file: UploadFile, allowed_types: list, max_size_mb: int) -> bool:
    """
    Validate uploaded file is an allowed image type and within size limits.
    
    Args:
        file: FastAPI UploadFile object
        allowed_types: List of allowed MIME types (e.g., ['image/jpeg', 'image/png'])
        max_size_mb: Maximum file size in MB
    
    Returns:
        bool: True if valid
    
    Raises:
        HTTPException: If validation fails
    """
    # Check content type
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Check file size by reading the file
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {max_size_mb} MB"
        )
    
    # Optional: Try to open as image to verify it's valid
    try:
        # Read first few bytes to check if it's a valid image
        file.file.seek(0)
        Image.open(file.file)
        file.file.seek(0)  # Reset for later reading
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not a valid image"
        )
    
    return True


def save_upload_file(
    file: UploadFile,
    upload_dir: Union[str, Path],
    prefix: str = None
) -> Path:
    """
    Save uploaded file to disk with unique filename.
    
    Args:
        file: FastAPI UploadFile object
        upload_dir: Directory to save the file
        prefix: Optional prefix for filename (e.g., inspection ID)
        
    Returns:
        Path to saved file
    """
    # Ensure directory exists
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Get file extension
    original_filename = Path(file.filename)
    file_extension = original_filename.suffix.lower()
    
    # Default to .jpg if no extension
    if not file_extension:
        file_extension = ".jpg"
    
    # Generate unique filename
    if prefix:
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}{file_extension}"
    else:
        filename = f"{uuid.uuid4().hex[:8]}{file_extension}"
    
    file_path = upload_dir / filename
    
    # Save file
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
    finally:
        file.file.close()
    
    return file_path


def delete_file(file_path: Union[str, Path]) -> bool:
    """
    Delete a file from disk.
    
    Args:
        file_path: Path to the file to delete
        
    Returns:
        bool: True if deleted successfully or file doesn't exist
    """
    file_path = Path(file_path)
    if file_path.exists():
        try:
            file_path.unlink()
            return True
        except Exception:
            return False
    return True


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Get file size in MB"""
    file_path = Path(file_path)
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory