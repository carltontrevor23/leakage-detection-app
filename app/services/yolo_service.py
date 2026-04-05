# app/services/yolo_service.py
import uuid
from pathlib import Path
from typing import Dict, Any
import cv2

from app.config import settings


class YOLOService:
    """
    Singleton-style service for YOLOv8 inference.
    The model is loaded once when the class is first used.
    """
    
    _model = None
    
    @classmethod
    def get_model(cls):
        """Lazy-load the model only once."""
        if cls._model is None:
            from ultralytics import YOLO

            model_path = settings.YOLO_MODEL_PATH
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"YOLOv8 model not found at: {model_path}"
                )
            cls._model = YOLO(str(model_path))
        return cls._model
    
    @classmethod
    def predict(cls, image_path: str, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Run inference on an image.
        
        Args:
            image_path: Path to the image file
            conf_threshold: Minimum confidence for detections
            
        Returns:
            Dict containing:
            - result_image_path: path to the annotated image
            - detections: list of dicts with label, confidence, bbox
            - detection_count: int
        """
        model = cls.get_model()
        
        # Run inference
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            save=False
        )
        
        result = results[0]
        
        # Parse detections
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'label': result.names[int(box.cls)],
                    'confidence': round(float(box.conf), 4),
                    'bbox': {
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2)
                    }
                })
        
        # Save annotated image
        annotated = result.plot()
        
        result_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        result_full_path = settings.RESULT_DIR / result_filename
        cv2.imwrite(str(result_full_path), annotated)
        
        return {
            'result_image_path': f"results/{result_filename}",
            'detections': detections,
            'detection_count': len(detections),
        }
