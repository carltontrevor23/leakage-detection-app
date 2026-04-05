import json
from typing import Any, Dict, List


class MultimodalService:
    """
    Combine image detections and sensor anomalies into one inspection-level result.

    This is currently a transparent heuristic fusion layer, not a learned fusion model.
    """

    @staticmethod
    def parse_sequence(sequence_payload: str) -> List[List[float]]:
        from app.services.transformer_service import TransformerService

        try:
            parsed = json.loads(sequence_payload)
        except json.JSONDecodeError as exc:
            raise ValueError("sensor_sequence must be valid JSON") from exc

        if not isinstance(parsed, list):
            raise ValueError("sensor_sequence must be a JSON array")

        normalized_rows: List[List[float]] = []
        for row in parsed:
            if not isinstance(row, list):
                raise ValueError("sensor_sequence must be a 2D JSON array")
            normalized_rows.append([float(value) for value in row])

        TransformerService.validate_sequence_shape(normalized_rows)
        return normalized_rows

    @staticmethod
    def fuse_predictions(
        image_prediction: Dict[str, Any],
        sensor_prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        threshold = max(sensor_prediction["threshold"], 1e-12)
        normalized_sensor_score = min(
            sensor_prediction["reconstruction_error"] / (threshold * 3.0),
            1.0,
        )
        image_signal = 1.0 if image_prediction["detection_count"] > 0 else 0.0
        combined_score = round((0.6 * image_signal) + (0.4 * normalized_sensor_score), 4)

        if combined_score >= 0.75:
            overall_status = "high_risk"
            decision = "Strong combined evidence of a potential leak event."
        elif combined_score >= 0.4:
            overall_status = "medium_risk"
            decision = "Mixed evidence detected; inspection is recommended."
        else:
            overall_status = "low_risk"
            decision = "Combined evidence is currently weak."

        return {
            "combined_risk_score": combined_score,
            "overall_status": overall_status,
            "decision": decision,
            "fusion_method": "heuristic_weighted_fusion_v1",
            "image_evidence_present": bool(image_prediction["detection_count"] > 0),
            "sensor_anomaly_present": bool(sensor_prediction["is_anomaly"]),
        }

    @classmethod
    def predict(cls, image_path: str, sensor_sequence: List[List[float]], conf_threshold: float) -> Dict[str, Any]:
        from app.services.transformer_service import TransformerService
        from app.services.yolo_service import YOLOService

        image_prediction = YOLOService.predict(
            image_path=image_path,
            conf_threshold=conf_threshold,
        )
        sensor_prediction = TransformerService.predict(sensor_sequence)
        fused = cls.fuse_predictions(image_prediction, sensor_prediction)

        return {
            "image_prediction": image_prediction,
            "sensor_prediction": sensor_prediction,
            "fusion": fused,
        }
