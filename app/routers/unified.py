import csv
import io
import uuid
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.config import settings
from app.services.transformer_service import TransformerService
from app.services.yolo_service import YOLOService
from app.utils.file_handling import save_upload_file, validate_image

router = APIRouter()


def _uploaded(file: UploadFile | None) -> bool:
    return bool(file and file.filename)


def _has_error(result: dict[str, Any] | None) -> bool:
    return bool(result and result.get("error"))


def parse_sensor_csv(file: UploadFile, sequence_length: int, num_features: int) -> list[list[float]]:
    try:
        raw_bytes = file.file.read()
        file.file.seek(0)
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV file must be UTF-8 encoded.",
        ) from exc

    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if any(cell.strip() for cell in row)]

    if len(rows) < sequence_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"CSV must contain at least {sequence_length} rows of sensor data.",
        )

    first_row = rows[0]
    expected_columns = num_features
    header_like = any(cell.strip() == "" for cell in first_row)
    if not header_like:
        try:
            [float(cell) for cell in first_row]
        except ValueError:
            header_like = True

    data_rows = rows[1:] if header_like else rows

    if len(data_rows) < sequence_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"CSV must contain at least {sequence_length} rows of sensor data.",
        )

    selected_rows = data_rows[:sequence_length]

    for index, row in enumerate(selected_rows, start=1):
        if len(row) != expected_columns:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"CSV column mismatch at row {index}: expected "
                    f"{expected_columns} feature columns but found {len(row)}."
                ),
            )

    try:
        parsed = [[float(cell) for cell in row] for row in selected_rows]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV contains non-numeric sensor values in the required 56 feature columns.",
        ) from exc

    return parsed


def compute_risk_level(
    visual_result: dict[str, Any] | None,
    sensor_result: dict[str, Any] | None,
) -> tuple[str, str]:
    visual_flagged = bool(visual_result and not _has_error(visual_result) and visual_result.get("visual_leak_detected"))
    sensor_flagged = bool(sensor_result and not _has_error(sensor_result) and sensor_result.get("sensor_anomaly_detected"))

    if visual_flagged and sensor_flagged:
        return (
            "CRITICAL",
            "Both visual and sensor evidence of leakage detected. Shut down pipeline segment and dispatch inspection team immediately.",
        )
    if visual_flagged:
        return (
            "HIGH",
            "Visual leak detected but sensor data is normal. Possible surface leak or early-stage event. Dispatch visual inspection.",
        )
    if sensor_flagged:
        return (
            "ELEVATED",
            "Sensor anomaly detected but no visual leak confirmed. Possible internal or subsurface leak. Review sensor trends and inspect the segment.",
        )
    return (
        "NORMAL",
        "No leakage indicators detected. Continue routine monitoring.",
    )


@router.post("/unified-detect")
async def unified_detect(
    request: Request,
    image: UploadFile | None = File(None, description="Optional pipeline image to analyze"),
    sensor_csv: UploadFile | None = File(None, description="Optional sensor CSV to analyze"),
):
    if not _uploaded(image) and not _uploaded(sensor_csv):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of image or sensor_csv must be provided.",
        )

    visual_result: dict[str, Any] | None = None
    sensor_result: dict[str, Any] | None = None

    if _uploaded(image):
        inspection_id = uuid.uuid4().hex[:8]
        try:
            validate_image(
                file=image,
                allowed_types=settings.ALLOWED_IMAGE_TYPES,
                max_size_mb=settings.MAX_UPLOAD_SIZE_MB,
            )
            upload_path = save_upload_file(
                file=image,
                upload_dir=settings.UPLOAD_DIR,
                prefix=inspection_id,
            )
            prediction = YOLOService.predict(image_path=str(upload_path), conf_threshold=0.25)
            top_confidence = max((d["confidence"] for d in prediction["detections"]), default=0.0)
            visual_result = {
                "visual_leak_detected": bool(prediction["detection_count"] > 0),
                "visual_confidence": float(top_confidence),
                "annotated_image_url": f"/media/{prediction['result_image_path']}",
                "detection_count": prediction["detection_count"],
                "detections": prediction["detections"],
            }
        except HTTPException as exc:
            visual_result = {
                "error": exc.detail,
                "visual_leak_detected": False,
            }
        except Exception as exc:
            visual_result = {
                "error": f"Visual model error: {str(exc)}",
                "visual_leak_detected": False,
            }

    if _uploaded(sensor_csv):
        try:
            sequence_length = getattr(request.app.state, "sequence_length", settings.SEQUENCE_LENGTH)
            num_features = getattr(request.app.state, "num_features", settings.NUM_FEATURES)
            anomaly_threshold = getattr(request.app.state, "anomaly_threshold", settings.ANOMALY_THRESHOLD)

            sequence = parse_sensor_csv(
                file=sensor_csv,
                sequence_length=sequence_length,
                num_features=num_features,
            )

            prediction = TransformerService.predict(sequence)
            sensor_result = {
                "sensor_anomaly_detected": bool(prediction["is_anomaly"]),
                "reconstruction_error": float(prediction["reconstruction_error"]),
                "anomaly_threshold": float(anomaly_threshold),
                "risk_level": prediction["risk_level"],
                "status": prediction["status"],
                "message": prediction["message"],
                "rows_analyzed": len(sequence),
                "num_features": num_features,
            }
        except HTTPException as exc:
            if exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
                raise
            sensor_result = {
                "error": exc.detail,
                "sensor_anomaly_detected": False,
            }
        except ValueError as exc:
            message = str(exc)
            if "Expected input shape" in message:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"CSV columns do not match the expected {settings.NUM_FEATURES} feature columns. "
                        f"Details: {message}"
                    ),
                ) from exc
            sensor_result = {
                "error": f"Sensor model error: {message}",
                "sensor_anomaly_detected": False,
            }
        except Exception as exc:
            sensor_result = {
                "error": f"Sensor model error: {str(exc)}",
                "sensor_anomaly_detected": False,
            }
        finally:
            sensor_csv.file.close()

    risk_level, recommendation = compute_risk_level(visual_result, sensor_result)

    return {
        "risk_level": risk_level,
        "recommendation": recommendation,
        "visual_result": visual_result,
        "sensor_result": sensor_result,
    }
