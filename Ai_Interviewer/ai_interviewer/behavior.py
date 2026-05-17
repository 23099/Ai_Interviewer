"""Camera-based behavioral analysis using OpenCV."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


TRACKING_FIELDS = [
    "timestamp",
    "face_detected",
    "eye_contact_score",
    "posture_score",
    "expression_score",
    "stress_score",
    "attention_score",
    "confidence_score",
    "honesty_score",
    "movement_score",
    "expression_label",
]


@dataclass(frozen=True)
class BehaviorMetrics:
    face_detected: bool
    eye_contact_score: float
    posture_score: float
    expression_label: str
    stress_score: float
    notes: list[str]
    attention_score: float = 50.0
    confidence_score: float = 50.0
    honesty_score: float = 50.0
    expression_score: float = 50.0
    movement_score: float = 70.0
    sample_count: int = 0
    face_detection_rate: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data


def neutral_behavior(reason: str = "Camera not used") -> BehaviorMetrics:
    return BehaviorMetrics(
        face_detected=False,
        eye_contact_score=50.0,
        posture_score=50.0,
        expression_label="not captured",
        stress_score=45.0,
        notes=[reason],
        attention_score=50.0,
        confidence_score=50.0,
        honesty_score=50.0,
        expression_score=50.0,
        movement_score=70.0,
    )


def analyze_camera_image(image_file) -> BehaviorMetrics:
    """Analyze a Streamlit uploaded camera image without storing the image."""
    if image_file is None:
        return neutral_behavior()

    try:
        image = Image.open(image_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception:
        return neutral_behavior("Camera image could not be read")

    return analyze_frame(frame)


def analyze_frame(frame: np.ndarray, movement_score: float = 70.0) -> BehaviorMetrics:
    """Analyze one BGR video frame and return transparent heuristic metrics."""
    if frame is None or frame.size == 0:
        return neutral_behavior("Empty camera frame")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = _face_cascade()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return BehaviorMetrics(
            face_detected=False,
            eye_contact_score=10.0,
            posture_score=25.0,
            expression_label="not visible",
            stress_score=78.0,
            notes=["No face detected in this sample"],
            attention_score=10.0,
            confidence_score=22.0,
            honesty_score=28.0,
            expression_score=35.0,
            movement_score=movement_score,
        )

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    frame_h, frame_w = gray.shape
    face_roi = gray[y : y + h, x : x + w]
    face_center_x = x + w / 2
    face_center_y = y + h / 2
    horizontal_offset = abs(face_center_x - frame_w / 2) / max(frame_w / 2, 1)
    vertical_offset = abs(face_center_y - frame_h * 0.46) / max(frame_h * 0.46, 1)
    face_size_ratio = (w * h) / max(frame_w * frame_h, 1)

    eye_contact_score, eye_note = _estimate_eye_contact(face_roi, horizontal_offset)
    posture_score = _clamp(94.0 - vertical_offset * 72.0 - abs(face_size_ratio - 0.18) * 130.0)
    expression_label, expression_score, expression_note = _estimate_expression(face_roi)
    attention_score = _clamp(0.72 * eye_contact_score + 0.18 * posture_score + 0.10 * movement_score)
    stress_score = _clamp(
        100.0
        - 0.31 * eye_contact_score
        - 0.27 * posture_score
        - 0.24 * expression_score
        - 0.10 * movement_score
        + (8.0 if expression_label in {"tense", "uncertain"} else 0.0)
    )
    confidence_score = _clamp(
        0.30 * eye_contact_score
        + 0.30 * posture_score
        + 0.20 * expression_score
        + 0.10 * movement_score
        + 0.10 * (100.0 - stress_score)
    )
    honesty_score = _clamp(
        0.34 * attention_score
        + 0.24 * expression_score
        + 0.18 * posture_score
        + 0.14 * movement_score
        + 0.10 * (100.0 - stress_score)
    )

    return BehaviorMetrics(
        face_detected=True,
        eye_contact_score=round(eye_contact_score, 2),
        posture_score=round(posture_score, 2),
        expression_label=expression_label,
        stress_score=round(stress_score, 2),
        notes=[
            f"Eye-contact proxy: {eye_contact_score:.0f}",
            f"Posture framing score: {posture_score:.0f}",
            eye_note,
            expression_note,
        ],
        attention_score=round(attention_score, 2),
        confidence_score=round(confidence_score, 2),
        honesty_score=round(honesty_score, 2),
        expression_score=round(expression_score, 2),
        movement_score=round(movement_score, 2),
    )


def combine_stress(
    behavior: BehaviorMetrics,
    response_seconds: float,
    answer: str,
) -> BehaviorMetrics:
    """Blend visual stress with response delay and answer sparseness."""
    delay_penalty = 0.0
    if response_seconds > 60:
        delay_penalty = 12.0
    elif response_seconds > 35:
        delay_penalty = 6.0

    word_count = len(answer.split())
    answer_penalty = 8.0 if 0 < word_count < 8 else 0.0
    stress_score = _clamp(behavior.stress_score + delay_penalty + answer_penalty)
    confidence_score = _clamp(behavior.confidence_score - delay_penalty * 0.4 - answer_penalty * 0.5)
    honesty_score = _clamp(behavior.honesty_score - answer_penalty * 0.4)
    notes = list(behavior.notes)
    if delay_penalty:
        notes.append("Long response delay increased stress estimate")
    if answer_penalty:
        notes.append("Very short answer increased uncertainty")

    return BehaviorMetrics(
        face_detected=behavior.face_detected,
        eye_contact_score=behavior.eye_contact_score,
        posture_score=behavior.posture_score,
        expression_label=behavior.expression_label,
        stress_score=round(stress_score, 2),
        notes=notes,
        attention_score=behavior.attention_score,
        confidence_score=round(confidence_score, 2),
        honesty_score=round(honesty_score, 2),
        expression_score=behavior.expression_score,
        movement_score=behavior.movement_score,
        sample_count=behavior.sample_count,
        face_detection_rate=behavior.face_detection_rate,
    )


def behavior_from_tracking_summary(
    tracking_summary: dict,
    response_seconds: float,
    answer: str,
) -> BehaviorMetrics:
    if tracking_summary.get("sample_count", 0) == 0:
        return combine_stress(neutral_behavior("Camera tracking did not collect samples"), response_seconds, answer)

    averages = tracking_summary.get("averages", {})
    expression_distribution = tracking_summary.get("expression_distribution", {})
    expression_label = max(expression_distribution, key=expression_distribution.get, default="not captured")
    face_rate = float(tracking_summary.get("face_detection_rate", 0.0))
    behavior = BehaviorMetrics(
        face_detected=face_rate >= 0.5,
        eye_contact_score=round(float(averages.get("eye_contact_score", 50.0)), 2),
        posture_score=round(float(averages.get("posture_score", 50.0)), 2),
        expression_label=expression_label,
        stress_score=round(float(averages.get("stress_score", 45.0)), 2),
        notes=[
            f"Tracking samples: {tracking_summary.get('sample_count', 0)}",
            f"Face detected in {face_rate:.0f}% of samples",
            "Honesty score is a consistency heuristic, not lie detection",
        ],
        attention_score=round(float(averages.get("attention_score", 50.0)), 2),
        confidence_score=round(float(averages.get("confidence_score", 50.0)), 2),
        honesty_score=round(float(averages.get("honesty_score", 50.0)), 2),
        expression_score=round(float(averages.get("expression_score", 50.0)), 2),
        movement_score=round(float(averages.get("movement_score", 70.0)), 2),
        sample_count=int(tracking_summary.get("sample_count", 0)),
        face_detection_rate=round(face_rate, 2),
    )
    return combine_stress(behavior, response_seconds, answer)


def tracking_row(metrics: BehaviorMetrics, timestamp: str) -> dict:
    return {
        "timestamp": timestamp,
        "face_detected": int(metrics.face_detected),
        "eye_contact_score": metrics.eye_contact_score,
        "posture_score": metrics.posture_score,
        "expression_score": metrics.expression_score,
        "stress_score": metrics.stress_score,
        "attention_score": metrics.attention_score,
        "confidence_score": metrics.confidence_score,
        "honesty_score": metrics.honesty_score,
        "movement_score": metrics.movement_score,
        "expression_label": metrics.expression_label,
    }


def summarize_tracking_file(path: Path) -> dict:
    if not path.exists():
        return _empty_tracking_summary(str(path))

    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)

    if not rows:
        return _empty_tracking_summary(str(path))

    numeric_fields = [
        field
        for field in TRACKING_FIELDS
        if field not in {"timestamp", "expression_label", "face_detected"}
    ]
    averages = {
        field: round(_average(float(row.get(field) or 0.0) for row in rows), 2)
        for field in numeric_fields
    }
    expression_distribution: dict[str, int] = {}
    for row in rows:
        label = row.get("expression_label") or "unknown"
        expression_distribution[label] = expression_distribution.get(label, 0) + 1

    return {
        "path": str(path),
        "sample_count": len(rows),
        "face_detection_rate": round(_average(float(row.get("face_detected") or 0.0) for row in rows) * 100.0, 2),
        "averages": averages,
        "expression_distribution": expression_distribution,
        "started_at": rows[0].get("timestamp"),
        "ended_at": rows[-1].get("timestamp"),
    }


def _estimate_eye_contact(face_gray: np.ndarray, horizontal_offset: float) -> tuple[float, str]:
    eyes = _eye_cascade().detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6)
    center_score = _clamp(100.0 - horizontal_offset * 125.0)
    if len(eyes) >= 2:
        eye_centers = sorted((x + w / 2 for x, _y, w, _h in eyes[:2]))
        eye_midpoint = sum(eye_centers) / len(eye_centers)
        eye_offset = abs(eye_midpoint - face_gray.shape[1] / 2) / max(face_gray.shape[1] / 2, 1)
        return _clamp(0.65 * center_score + 0.35 * (100.0 - eye_offset * 110.0)), "Both eyes detected"
    if len(eyes) == 1:
        return _clamp(center_score * 0.82), "One eye detected"
    return _clamp(center_score * 0.62), "Eyes not clearly detected"


def _estimate_expression(face_gray: np.ndarray) -> tuple[str, float, str]:
    smiles = _smile_cascade().detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=20)
    brightness = float(np.mean(face_gray))
    contrast = float(np.std(face_gray))

    if len(smiles) > 0:
        return "positive", 88.0, "Smile-like expression detected"
    if contrast < 28 or brightness < 55:
        return "uncertain", 48.0, "Low visual clarity reduced expression confidence"
    if contrast > 68:
        return "tense", 52.0, "High contrast facial region may indicate tension"
    return "neutral", 68.0, "Neutral expression detected"


def _face_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _eye_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def _smile_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


def _empty_tracking_summary(path: str) -> dict:
    return {
        "path": path,
        "sample_count": 0,
        "face_detection_rate": 0.0,
        "averages": {},
        "expression_distribution": {},
        "started_at": None,
        "ended_at": None,
    }


def _average(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))
