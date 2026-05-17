"""Background webcam monitoring that writes tracking samples to CSV."""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .behavior import TRACKING_FIELDS, analyze_frame, summarize_tracking_file, tracking_row


class CameraMonitor:
    """Sample webcam behavior metrics until stopped."""

    def __init__(self, output_path: Path, interval_seconds: float = 1.0, camera_index: int = 0):
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.camera_index = camera_index
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.sample_count = 0
        self.status_message = "Not started"

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="camera-monitor", daemon=True)
        self._thread.start()

    def stop(self, timeout_seconds: float = 3.0) -> dict:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
        self.status_message = "Stopped"
        return summarize_tracking_file(self.output_path)

    def summary(self) -> dict:
        return summarize_tracking_file(self.output_path)

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.status_message = "Camera unavailable"
            return

        self.status_message = "Monitoring"
        last_sample_at = 0.0
        previous_gray: np.ndarray | None = None

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.status_message = "Camera frame unavailable"
                    time.sleep(0.2)
                    continue

                now = time.time()
                if now - last_sample_at < self.interval_seconds:
                    time.sleep(0.05)
                    continue

                movement_score, previous_gray = _movement_score(frame, previous_gray)
                metrics = analyze_frame(frame, movement_score=movement_score)
                self._append_row(
                    tracking_row(
                        metrics,
                        datetime.now().isoformat(timespec="seconds"),
                    )
                )
                self.sample_count += 1
                last_sample_at = now
        finally:
            cap.release()

    def _ensure_header(self) -> None:
        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            return
        with self.output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=TRACKING_FIELDS)
            writer.writeheader()

    def _append_row(self, row: dict) -> None:
        with self.output_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=TRACKING_FIELDS)
            writer.writerow(row)


def _movement_score(frame: np.ndarray, previous_gray: np.ndarray | None) -> tuple[float, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (80, 60))
    if previous_gray is None:
        return 78.0, small_gray

    diff = cv2.absdiff(small_gray, previous_gray)
    movement_level = float(np.mean(diff))
    stability_score = max(20.0, 100.0 - movement_level * 3.0)
    return round(stability_score, 2), small_gray
