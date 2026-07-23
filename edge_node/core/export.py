"""Violation event export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from edge_node.core.contracts import PlateObservation, Point, ViolationEvent


def point_to_dict(point: Point) -> dict[str, float]:
    return {"x": point.x, "y": point.y}


def plate_to_dict(plate: PlateObservation | None) -> dict[str, object] | None:
    if plate is None:
        return None
    return {
        "text": plate.text,
        "confidence": plate.confidence,
        "metadata": dict(plate.metadata),
    }


def event_to_dict(event: ViolationEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "track_id": event.track_id,
        "frame_index": event.frame_index,
        "timestamp_ms": event.timestamp_ms,
        "crossing_point": point_to_dict(event.crossing_point),
        "previous_point": point_to_dict(event.previous_point),
        "bbox_xyxy": list(event.bbox.as_xyxy()),
        "light_state": event.light_state.value,
        "light_confidence": event.light_confidence,
        "previous_side": event.previous_side,
        "current_side": event.current_side,
        "plate": plate_to_dict(event.plate),
        "metadata": dict(event.metadata),
    }


def export_events(events: Iterable[ViolationEvent], output_path: Path, export_format: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [event_to_dict(event) for event in events]

    if export_format == "json":
        with path.open("w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2)
        return

    if export_format == "csv":
        fieldnames = [
            "event_id",
            "track_id",
            "frame_index",
            "timestamp_ms",
            "crossing_x",
            "crossing_y",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "light_state",
            "light_confidence",
            "plate_text",
            "plate_confidence",
        ]
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for event in rows:
                bbox = event["bbox_xyxy"]
                crossing = event["crossing_point"]
                plate = event["plate"] or {}
                writer.writerow(
                    {
                        "event_id": event["event_id"],
                        "track_id": event["track_id"],
                        "frame_index": event["frame_index"],
                        "timestamp_ms": event["timestamp_ms"],
                        "crossing_x": crossing["x"],
                        "crossing_y": crossing["y"],
                        "bbox_x1": bbox[0],
                        "bbox_y1": bbox[1],
                        "bbox_x2": bbox[2],
                        "bbox_y2": bbox[3],
                        "light_state": event["light_state"],
                        "light_confidence": event["light_confidence"],
                        "plate_text": plate.get("text"),
                        "plate_confidence": plate.get("confidence"),
                    }
                )
        return

    raise ValueError(f"Unsupported export format: {export_format}")
