from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .overlay_state import anchor_overlay_records, controller_overlay_records
from .triptych import HAG4R_LABELS, PANEL_ORDER, png_record, stitch_triptych

PANEL_SIZE = (256, 256)
GENESIS_NATIVE_DEBUG_CAMERA_RENDERER = "genesis_native_debug_camera_renderer"
ANCHOR_DEBUG_BOX_COLOR = (0.16, 0.43, 1.0, 1.0)
CONTROLLER_DEBUG_BOX_COLOR = (1.0, 0.35, 0.16, 1.0)
DEBUG_BOX_WIREFRAME_RADIUS = 0.004


def _entity_positions(entity) -> np.ndarray:
    from genesis.utils.misc import tensor_to_array

    if getattr(entity, "active", False):
        positions = tensor_to_array(entity.get_state().pos)
    else:
        positions = tensor_to_array(entity.init_positions)

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim == 3:
        positions = positions[0]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"FEM entity positions must have shape (N, 3); got {positions.shape}")
    if not np.all(np.isfinite(positions)):
        raise ValueError("FEM entity positions must be finite")
    return positions


def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError(f"Cannot construct RGB triptych camera with degenerate {label} vector")
    return (vector / norm).astype(np.float32)


def _triptych_world_bounds(session, boxes: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for entity in session.entities.values():
        positions = _entity_positions(entity)
        mins.append(positions.min(axis=0))
        maxs.append(positions.max(axis=0))

    for box in boxes or []:
        box = np.asarray(box, dtype=np.float32)
        if box.shape != (6,):
            raise ValueError(f"RGB triptych overlay boxes must have shape (6,); got {box.shape}")
        if not np.all(np.isfinite(box)):
            raise ValueError("RGB triptych overlay boxes must be finite")
        if np.any(box[:3] > box[3:]):
            raise ValueError(f"RGB triptych overlay box has inverted min/max bounds: {box.tolist()}")
        mins.append(box[:3])
        maxs.append(box[3:])

    if not mins:
        mins.append(np.array([-0.5, -0.5, -0.5], dtype=np.float32))
        maxs.append(np.array([0.5, 0.5, 0.5], dtype=np.float32))

    bbox_min = np.min(np.stack(mins), axis=0)
    bbox_max = np.max(np.stack(maxs), axis=0)
    pad = np.maximum(0.05, 0.08 * np.maximum(bbox_max - bbox_min, 1e-6))
    return bbox_min - pad, bbox_max + pad


def _triptych_camera_pose(label: str, bbox_min: np.ndarray, bbox_max: np.ndarray) -> dict[str, tuple[float, float, float]]:
    if label not in PANEL_ORDER:
        raise ValueError(f"Unsupported RGB triptych panel label: {label}")

    center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
    max_extent = float(np.max(bbox_max - bbox_min))
    d = max(0.35, 2.5 * max_extent)
    if label == "top":
        position = center + np.array([0.0, d, 0.0], dtype=np.float32)
        nominal_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    elif label == "northeast":
        position = center + np.array([d, 0.75 * d, d], dtype=np.float32)
        nominal_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        position = center + np.array([-d, 0.75 * d, -d], dtype=np.float32)
        nominal_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    lookat = center
    view_dir = _normalize(lookat - position, "view direction")
    right = _normalize(np.cross(view_dir, nominal_up), "right")
    up = _normalize(np.cross(right, view_dir), "up")
    return {
        "pos": tuple(float(value) for value in position),
        "lookat": tuple(float(value) for value in lookat),
        "up": tuple(float(value) for value in up),
    }


def _overlay_box_values(record: dict[str, Any]) -> np.ndarray:
    box = record.get("env_local_box")
    if box is None:
        box = record.get("source_box")
    if box is None:
        raise ValueError(f"RGB triptych overlay {record.get('kind')!r} is missing box bounds")
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (6,):
        raise ValueError(f"RGB triptych overlay box must have shape (6,); got {box.shape}")
    if not np.all(np.isfinite(box)):
        raise ValueError("RGB triptych overlay box must be finite")
    if np.any(box[:3] > box[3:]):
        raise ValueError(f"RGB triptych overlay box has inverted min/max bounds: {box.tolist()}")
    return box


def _boxes_from_overlays(records: list[dict[str, Any]]) -> list[np.ndarray]:
    boxes = []
    for record in records:
        kind = record.get("kind")
        if kind not in {"static_anchor", "live_box_controller"}:
            raise ValueError(f"Unsupported RGB triptych overlay kind: {kind!r}")
        boxes.append(_overlay_box_values(record))
    return boxes


def _overlay_box_bounds(record: dict[str, Any]) -> np.ndarray:
    box = _overlay_box_values(record)
    return np.stack((box[:3], box[3:])).astype(np.float32)


def _overlay_debug_color(record: dict[str, Any]) -> tuple[float, float, float, float]:
    if record.get("kind") == "static_anchor":
        return ANCHOR_DEBUG_BOX_COLOR
    if record.get("kind") == "live_box_controller":
        return CONTROLLER_DEBUG_BOX_COLOR
    raise ValueError(f"Unsupported RGB triptych overlay kind: {record.get('kind')!r}")


def _normalize_camera_rgb(rgb: Any, *, label: str, camera: Any) -> np.ndarray:
    from genesis.utils.misc import tensor_to_array

    array = np.asarray(tensor_to_array(rgb))
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Genesis debug camera {label!r} returned RGB with invalid shape {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]

    expected_width, expected_height = camera.res
    if array.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            f"Genesis debug camera {label!r} returned RGB shape {array.shape[:2]}, "
            f"expected {(expected_height, expected_width)}"
        )

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _render_camera_rgb(camera: Any, *, label: str) -> np.ndarray:
    rgb, _depth, _seg, _normal = camera.render(
        rgb=True,
        depth=False,
        segmentation=False,
        normal=False,
        force_render=True,
    )
    if rgb is None:
        raise ValueError(f"Genesis debug camera {label!r} did not return RGB data")
    return _normalize_camera_rgb(rgb, label=label, camera=camera)


def _json_vec3(value: Any) -> list[float]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"RGB triptych camera vector must have shape (3,); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("RGB triptych camera vector must be finite")
    return [float(component) for component in array]


def _camera_metadata(camera: Any, label: str, pose: dict[str, tuple[float, float, float]]) -> dict[str, Any]:
    return {
        "label": label,
        "model": camera.model,
        "debug": bool(camera.debug),
        "res": [int(camera.res[0]), int(camera.res[1])],
        "fov": float(camera.fov),
        "near": float(camera.near),
        "far": float(camera.far),
        "pos": _json_vec3(camera.pos),
        "lookat": _json_vec3(camera.lookat),
        "up": _json_vec3(camera.up),
        "pose": {
            "pos": [float(value) for value in pose["pos"]],
            "lookat": [float(value) for value in pose["lookat"]],
            "up": [float(value) for value in pose["up"]],
        },
    }


class VisualTelemetry:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.triptych_cameras: dict[str, Any] = {}
        self._triptych_debug_marker_handles: list[Any] = []

    def reset_triptych_cameras(self) -> None:
        self.triptych_cameras.clear()
        self._triptych_debug_marker_handles.clear()

    def register_triptych_cameras(self, session) -> None:
        self.reset_triptych_cameras()
        if session.scene is None:
            raise ValueError("Cannot register RGB triptych cameras before the Genesis scene exists")
        bbox_min, bbox_max = _triptych_world_bounds(session)
        for label in PANEL_ORDER:
            pose = _triptych_camera_pose(label, bbox_min, bbox_max)
            camera = session.scene.add_camera(
                model="pinhole",
                res=PANEL_SIZE,
                pos=pose["pos"],
                lookat=pose["lookat"],
                up=pose["up"],
                GUI=False,
                debug=True,
            )
            self.triptych_cameras[label] = camera
        if tuple(self.triptych_cameras) != PANEL_ORDER:
            raise ValueError("RGB triptych camera registration did not produce the expected panel order")

    def triptych_camera(self, label: str):
        if label not in self.triptych_cameras:
            raise ValueError(f"RGB triptych camera is not registered for panel {label!r}")
        return self.triptych_cameras[label]

    def _clear_triptych_debug_markers(self, session) -> None:
        if session.scene is None:
            self._triptych_debug_marker_handles.clear()
            return
        while self._triptych_debug_marker_handles:
            handle = self._triptych_debug_marker_handles.pop()
            session.scene.clear_debug_object(handle)

    def _draw_triptych_debug_markers(self, session, overlays: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if session.scene is None:
            raise ValueError("Cannot draw RGB triptych debug markers before the Genesis scene exists")
        self._clear_triptych_debug_markers(session)
        marker_records = []
        for index, record in enumerate(overlays):
            bounds = _overlay_box_bounds(record)
            color = _overlay_debug_color(record)
            handle = session.scene.draw_debug_box(
                bounds,
                color=color,
                wireframe=True,
                wireframe_radius=DEBUG_BOX_WIREFRAME_RADIUS,
            )
            self._triptych_debug_marker_handles.append(handle)
            marker_records.append(
                {
                    "index": int(index),
                    "kind": record.get("kind"),
                    "bounds": bounds.tolist(),
                    "color": list(color),
                    "wireframe": True,
                    "wireframe_radius": DEBUG_BOX_WIREFRAME_RADIUS,
                }
            )
        return marker_records

    def capture_rgb_triptych(self, session, *, frame_index: int | None = None) -> dict[str, Any]:
        if frame_index is None:
            frame_index = int(session.current_step)
        anchor_records = anchor_overlay_records(session.anchor_records)
        controller_records = controller_overlay_records(session.controllers)
        overlays = anchor_records + controller_records
        boxes = _boxes_from_overlays(overlays)
        bbox_min, bbox_max = _triptych_world_bounds(session, boxes)

        marker_records = []
        panel_records = []
        panel_paths = []
        try:
            marker_records = self._draw_triptych_debug_markers(session, overlays)
            for label in PANEL_ORDER:
                camera = self.triptych_camera(label)
                if not bool(camera.debug):
                    raise ValueError(f"RGB triptych camera {label!r} is not a debug camera")
                pose = _triptych_camera_pose(label, bbox_min, bbox_max)
                camera.set_pose(pos=pose["pos"], lookat=pose["lookat"], up=pose["up"])
                rgb = _render_camera_rgb(camera, label=label)

                path = self.output_dir / "png_rgb_panels" / label / f"frame_{frame_index:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb, mode="RGB").save(path)
                panel_paths.append(path)
                panel_record = png_record(
                    path,
                    label=label,
                    hag4r_label=HAG4R_LABELS[label],
                    frame_index=frame_index,
                    simulation_step=session.current_step,
                )
                panel_record.update(
                    {
                        "render_backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                        "renderer": {
                            "backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                            "debug_camera": True,
                            "native_camera_registered": True,
                        },
                        "camera": _camera_metadata(camera, label, pose),
                    }
                )
                panel_records.append(panel_record)

            triptych_path = self.output_dir / "png_rgb_triptych" / f"frame_{frame_index:06d}.png"
            stitch_triptych(panel_paths, triptych_path)
            triptych_record = png_record(
                triptych_path,
                label="triptych",
                hag4r_label="triptych",
                frame_index=frame_index,
                simulation_step=session.current_step,
            )
            triptych_record.update(
                {
                    "render_backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                    "renderer": {"backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER},
                    "source_panel_count": 3,
                }
            )

            metadata_path = self.output_dir / "rgb_triptych_metadata" / f"frame_{frame_index:06d}.json"
            metadata = {
                "requested": True,
                "mode": "rgb_triptych",
                "rendered": True,
                "render_backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                "renderer": {
                    "backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                    "camera_model": "pinhole",
                    "debug_camera": True,
                    "panel_camera_models": {
                        record["label"]: record["camera"]["model"] for record in panel_records
                    },
                    "panel_debug_cameras": {
                        record["label"]: record["camera"]["debug"] for record in panel_records
                    },
                    "panel_size": list(PANEL_SIZE),
                },
                "geometry": {
                    "entity_count": len(session.entities),
                    "bounds": {"min": bbox_min.tolist(), "max": bbox_max.tolist()},
                },
                "panel_order": list(PANEL_ORDER),
                "panel_paths": {record["label"]: record["path"] for record in panel_records},
                "views": panel_records,
                "stitched": triptych_record,
                "frame_metadata": panel_records + [triptych_record],
                "overlays": overlays,
                "debug_markers": marker_records,
                "capture_time": time.time(),
                "metadata_path": str(metadata_path),
            }
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return metadata
        finally:
            self._clear_triptych_debug_markers(session)
