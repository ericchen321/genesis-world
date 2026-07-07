import hashlib
import json
from pathlib import Path

import igl
import numpy as np
from PIL import Image

from genesis.live.session import GenesisLiveSession
from genesis.live.visual_telemetry import (
    ANCHOR_DEBUG_BOX_COLOR,
    CONTROLLER_DEBUG_BOX_COLOR,
    DEBUG_BOX_WIREFRAME_RADIUS,
    GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
)

_TWO_TET_VERTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_TWO_TETS = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int64)


def _write_scene_config(tmp_path: Path) -> Path:
    mesh_path = tmp_path / "two_tets.mesh"
    igl.writeMESH(str(mesh_path), _TWO_TET_VERTS, _TWO_TETS, np.empty((0, 3), dtype=np.int64))
    config = {
        "sim_options": {"dt": 0.001},
        "entities": [
            {
                "name": "body",
                "morph": {"type": "tet_mesh", "file": str(mesh_path)},
                "material": {"type": "elastic"},
                "anchors": [
                    {"anchor_id": "bottom_pin", "frame": "env_local", "box": [-0.05, -0.05, -1.05, 0.05, 0.05, -0.95]}
                ],
            }
        ],
    }
    config_path = tmp_path / "scene.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _assert_png_record(record):
    path = Path(record["path"])
    assert path.exists()
    assert record["byte_size"] == path.stat().st_size
    assert record["byte_size"] > 0
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        assert image.size == (record["width"], record["height"])


def _assert_vec3(values):
    assert len(values) == 3
    assert all(np.isfinite(float(value)) for value in values)


def test_rgb_triptych_writes_panels_stitched_metadata_and_visible_overlays(tmp_path):
    session = GenesisLiveSession(
        scene_config_path=str(_write_scene_config(tmp_path)),
        start_paused=True,
        output_dir=str(tmp_path / "outputs"),
    )
    session.dispatch(
        "probe.action.register",
        {
            "action_id": "diag_box_action",
            "action": "box_ee_grasp_and_move",
            "entity": "body",
            "duration_frames": 1,
            "controllers": [
                {
                    "controller_id": "diag_box",
                    "aabb_box": {"frame": "env_local", "box": [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05]},
                    "distance_scale": 0.5,
                }
            ],
        },
    )
    session.dispatch("probe.apply", {"action_id": "diag_box_action"})

    metadata = session.dispatch("sim.resume", {"steps": 10, "diagnostic_visual": {"mode": "rgb_triptych"}})[
        "visual_telemetry"
    ]

    assert metadata["rendered"] is True
    assert metadata["count"] == 1
    assert metadata["render_every_steps"] == 10
    assert metadata["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert metadata["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert metadata["renderer"]["camera_model"] == "pinhole"
    assert metadata["renderer"]["debug_camera"] is True
    assert metadata["renderer"]["panel_camera_models"] == {
        "top": "pinhole",
        "northeast": "pinhole",
        "southwest": "pinhole",
    }
    assert metadata["renderer"]["panel_debug_cameras"] == {
        "top": True,
        "northeast": True,
        "southwest": True,
    }
    assert metadata["panel_order"] == ["top", "northeast", "southwest"]
    assert set(metadata["panel_paths"]) == {"top", "northeast", "southwest"}
    assert Path(metadata["metadata_path"]).exists()

    _assert_png_record(metadata["stitched"])
    assert metadata["stitched"]["width"] == 768
    assert metadata["stitched"]["height"] == 256
    assert metadata["stitched"]["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert metadata["stitched"]["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert metadata["stitched"]["source_panel_count"] == 3

    assert len(metadata["views"]) == 3
    expected_hag4r_labels = {"top": "top", "northeast": "ne_3q", "southwest": "sw_3q"}
    for record in metadata["views"]:
        _assert_png_record(record)
        assert record["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
        assert record["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
        assert record["renderer"]["debug_camera"] is True
        assert record["renderer"]["native_camera_registered"] is True
        assert record["hag4r_label"] == expected_hag4r_labels[record["label"]]
        camera = record["camera"]
        assert camera["label"] == record["label"]
        assert camera["model"] == "pinhole"
        assert camera["debug"] is True
        assert camera["res"] == [256, 256]
        assert camera["fov"] > 0
        assert camera["near"] > 0
        assert camera["far"] > camera["near"]
        _assert_vec3(camera["pos"])
        _assert_vec3(camera["lookat"])
        _assert_vec3(camera["up"])
        _assert_vec3(camera["pose"]["pos"])
        _assert_vec3(camera["pose"]["lookat"])
        _assert_vec3(camera["pose"]["up"])

    assert metadata["frame_metadata"] == metadata["views"] + [metadata["stitched"]]
    assert any(overlay.get("kind") == "static_anchor" for overlay in metadata["overlays"])
    assert any(
        overlay.get("kind") == "live_box_controller" and overlay.get("controller_id") == "diag_box"
        for overlay in metadata["overlays"]
    )
    markers_by_kind = {record["kind"]: record for record in metadata["debug_markers"]}
    assert markers_by_kind["static_anchor"]["color"] == list(ANCHOR_DEBUG_BOX_COLOR)
    assert markers_by_kind["live_box_controller"]["color"] == list(CONTROLLER_DEBUG_BOX_COLOR)
    for marker in metadata["debug_markers"]:
        assert np.asarray(marker["bounds"], dtype=np.float32).shape == (2, 3)
        assert marker["wireframe"] is True
        assert marker["wireframe_radius"] == DEBUG_BOX_WIREFRAME_RADIUS

    serialized = Path(metadata["metadata_path"]).read_text(encoding="utf-8")
    assert GENESIS_NATIVE_DEBUG_CAMERA_RENDERER in serialized
    assert ("software" + "_surface_rasterizer") not in serialized


def test_visual_telemetry_source_has_no_software_triptych_renderer():
    live_dir = Path(__file__).resolve().parents[1] / "genesis" / "live"
    forbidden = (
        "software" + "_surface_rasterizer",
        "_" + "projection" + "_",
        "_" + "project_to_panel",
        "_" + "draw_box",
        "_" + "rasterize_surface_panel",
        "Image" + "Draw",
        "SCENE" + "_CONTEXT",
    )
    for path in live_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text
