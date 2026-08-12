import json
import socket
import subprocess
import sys
import time
import hashlib
from pathlib import Path
from types import SimpleNamespace

import igl
import numpy as np
import pytest
from PIL import Image

import genesis as gs
from genesis.engine.controllers.box_end_effector import BoxAnchorRecord
from genesis.live.capabilities import capability_report
from genesis.live.protocol import (
    ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY,
    MAX_MESSAGE_BYTES,
    PROTOCOL,
    GenesisLiveError,
    encode_frame,
    recv_json,
    send_json,
)
from genesis.live.server import GenesisLiveServer
from genesis.live.session import GenesisLiveSession, ProbeMeasurementTracker
from genesis.live.visual_telemetry import GENESIS_NATIVE_DEBUG_CAMERA_RENDERER

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


def _measurement_session_with_anchors(anchors: list[BoxAnchorRecord]) -> GenesisLiveSession:
    entity = object()
    session = GenesisLiveSession.__new__(GenesisLiveSession)
    session.entities = {"body": entity}
    session.anchor_records = {"body": anchors}
    session.probe_measurements = {}
    session.controllers = {}
    session.visual_telemetry = SimpleNamespace(freeze_triptych_for_measurement=lambda *_: None)
    session._entity_positions_for_validation = lambda *_args, **_kwargs: _TWO_TET_VERTS
    return session


def _measurement_anchor(anchor_id: str, selected_vertices: list[int]) -> BoxAnchorRecord:
    box = np.array([-0.1, -0.1, -0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    return BoxAnchorRecord(
        anchor_id=anchor_id,
        frame="env_local",
        source_box=box.copy(),
        env_local_box=box.copy(),
        selected_vertices=np.asarray(selected_vertices, dtype=np.int64),
    )


def test_probe_measurement_matches_canonical_static_anchor_id_and_release():
    anchor_id = "brim_midspan_from_front_band"
    session = _measurement_session_with_anchors([_measurement_anchor(anchor_id, [0])])

    tracker = session.begin_probe_measurement(
        measurement={"measurement_id": "measurement_0001", "anchor_id": anchor_id},
        entity_name="body",
        controller_id="controller_1",
        target_vertices=np.array([1], dtype=np.int64),
    )
    session.release_probe_measurement("controller_1")

    assert tracker.anchor_id == anchor_id
    assert tracker.phase == "post_release"


def test_probe_measurement_rejects_conflicting_static_anchor_matches():
    anchor_id = "brim_midspan_from_front_band"
    session = _measurement_session_with_anchors(
        [_measurement_anchor(anchor_id, [0]), _measurement_anchor(anchor_id, [1])]
    )

    with pytest.raises(GenesisLiveError, match="exactly one non-empty matching static anchor") as exc_info:
        session.begin_probe_measurement(
            measurement={"measurement_id": "measurement_0001", "anchor_id": anchor_id},
            entity_name="body",
            controller_id="controller_1",
            target_vertices=np.array([2], dtype=np.int64),
        )

    assert exc_info.value.code == "invalid_probe_measurement"
    assert exc_info.value.details["matches"] == 2


def test_non_y_live_apply_passes_runtime_selected_vertices_to_tracker(tmp_path):
    session = GenesisLiveSession(
        scene_config_path=str(_write_scene_config(tmp_path)), start_paused=True, output_dir=str(tmp_path / "outputs")
    )
    response = session.dispatch(
        "probe.apply",
        {
            "action": "box_ee_grasp_and_move",
            "entity": "body",
            "duration_steps": 12,
            "measurement": {"measurement_id": "non_y_measurement", "anchor_id": "bottom_pin", "motion_axis": "-X"},
            "controllers": [
                {
                    "controller_id": "non_y_box",
                    "aabb_box": {"frame": "env_local", "box": [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05]},
                    "distance_scale": 0.5,
                    "motion_axis": "-X",
                }
            ],
        },
    )
    selected = response["probe"]["selected_vertices"]
    tracker = session.probe_measurements["non_y_measurement"]

    assert selected == [3]
    assert tracker.target_vertices.tolist() == selected
    assert session.controllers["non_y_box"].state.motion_axis == "-X"
    session.dispatch("sim.resume", {"steps": 12})
    assert tracker.samples[-1]["controller_lineage"]["motion_axis"] == "-X"


def test_probe_measurement_socket_payload_is_bounded_and_keeps_large_trace_private(tmp_path):
    target_vertices = np.arange(1, 100_001, dtype=np.int64)
    positions = np.zeros((100_001, 3), dtype=np.float64)
    entity = object()
    controller = SimpleNamespace(
        snapshot=lambda: {
            "displacement": [0.0, 0.01, 0.0],
            "distance": 0.01,
            "moved_distance": 0.01,
            "duration_steps": 32,
            "estimated_motion_steps": 32,
            "active": True,
            "motion_active": True,
            "selected_vertices": target_vertices.tolist(),
            "target_positions": positions[target_vertices].tolist(),
        }
    )
    sampling_session = SimpleNamespace(
        entities={"body": entity},
        controllers={"controller_1": controller},
        current_step=1,
        _entity_positions_for_validation=lambda *_args, **_kwargs: positions,
    )
    tracker = ProbeMeasurementTracker(
        measurement_id="measurement_0001",
        entity_name="body",
        entity=entity,
        controller_id="controller_1",
        anchor_id="brim_midspan_from_front_band",
        target_vertices=target_vertices,
        anchor_vertices=np.array([0], dtype=np.int64),
        baseline_target_centroid=np.zeros(3),
        baseline_anchor_centroid=np.zeros(3),
        baseline_relative=np.zeros(3),
        lineage={},
    )
    tracker.sample(sampling_session)
    tracker.samples *= 1_000

    session = GenesisLiveSession.__new__(GenesisLiveSession)
    session.output_dir = str(tmp_path)
    session.probe_measurements = {tracker.measurement_id: tracker}
    payload = session._probe_measurement_payload()
    encoded_response = encode_frame({"ok": True, "data": {"probe_measurement": payload}})

    assert len(encoded_response) < 4_096
    assert len(encoded_response) < MAX_MESSAGE_BYTES
    assert b"target_vertices" not in encoded_response
    assert b"samples" not in encoded_response
    assert b"target_positions" not in encoded_response
    artifact_path = tmp_path / payload["private_trace_ref"]["relative_path"]
    private_trace = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert private_trace["target_vertices"] == target_vertices.tolist()
    assert len(private_trace["samples"]) == 1_000
    assert "target_positions" not in private_trace["samples"][0]["controller_lineage"]


def test_probe_measurement_capability_is_reported_and_requireable():
    report = capability_report((ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY,), diagnostic_scene=True)

    assert ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY in report["capabilities"]
    assert report["missing_required_capabilities"] == []


def _write_scene_config(tmp_path: Path) -> Path:
    mesh_path = tmp_path / "two_tets.mesh"
    igl.writeMESH(str(mesh_path), _TWO_TET_VERTS, _TWO_TETS, np.empty((0, 3), dtype=np.int64))
    config = {
        "sim_options": {"dt": 0.001},
        "fem_options": {"enable_vertex_constraints": True, "enable_floor": True},
        "entities": [
            {
                "name": "body",
                "morph": {"type": "tet_mesh", "file": str(mesh_path)},
                "material": {"type": "elastic", "E": 100000.0, "nu": 0.3, "rho": 1000.0},
                "anchors": [
                    {"anchor_id": "bottom_pin", "frame": "env_local", "box": [-0.05, -0.05, -1.05, 0.05, 0.05, -0.95]}
                ],
            }
        ],
    }
    config_path = tmp_path / "scene.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _request(sock, method, params=None, request_id="req"):
    send_json(sock, {"request_id": request_id, "method": method, "params": params or {}})
    return recv_json(sock)


def _assert_png_record(record):
    path = Path(record["path"])
    assert path.exists()
    assert record["byte_size"] == path.stat().st_size
    assert record["byte_size"] > 0
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        assert image.size == (record["width"], record["height"])


def _wait_ready(path: Path, proc: subprocess.Popen, timeout_s: float = 90.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server exited early with {proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        time.sleep(0.1)
    raise AssertionError("server did not write ready file")


def test_protocol_framing_round_trip():
    left, right = socket.socketpair()
    try:
        send_json(left, {"request_id": "a", "method": "session.handshake", "params": {}})
        assert recv_json(right) == {"request_id": "a", "method": "session.handshake", "params": {}}
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize(("requested_backend", "expected"), (("cpu", gs.cpu), ("cuda", gs.cuda)))
def test_live_session_initialization_honors_requested_volumetric_backend(
    monkeypatch, backend, requested_backend, expected
):
    del backend
    calls = []
    monkeypatch.setattr(gs, "_initialized", False)
    monkeypatch.setattr(gs, "init", lambda **kwargs: calls.append(kwargs))

    session = GenesisLiveSession.__new__(GenesisLiveSession)
    session._ensure_genesis_initialized(requested_backend, requires_surface_backend=False)

    assert calls == [{"backend": expected, "logging_level": "warning"}]


@pytest.mark.parametrize("backend", [None])
def test_live_session_rejects_initialized_volumetric_backend_mismatch(monkeypatch, backend):
    del backend
    monkeypatch.setattr(gs, "_initialized", True)
    monkeypatch.setattr(gs, "backend", gs.cpu)

    session = GenesisLiveSession.__new__(GenesisLiveSession)
    with pytest.raises(GenesisLiveError) as exc_info:
        session._ensure_genesis_initialized("cuda", requires_surface_backend=False)

    assert exc_info.value.code == "backend_mismatch"
    assert exc_info.value.details == {"current_backend": str(gs.cpu), "required_backend": "cuda"}


def test_live_session_lifecycle_and_box_action(tmp_path):
    output_dir = tmp_path / "outputs"
    session = GenesisLiveSession(
        scene_config_path=str(_write_scene_config(tmp_path)), start_paused=True, output_dir=str(output_dir)
    )

    handshake = session.dispatch("session.handshake", {})
    assert handshake["protocol"] == PROTOCOL
    assert "live_box_controller_actions" in handshake["capabilities"]
    assert ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY in handshake["capabilities"]
    assert session.status()["current_step"] == 0
    assert session.status()["paused"]
    assert session.scene.fem_solver.enable_floor is False

    geometry = session.dispatch("geometry.context.get", {"entity": "body"})
    assert geometry["representation"] == "volumetric"
    assert geometry["primitive_kind"] == "tetrahedron"
    assert geometry["vertex_count"] == 5
    assert geometry["element_count"] == 2
    assert geometry["max_extent"] == pytest.approx(2.0)

    resumed = session.dispatch("sim.resume", {"steps": 3})
    assert resumed["status"]["current_step"] == 3
    assert resumed["status"]["paused"]
    assert session.dispatch("sim.pause", {})["paused"]

    action = session.dispatch(
        "probe.apply",
        {
            "action": "box_ee_grasp_and_move",
            "entity": "body",
            "duration_steps": 12,
            "controllers": [
                {
                    "controller_id": "diag_box",
                    "aabb_box": {"frame": "env_local", "box": [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05]},
                    "distance_scale": 0.5,
                    "motion_axis": "+Y",
                }
            ],
        },
    )
    assert action["probe"]["selected_vertex_count"] == 1
    action_state = action["probe"]["controller_state"]
    assert action_state["speed"] == pytest.approx(0.6)
    assert action_state["motion_axis"] == "+Y"
    assert action_state["motion_active"] is True
    assert action_state["moved_distance"] == pytest.approx(0.0)
    assert action_state["displacement"] == pytest.approx([0.0, 0.0, 0.0])

    visual = session.dispatch("sim.resume", {"steps": 10, "diagnostic_visual": {"mode": "rgb_triptych"}})[
        "visual_telemetry"
    ]
    assert visual["rendered"]
    assert visual["count"] == 1
    assert visual["render_every_steps"] == 10
    assert visual["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert visual["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert visual["renderer"]["camera_model"] == "pinhole"
    assert visual["renderer"]["debug_camera"] is True
    assert visual["panel_order"] == ["top", "northeast", "southwest"]
    assert len(visual["views"]) == 3
    for record in visual["views"]:
        assert record["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
        assert record["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
        assert record["camera"]["model"] == "pinhole"
        assert record["camera"]["debug"] is True
    for record in visual["frame_metadata"]:
        _assert_png_record(record)
    live_overlay = next(
        record
        for record in visual["overlays"]
        if record.get("kind") == "live_box_controller" and record.get("controller_id") == "diag_box"
    )
    assert live_overlay["motion_active"] is True
    assert live_overlay["moved_distance"] > 0.0
    assert live_overlay["displacement"][1] == pytest.approx(live_overlay["moved_distance"])
    assert live_overlay["rendered_env_local_box"][1] > live_overlay["env_local_box"][1]
    live_marker = next(
        record
        for record in visual["debug_markers"]
        if record.get("kind") == "live_box_controller" and record.get("controller_id") == "diag_box"
    )
    assert live_marker["bounds"][0][1] == pytest.approx(live_overlay["rendered_env_local_box"][1])
    assert live_marker["bounds"][1][1] == pytest.approx(live_overlay["rendered_env_local_box"][4])
    assert session.dispatch("command.status", {})["last_rendered_frame_index"] == visual["stitched"]["frame_index"]
    assert visual["frames"][0]["stitched"]["frame_index"] == visual["stitched"]["frame_index"]

    fused = session.dispatch("observation.fused", {})
    assert fused["material"]["material_type"] == "Elastic"
    assert fused["material"]["heterogeneous"] is False

    release = session.dispatch("probe.apply", {"action": "probe_release", "controller_id": "diag_box"})
    assert not release["probe"]["controller_state"]["active"]
    assert session.dispatch("sim.reset", {})["current_step"] == 0


def test_live_session_forces_floor_off_and_allows_below_floor_downward_motion(tmp_path):
    config_path = _write_scene_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["sim_options"]["gravity"] = [0.0, 0.0, 0.0]
    config["fem_options"]["enable_floor"] = True
    config["entities"][0]["anchors"] = []
    config["entities"][0]["morph"]["pos"] = [0.0, 0.0, -2.0]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    session = GenesisLiveSession(scene_config_path=str(config_path), start_paused=True)
    entity = session.entities["body"]
    assert session.scene.fem_solver.enable_floor is False
    entity.set_velocity((0.0, 0.0, -0.5))
    initial_z = float(entity.get_state().pos[..., 2].mean())
    session.scene.step(update_visualizer=False)
    final_z = float(entity.get_state().pos[..., 2].mean())
    assert final_z < initial_z - 1.0e-5


def test_live_session_rejects_unsupported_visual_modes(tmp_path):
    session = GenesisLiveSession(scene_config_path=str(_write_scene_config(tmp_path)), start_paused=True)

    response = session.handle_request(
        {"request_id": "depth", "method": "sim.resume", "params": {"steps": 1, "diagnostic_visual": {"mode": "depth"}}}
    )

    assert response["status"] == "error"
    assert response["error"]["code"] == "unsupported_visual_mode"


def test_resume_reports_invalid_fem_state_before_visual_capture(monkeypatch, tmp_path):
    output_dir = tmp_path / "outputs"
    session = GenesisLiveSession(
        scene_config_path=str(_write_scene_config(tmp_path)), start_paused=True, output_dir=str(output_dir)
    )
    entity = session.entities["body"]
    bad_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [np.nan, 2.0, 3.0],
            [4.0, np.inf, 5.0],
            [-1.0, -2.0, -3.0],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(session.scene, "step", lambda: None)
    monkeypatch.setattr(entity, "get_state", lambda: SimpleNamespace(pos=bad_positions))

    response = session.handle_request(
        {
            "request_id": "bad",
            "method": "sim.resume",
            "params": {"steps": 1, "diagnostic_visual": {"mode": "rgb_triptych"}},
        }
    )

    assert response["status"] == "error"
    assert response["error"]["code"] == "invalid_simulation_state"
    details = response["error"]["details"]
    assert details["entity"] == "body"
    assert details["current_step"] == 1
    assert details["first_bad_step"] == 1
    assert details["checked_at"] == "after_step"
    assert details["shape"] == [5, 3]
    assert details["finite_vertex_count"] == 3
    assert details["nonfinite_vertex_count"] == 2
    assert details["finite_scalar_count"] == 13
    assert details["nonfinite_scalar_count"] == 2
    assert details["finite_min"] == pytest.approx(-3.0)
    assert details["finite_max"] == pytest.approx(5.0)
    assert details["first_nonfinite_vertex_index"] == 2
    status = session.status()
    assert status["paused"] is True
    assert status["running"] is False
    assert status["fatal_error"]["code"] == "invalid_simulation_state"
    assert status["last_rendered_frame_index"] is None
    assert not (output_dir / "png_rgb_triptych" / "frame_000001.png").exists()


def test_rgb_triptych_render_cadence_is_frame_not_step(tmp_path):
    output_dir = tmp_path / "outputs"
    session = GenesisLiveSession(
        scene_config_path=str(_write_scene_config(tmp_path)), start_paused=True, output_dir=str(output_dir)
    )

    visual = session.dispatch("sim.resume", {"steps": 20, "diagnostic_visual": {"mode": "rgb_triptych"}})[
        "visual_telemetry"
    ]

    assert visual["rendered"]
    assert visual["count"] == 2
    assert visual["render_every_steps"] == 10
    assert [frame["stitched"]["frame_index"] for frame in visual["frames"]] == [10, 20]
    assert session.dispatch("command.status", {})["current_step"] == 20
    assert session.dispatch("command.status", {})["last_rendered_frame_index"] == 20
    assert (output_dir / "png_rgb_triptych" / "frame_000010.png").is_file()
    assert (output_dir / "png_rgb_triptych" / "frame_000020.png").is_file()
    assert not (output_dir / "png_rgb_triptych" / "frame_000001.png").exists()


def test_live_server_subprocess_ready_handshake_status_and_close(tmp_path):
    config_path = _write_scene_config(tmp_path)
    ready_path = tmp_path / "ready.json"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "genesis.live.server",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ready-file",
            str(ready_path),
            "--scene-config",
            str(config_path),
            "--start-paused",
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _wait_ready(ready_path, proc)
        assert ready["protocol"] == PROTOCOL
        assert ready["server_pid"] == proc.pid
        assert ready["start_paused"] is True
        assert ready["scene_config_path"] == str(config_path)
        assert "live_box_controller_actions" in ready["capabilities"]
        assert ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY in ready["capabilities"]

        with socket.create_connection((ready["host"], ready["port"]), timeout=30.0) as sock:
            handshake = _request(sock, "session.handshake", request_id="hello")
            assert handshake["status"] == "ok"
            assert handshake["data"]["protocol"] == PROTOCOL
            assert handshake["data"]["capabilities"] == ready["capabilities"]

            status = _request(sock, "command.status", request_id="status")
            assert status["data"]["paused"]
            assert status["data"]["current_step"] == 0

            resumed = _request(sock, "sim.resume", {"steps": 2}, request_id="resume")
            assert resumed["data"]["status"]["current_step"] == 2
            assert resumed["data"]["status"]["paused"]

            closed = _request(sock, "session.close", request_id="close")
            assert closed["status"] == "ok"

        proc.wait(timeout=30)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)


def test_diagnostic_live_ready_and_handshake_hide_textured_overlay_capabilities(tmp_path):
    config_path = _write_scene_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["agentic_diagnostics"] = {"mode": "structured_live_diagnostic"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    ready_path = tmp_path / "diagnostic_ready.json"
    server = GenesisLiveServer(
        host="127.0.0.1",
        port=0,
        ready_file=str(ready_path),
        scene_config_path=str(config_path),
        start_paused=True,
    )

    server.write_ready()
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    handshake = server.session.handshake()
    forbidden = {
        "deformable_textured_visual_overlay",
        "visual_overlay_depth_normal_triptych_telemetry",
        "visual_overlay_vertex_trace",
    }

    assert forbidden.isdisjoint(ready["capabilities"])
    assert forbidden.isdisjoint(handshake["capabilities"])
    assert ready["capabilities"] == handshake["capabilities"]
    assert "part_segmentation_triptych_telemetry" in ready["capabilities"]
    assert ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY in ready["capabilities"]


def test_live_package_does_not_contain_removed_server_protocol_names():
    live_dir = Path(__file__).resolve().parents[1] / "genesis" / "live"
    forbidden = ("RealSimLiveServer", "realsim-live-v1")
    for path in live_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text
