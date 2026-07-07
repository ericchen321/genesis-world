from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import genesis as gs
import numpy as np

from genesis.engine.controllers.box_end_effector import apply_static_box_anchors

from .actions import ActionRegistry, apply_probe_action
from .protocol import CAPABILITIES, PROTOCOL, GenesisLiveError, error_response, ok_response
from .snapshots import controller_snapshots, fused_observation, geometry_context
from .visual_telemetry import GENESIS_NATIVE_DEBUG_CAMERA_RENDERER, VisualTelemetry

DEFAULT_RENDER_EVERY_STEPS = 10


class GenesisLiveSession:
    def __init__(
        self,
        *,
        scene_config_path: str | None = None,
        start_paused: bool = True,
        output_dir: str | None = None,
    ):
        self.scene_config_path = scene_config_path
        self.start_paused = bool(start_paused)
        self.output_dir = output_dir
        self.session_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.actions = ActionRegistry()
        self.controllers = {}
        self.anchor_records = {}
        self.last_request_id = None
        self.last_frame_index = None
        self.fatal_error = None
        self.heartbeat_timestamp = time.time()
        self.running = False
        self.paused = bool(start_paused)
        self.current_step = 0
        self._scene_config = self._load_scene_config(scene_config_path)
        self.scene = None
        self.entities = {}
        self.visual_telemetry = VisualTelemetry(self._resolve_output_dir())
        self.build_scene()

    def _resolve_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        if self.scene_config_path is not None:
            return Path(self.scene_config_path).resolve().parent / "genesis_live_outputs" / self.session_id
        return Path.cwd() / "genesis_live_outputs" / self.session_id

    def _load_scene_config(self, path: str | None) -> dict[str, Any]:
        if path is None:
            return {"entities": []}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise GenesisLiveError("invalid_scene_config", "scene config must be a JSON object")
        return data

    def _ensure_genesis_initialized(self) -> None:
        if not gs._initialized:
            gs.init(backend=gs.cpu, logging_level="warning")

    def build_scene(self) -> None:
        self._ensure_genesis_initialized()
        self.visual_telemetry.reset_triptych_cameras()
        if self.scene is not None:
            self.scene.destroy()
        self.controllers.clear()
        self.actions.clear()
        self.anchor_records.clear()

        sim_options = self._scene_config.get("sim_options", {})
        fem_options = {"enable_vertex_constraints": True}
        fem_options.update(self._scene_config.get("fem_options", {}))
        self.scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(**sim_options),
            fem_options=gs.options.FEMOptions(**fem_options),
        )

        self.entities = {}
        pending_anchors = []
        for index, entity_cfg in enumerate(self._scene_config.get("entities", [])):
            if not isinstance(entity_cfg, dict):
                raise GenesisLiveError("invalid_scene_config", "entity entries must be JSON objects")
            name = str(entity_cfg.get("name") or f"entity_{index}")
            morph_cfg = dict(entity_cfg.get("morph", {}))
            material_cfg = dict(entity_cfg.get("material", {}))
            morph = self._build_morph(morph_cfg)
            material = self._build_material(material_cfg)
            entity = self.scene.add_entity(morph=morph, material=material, name=name)
            self.entities[name] = entity
            pending_anchors.append((entity, entity_cfg.get("anchors", [])))

        self.visual_telemetry.register_triptych_cameras(self)
        self.scene.build()
        for entity, anchors in pending_anchors:
            if anchors:
                name = next(name for name, candidate in self.entities.items() if candidate is entity)
                self.anchor_records[name] = apply_static_box_anchors(entity, anchors, frame="env_local")
        self.current_step = 0
        self.paused = self.start_paused
        self.running = False
        self.last_frame_index = None
        self.heartbeat_timestamp = time.time()

    def _build_morph(self, morph_cfg: dict[str, Any]):
        morph_type = morph_cfg.get("type")
        if morph_type != "tet_mesh":
            raise GenesisLiveError("invalid_scene_config", "only morph.type='tet_mesh' is supported by this feature")
        if "file" not in morph_cfg:
            raise GenesisLiveError("invalid_scene_config", "tet_mesh morph requires file")
        kwargs = {"file": morph_cfg["file"]}
        if "scale" in morph_cfg:
            kwargs["scale"] = tuple(morph_cfg["scale"])
        if "pos" in morph_cfg:
            kwargs["pos"] = tuple(morph_cfg["pos"])
        return gs.morphs.TetMesh(**kwargs)

    def _build_material(self, material_cfg: dict[str, Any]):
        material_type = material_cfg.get("type", "elastic")
        if material_type != "elastic":
            raise GenesisLiveError("invalid_scene_config", "only material.type='elastic' is supported by this feature")
        kwargs = {
            key: material_cfg[key]
            for key in ("E", "nu", "rho", "friction_mu")
            if key in material_cfg
        }
        heterogeneous = material_cfg.get("heterogeneous")
        if heterogeneous is not None:
            kwargs["heterogeneous"] = gs.materials.FEM.HeterogeneousMaterial(**heterogeneous)
        return gs.materials.FEM.Elastic(**kwargs)

    def default_entity(self):
        if not self.entities:
            raise GenesisLiveError("no_entity", "session has no diagnostic entity")
        name = next(iter(self.entities))
        return name, self.entities[name]

    def entity_by_name(self, name: str):
        entity = self.entities.get(name)
        if entity is None:
            raise GenesisLiveError("unknown_entity", f"unknown entity: {name}")
        return name, entity

    def status(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "session_id": self.session_id,
            "paused": bool(self.paused),
            "running": bool(self.running),
            "current_step": int(self.current_step),
            "last_completed_request_id": self.last_request_id,
            "last_rendered_frame_index": self.last_frame_index,
            "heartbeat_timestamp": self.heartbeat_timestamp,
            "fatal_error": self.fatal_error,
            "controller_count": len(self.controllers),
        }

    def handshake(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "session_id": self.session_id,
            "genesis_version": gs.__version__,
            "capabilities": list(CAPABILITIES),
            "status": self.status(),
        }

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id")
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(method, str):
            return error_response(request_id, GenesisLiveError("invalid_request", "request method must be a string"))
        if not isinstance(params, dict):
            return error_response(request_id, GenesisLiveError("invalid_request", "request params must be an object"))

        with self.lock:
            try:
                data = self.dispatch(method, params)
                self.last_request_id = request_id
                self.heartbeat_timestamp = time.time()
                return ok_response(request_id, data)
            except GenesisLiveError as exc:
                self.heartbeat_timestamp = time.time()
                return error_response(request_id, exc)
            except Exception as exc:
                self.fatal_error = str(exc)
                self.heartbeat_timestamp = time.time()
                return error_response(request_id, GenesisLiveError("internal_error", str(exc)))

    def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "session.handshake":
            return self.handshake()
        if method == "session.close":
            return {"closed": True, "status": self.status()}
        if method in {"command.status", "command.heartbeat"}:
            self.heartbeat_timestamp = time.time()
            return self.status()
        if method == "sim.pause":
            self.paused = True
            self.running = False
            return self.status()
        if method == "sim.reset":
            self.build_scene()
            return self.status()
        if method == "sim.resume":
            return self.resume(params)
        if method == "geometry.context.get":
            entity_name, entity = self.entity_by_name(str(params["entity"])) if "entity" in params else self.default_entity()
            return geometry_context(entity, entity_name=entity_name)
        if method == "probe.action.register":
            return self.actions.register(params)
        if method == "probe.apply":
            result = apply_probe_action(self, params)
            return {"probe": result, "status": self.status()}
        if method == "observation.fused":
            return fused_observation(self)
        raise GenesisLiveError("unknown_method", f"unknown live method: {method}")

    def resume(self, params: dict[str, Any]) -> dict[str, Any]:
        steps = int(params.get("steps", params.get("duration_frames", 1)))
        if steps <= 0:
            raise GenesisLiveError("invalid_resume", "sim.resume steps must be positive")
        visual = params.get("diagnostic_visual")
        self._validate_visual_request(visual)
        render_every_steps = self._visual_render_every_steps(visual)
        visual_frames = []

        self.paused = False
        self.running = True
        try:
            for local_step_index in range(steps):
                for controller in list(self.controllers.values()):
                    controller.advance_motion(frames=1)
                self.scene.step()
                self.current_step += 1
                self._validate_fem_state(checked_at="after_step")
                if render_every_steps is not None and (local_step_index + 1) % render_every_steps == 0:
                    visual_frames.append(
                        self._capture_visual_request(
                            visual,
                            frame_index=int(self.current_step),
                            frame_sequence_index=len(visual_frames),
                            render_every_steps=render_every_steps,
                        )
                    )
            visual_result = self._visual_result_from_frames(
                visual,
                visual_frames,
                render_every_steps=render_every_steps,
                steps_requested=steps,
            )
        except GenesisLiveError as exc:
            if exc.code == "invalid_simulation_state":
                self.fatal_error = exc.to_dict()
            raise
        finally:
            self.running = False
            self.paused = True
        return {"steps": steps, "status": self.status(), "visual_telemetry": visual_result}

    def _entity_positions_for_validation(self, entity_name: str, entity, *, checked_at: str) -> np.ndarray:
        from genesis.utils.misc import tensor_to_array

        if getattr(entity, "active", False):
            positions = tensor_to_array(entity.get_state().pos)
        else:
            positions = tensor_to_array(entity.init_positions)

        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim == 3:
            positions = positions[0]
        if positions.ndim != 2 or positions.shape[1] != 3:
            details = {
                "entity": entity_name,
                "current_step": int(self.current_step),
                "first_bad_step": int(self.current_step),
                "checked_at": checked_at,
                "shape": [int(dim) for dim in positions.shape],
            }
            raise GenesisLiveError(
                "invalid_simulation_state",
                "FEM entity positions have invalid shape during sim.resume",
                details=details,
            )
        return positions

    def _invalid_fem_state_details(self, entity_name: str, positions: np.ndarray, *, checked_at: str) -> dict[str, Any]:
        finite_mask = np.isfinite(positions)
        finite_vertex_mask = np.all(finite_mask, axis=1)
        nonfinite_vertex_indices = np.flatnonzero(~finite_vertex_mask)
        finite_values = positions[finite_mask]
        return {
            "entity": entity_name,
            "current_step": int(self.current_step),
            "first_bad_step": int(self.current_step),
            "checked_at": checked_at,
            "shape": [int(dim) for dim in positions.shape],
            "finite_vertex_count": int(np.count_nonzero(finite_vertex_mask)),
            "nonfinite_vertex_count": int(np.count_nonzero(~finite_vertex_mask)),
            "finite_scalar_count": int(np.count_nonzero(finite_mask)),
            "nonfinite_scalar_count": int(positions.size - np.count_nonzero(finite_mask)),
            "finite_min": float(np.min(finite_values)) if finite_values.size else None,
            "finite_max": float(np.max(finite_values)) if finite_values.size else None,
            "first_nonfinite_vertex_index": int(nonfinite_vertex_indices[0])
            if nonfinite_vertex_indices.size
            else None,
        }

    def _validate_fem_state(self, *, checked_at: str) -> None:
        for entity_name, entity in self.entities.items():
            positions = self._entity_positions_for_validation(entity_name, entity, checked_at=checked_at)
            if np.all(np.isfinite(positions)):
                continue
            raise GenesisLiveError(
                "invalid_simulation_state",
                "FEM entity positions became non-finite during sim.resume",
                details=self._invalid_fem_state_details(entity_name, positions, checked_at=checked_at),
            )

    def _validate_visual_request(self, visual):
        if visual is None:
            return
        if not isinstance(visual, dict):
            raise GenesisLiveError("invalid_visual_request", "diagnostic_visual must be an object")
        mode = visual.get("mode", "rgb_triptych")
        if mode in {"depth", "depth_triptych", "von_mises", "von_mises_triptych"}:
            raise GenesisLiveError("unsupported_visual_mode", f"{mode} is not supported by this migration feature")
        if mode != "rgb_triptych":
            raise GenesisLiveError("unsupported_visual_mode", f"unsupported diagnostic visual mode: {mode}")
        self._visual_render_every_steps(visual)

    def _visual_render_every_steps(self, visual) -> int | None:
        if visual is None:
            return None
        render_every_steps = visual.get("render_every_steps", visual.get("capture_every_n_steps", DEFAULT_RENDER_EVERY_STEPS))
        if isinstance(render_every_steps, bool) or not isinstance(render_every_steps, int) or render_every_steps <= 0:
            raise GenesisLiveError("invalid_visual_request", "diagnostic_visual.render_every_steps must be a positive integer")
        return int(render_every_steps)

    def _capture_visual_request(
        self,
        visual,
        *,
        frame_index: int,
        frame_sequence_index: int,
        render_every_steps: int,
    ):
        if visual is None:
            return {"requested": False}
        self._validate_fem_state(checked_at="before_visual_capture")
        metadata = self.visual_telemetry.capture_rgb_triptych(self, frame_index=frame_index)
        metadata["frame_sequence_index"] = int(frame_sequence_index)
        metadata["render_every_steps"] = int(render_every_steps)
        self.last_frame_index = int(metadata["stitched"]["frame_index"])
        return metadata

    def _visual_result_from_frames(
        self,
        visual,
        frames: list[dict[str, Any]],
        *,
        render_every_steps: int | None,
        steps_requested: int,
    ) -> dict[str, Any]:
        if visual is None:
            return {"requested": False}
        if not frames:
            return {
                "requested": True,
                "mode": "rgb_triptych",
                "rendered": False,
                "render_backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                "renderer": {
                    "backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                    "mode": "rgb_triptych",
                    "debug_camera": True,
                    "reason": "no RGB triptych frame boundary reached during this resume",
                },
                "render_every_steps": int(render_every_steps or DEFAULT_RENDER_EVERY_STEPS),
                "steps_requested": int(steps_requested),
                "frames": [],
                "frame_metadata": [],
                "count": 0,
                "reason": "no RGB triptych frame boundary reached during this resume",
            }
        result = dict(frames[-1])
        result["frames"] = frames
        result["count"] = len(frames)
        result["render_every_steps"] = int(render_every_steps or DEFAULT_RENDER_EVERY_STEPS)
        result["steps_requested"] = int(steps_requested)
        return result
