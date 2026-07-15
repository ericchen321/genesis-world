from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import genesis as gs
import numpy as np

from genesis.engine.controllers.box_end_effector import apply_static_box_anchors
from genesis.utils.heterogeneous_materials import (
    SurfaceHeterogeneousMaterial,
    load_obj_triangle_faces,
    load_surface_heterogeneous_material,
    validate_surface_mesh_material_contract,
)

from .actions import ActionRegistry, apply_probe_action
from .capabilities import capability_report, surface_backend_status
from .protocol import PROTOCOL, GenesisLiveError, error_response, ok_response
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

    def _scene_requires_surface_backend(self, scene_config: dict[str, Any]) -> bool:
        for entity_cfg in scene_config.get("entities", []):
            if not isinstance(entity_cfg, dict):
                continue
            morph_cfg = entity_cfg.get("morph", {})
            material_cfg = entity_cfg.get("material", {})
            if isinstance(morph_cfg, dict) and morph_cfg.get("type") == "surface_mesh":
                return True
            if isinstance(material_cfg, dict) and material_cfg.get("type") in {"surface_shell", "cloth"}:
                return True
        return False

    def _scene_requested_backend(self, scene_config: dict[str, Any]) -> Literal["cpu", "cuda"]:
        backend = scene_config.get("backend", "cpu")
        if backend not in {"cpu", "cuda"}:
            raise GenesisLiveError("invalid_scene_config", "scene backend must be 'cpu' or 'cuda'")
        return backend

    def _validate_scene_config_before_build(self, scene_config: dict[str, Any]) -> None:
        for index, entity_cfg in enumerate(scene_config.get("entities", [])):
            if not isinstance(entity_cfg, dict):
                raise GenesisLiveError("invalid_scene_config", "entity entries must be JSON objects")
            morph_cfg = entity_cfg.get("morph", {})
            material_cfg = entity_cfg.get("material", {})
            if not isinstance(morph_cfg, dict) or not isinstance(material_cfg, dict):
                raise GenesisLiveError("invalid_scene_config", "entity morph and material entries must be JSON objects")
            morph_type = morph_cfg.get("type")
            material_type = material_cfg.get("type", "elastic")
            if morph_type == "surface_mesh":
                if material_type not in {"surface_shell", "cloth"}:
                    raise GenesisLiveError(
                        "invalid_scene_config",
                        "surface_mesh morph requires material.type='surface_shell' or 'cloth'",
                        details={"entity_index": index, "material_type": material_type},
                    )
                if "file" not in morph_cfg:
                    raise GenesisLiveError("invalid_scene_config", "surface_mesh morph requires file")
                faces = self._validate_surface_obj_faces(morph_cfg["file"], entity_index=index)
                heterogeneous = material_cfg.get("heterogeneous")
                if heterogeneous is not None:
                    material_file = self._surface_heterogeneous_material_file(
                        heterogeneous,
                        entity_index=index,
                        mesh_file=morph_cfg["file"],
                    )
                    self._validate_surface_heterogeneous_material_contract(
                        mesh_file=morph_cfg["file"],
                        material_file=material_file,
                        source_triangle_count=int(faces.shape[0]),
                        entity_index=index,
                    )
            elif morph_type == "tet_mesh":
                if material_type != "elastic":
                    raise GenesisLiveError(
                        "invalid_scene_config",
                        "tet_mesh morph requires material.type='elastic' for this feature",
                        details={"entity_index": index, "material_type": material_type},
                    )

    def _validate_surface_obj_faces(self, mesh_file: str | Path, *, entity_index: int) -> np.ndarray:
        path = Path(mesh_file)
        if path.suffix.lower() != ".obj":
            raise GenesisLiveError(
                "invalid_scene_config",
                "surface_mesh live path requires a source OBJ mesh with triangle faces",
                details={"entity_index": entity_index, "file": str(path)},
            )
        try:
            faces = load_obj_triangle_faces(path)
        except gs.GenesisException as exc:
            raise GenesisLiveError(
                "invalid_scene_config",
                "surface_mesh live path requires source OBJ triangle faces",
                details={"entity_index": entity_index, "file": str(path), "error": str(exc)},
            ) from exc
        if faces.shape[0] == 0:
            raise GenesisLiveError(
                "invalid_scene_config",
                "surface_mesh live path requires at least one source OBJ face",
                details={"entity_index": entity_index, "file": str(path)},
            )
        return faces

    def _surface_heterogeneous_material_file(
        self,
        heterogeneous: Any,
        *,
        entity_index: int,
        mesh_file: str | Path,
    ) -> str:
        details = {"entity_index": entity_index, "mesh_file": str(mesh_file)}
        if not isinstance(heterogeneous, dict):
            raise GenesisLiveError(
                "invalid_surface_heterogeneous_material",
                "surface material.heterogeneous must be an object",
                details=details,
            )
        allowed_keys = {"kind", "material_file"}
        if set(heterogeneous) != allowed_keys:
            raise GenesisLiveError(
                "invalid_surface_heterogeneous_material",
                "surface material.heterogeneous must contain exactly kind='surface_triangles' and material_file",
                details={**details, "keys": sorted(heterogeneous), "allowed_keys": sorted(allowed_keys)},
            )
        if heterogeneous["kind"] != "surface_triangles":
            raise GenesisLiveError(
                "invalid_surface_heterogeneous_material",
                "surface material.heterogeneous.kind must be 'surface_triangles'",
                details={**details, "kind": heterogeneous["kind"]},
            )
        material_file = heterogeneous["material_file"]
        if not isinstance(material_file, str) or not material_file:
            raise GenesisLiveError(
                "invalid_surface_heterogeneous_material",
                "surface material.heterogeneous.material_file must be a filesystem path string",
                details=details,
            )
        return material_file

    def _validate_surface_heterogeneous_material_contract(
        self,
        *,
        mesh_file: str | Path,
        material_file: str | Path,
        source_triangle_count: int,
        entity_index: int,
    ) -> None:
        try:
            material_data = load_surface_heterogeneous_material(
                SurfaceHeterogeneousMaterial(file=material_file),
                triangle_count=source_triangle_count,
            )
            validate_surface_mesh_material_contract(mesh_file, material_data)
        except gs.GenesisException as exc:
            raise GenesisLiveError(
                "invalid_surface_heterogeneous_material",
                "surface heterogeneous material does not match the source OBJ triangle contract",
                details={
                    "entity_index": entity_index,
                    "mesh_file": str(mesh_file),
                    "material_file": str(material_file),
                    "source_triangle_count": int(source_triangle_count),
                    "error": str(exc),
                },
            ) from exc

    def _ensure_genesis_initialized(self, backend: Literal["cpu", "cuda"], *, requires_surface_backend: bool) -> None:
        if requires_surface_backend:
            if backend != "cuda":
                raise GenesisLiveError(
                    "unsupported_surface_backend",
                    "surface shell diagnostics require scene backend='cuda'",
                    details={"requested_backend": backend},
                )
            status = surface_backend_status()
            if not status["available"]:
                raise GenesisLiveError(
                    "unsupported_surface_backend",
                    "surface shell diagnostics require uipc, CUDA, and IPCCoupler support",
                    details=status,
                )

        requested_backend = gs.cuda if backend == "cuda" else gs.cpu
        if gs._initialized:
            if gs.backend != requested_backend:
                error_code = "unsupported_surface_backend" if requires_surface_backend else "backend_mismatch"
                raise GenesisLiveError(
                    error_code,
                    "Genesis is already initialized on a different backend",
                    details={"current_backend": str(gs.backend), "required_backend": backend},
                )
            return
        gs.init(backend=requested_backend, logging_level="warning")

    def build_scene(self) -> None:
        self._validate_scene_config_before_build(self._scene_config)
        requires_surface_backend = self._scene_requires_surface_backend(self._scene_config)
        backend = self._scene_requested_backend(self._scene_config)
        self._ensure_genesis_initialized(backend, requires_surface_backend=requires_surface_backend)
        self.visual_telemetry.reset_triptych_cameras()
        if self.scene is not None:
            self.scene.destroy()
        self.controllers.clear()
        self.actions.clear()
        self.anchor_records.clear()

        sim_options = self._scene_config.get("sim_options", {})
        fem_options = {"enable_vertex_constraints": True}
        fem_options.update(self._scene_config.get("fem_options", {}))
        scene_kwargs = {
            "show_viewer": False,
            "sim_options": gs.options.SimOptions(**sim_options),
            "fem_options": gs.options.FEMOptions(**fem_options),
        }
        if requires_surface_backend:
            scene_kwargs["coupler_options"] = gs.options.IPCCouplerOptions(
                **self._scene_config.get("coupler_options", {})
            )
        self.scene = gs.Scene(**scene_kwargs)

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
                try:
                    self.anchor_records[name] = apply_static_box_anchors(entity, anchors, frame="env_local")
                except gs.GenesisException as exc:
                    raise GenesisLiveError(
                        "invalid_scene_config",
                        "static box anchor selection failed",
                        details={"entity": name, "error": str(exc)},
                    ) from exc
        self.current_step = 0
        self.paused = self.start_paused
        self.running = False
        self.last_frame_index = None
        self.heartbeat_timestamp = time.time()

    def _build_morph(self, morph_cfg: dict[str, Any]):
        morph_type = morph_cfg.get("type")
        if morph_type == "tet_mesh":
            if "file" not in morph_cfg:
                raise GenesisLiveError("invalid_scene_config", "tet_mesh morph requires file")
            kwargs = {"file": morph_cfg["file"]}
            if "scale" in morph_cfg:
                kwargs["scale"] = tuple(morph_cfg["scale"])
            if "pos" in morph_cfg:
                kwargs["pos"] = tuple(morph_cfg["pos"])
            return gs.morphs.TetMesh(**kwargs)
        if morph_type == "surface_mesh":
            if "file" not in morph_cfg:
                raise GenesisLiveError("invalid_scene_config", "surface_mesh morph requires file")
            for key in ("decimate", "convexify", "force_retet", "quality", "retet"):
                if bool(morph_cfg.get(key, False)):
                    raise GenesisLiveError(
                        "invalid_scene_config",
                        f"surface_mesh live path does not allow mesh rewriting option: {key}",
                    )
            kwargs = {"file": morph_cfg["file"], "convexify": False, "decimate": False}
            for key in ("scale", "pos", "euler", "quat"):
                if key in morph_cfg:
                    kwargs[key] = tuple(morph_cfg[key])
            if "file_meshes_are_zup" in morph_cfg:
                kwargs["file_meshes_are_zup"] = bool(morph_cfg["file_meshes_are_zup"])
            return gs.morphs.Mesh(**kwargs)
        raise GenesisLiveError(
            "invalid_scene_config",
            "only morph.type='tet_mesh' or morph.type='surface_mesh' is supported by this feature",
        )

    def _build_material(self, material_cfg: dict[str, Any]):
        material_type = material_cfg.get("type", "elastic")
        if material_type == "elastic":
            kwargs = {
                key: material_cfg[key]
                for key in ("E", "nu", "rho", "friction_mu")
                if key in material_cfg
            }
            heterogeneous = material_cfg.get("heterogeneous")
            if heterogeneous is not None:
                kwargs["heterogeneous"] = gs.materials.FEM.HeterogeneousMaterial(**heterogeneous)
            return gs.materials.FEM.Elastic(**kwargs)
        if material_type in {"surface_shell", "cloth"}:
            kwargs = {
                key: material_cfg[key]
                for key in ("E", "nu", "rho", "thickness", "bending_stiffness", "friction_mu", "contact_resistance")
                if key in material_cfg
            }
            heterogeneous = material_cfg.get("heterogeneous")
            if heterogeneous is not None:
                kwargs["heterogeneous"] = gs.materials.FEM.SurfaceHeterogeneousMaterial(
                    file=self._surface_heterogeneous_material_file(
                        heterogeneous,
                        entity_index=-1,
                        mesh_file="<material_build>",
                    )
                )
            return gs.materials.FEM.Cloth(**kwargs)
        raise GenesisLiveError(
            "invalid_scene_config",
            "only material.type='elastic', 'surface_shell', or 'cloth' is supported by this feature",
        )

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
        coupler_type = None
        if self.scene is not None and getattr(self.scene, "sim", None) is not None:
            coupler_type = self.scene.sim.coupler.__class__.__name__
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
            "backend": str(gs.backend) if gs._initialized else None,
            "coupler_type": coupler_type,
            "surface_scene": self._scene_requires_surface_backend(self._scene_config),
        }

    def backend_requirements(self) -> dict[str, Any]:
        return capability_report()["backend_requirements"]

    def handshake(self) -> dict[str, Any]:
        report = capability_report()
        return {
            "protocol": PROTOCOL,
            "session_id": self.session_id,
            "genesis_version": gs.__version__,
            "capabilities": report["capabilities"],
            "backend_requirements": report["backend_requirements"],
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
        steps = int(params.get("steps", params.get("duration_steps", 1)))
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
                    controller.advance_motion(steps=1)
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
