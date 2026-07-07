from __future__ import annotations

from typing import Any

from genesis.engine.controllers.box_end_effector import BoxEndEffectorController

from .protocol import GenesisLiveError


class ActionRegistry:
    def __init__(self):
        self._actions: dict[str, dict[str, Any]] = {}

    def register(self, params: dict[str, Any]) -> dict[str, Any]:
        action_id = str(params.get("action_id") or params.get("probe_id") or f"action_{len(self._actions):04d}")
        spec = dict(params)
        spec["action_id"] = action_id
        self._actions[action_id] = spec
        return {"action_id": action_id, "registered": True, "action": spec.get("action")}

    def get(self, action_id: str) -> dict[str, Any] | None:
        return self._actions.get(action_id)

    def clear(self) -> None:
        self._actions.clear()


def _controller_spec(params: dict[str, Any]) -> dict[str, Any]:
    controllers = params.get("controllers")
    if not isinstance(controllers, list) or len(controllers) != 1 or not isinstance(controllers[0], dict):
        raise GenesisLiveError("invalid_controller_request", "box action requires exactly one controller object")
    return controllers[0]


def _entity_for_action(session, params: dict[str, Any]):
    entity_name = params.get("entity") or params.get("entity_name")
    if entity_name is None:
        return session.default_entity()
    return session.entity_by_name(str(entity_name))


def apply_probe_action(session, params: dict[str, Any]) -> dict[str, Any]:
    action = params.get("action")
    if action is None and params.get("action_id"):
        registered = session.actions.get(str(params["action_id"]))
        if registered is None:
            raise GenesisLiveError("unknown_action", f"unknown registered action: {params['action_id']}")
        merged = dict(registered)
        merged.update(params)
        params = merged
        action = params.get("action")

    if action == "box_ee_grasp_and_move":
        entity_name, entity = _entity_for_action(session, params)
        controller_spec = _controller_spec(params)
        controller_id = str(controller_spec.get("controller_id") or params.get("controller_id") or "box_ee_0")
        aabb_box = controller_spec.get("aabb_box")
        if aabb_box is None:
            raise GenesisLiveError("invalid_controller_request", "box controller requires aabb_box")
        distance_scale = controller_spec.get("distance_scale")
        if distance_scale is None:
            raise GenesisLiveError("invalid_controller_request", "box controller requires distance_scale")
        duration_frames = int(params.get("duration_frames", controller_spec.get("duration_frames", 1)))

        controller = BoxEndEffectorController(entity, controller_id=controller_id)
        frame = aabb_box.get("frame", "env_local") if isinstance(aabb_box, dict) else "env_local"
        state = controller.grasp_and_move_positive_y(
            aabb_box,
            frame=frame,
            distance_scale=float(distance_scale),
            duration_frames=duration_frames,
        )
        session.controllers[controller_id] = controller
        return {
            "action": action,
            "entity": entity_name,
            "controller_id": controller_id,
            "controller_state": state.to_dict(),
            "selected_vertex_count": state.selected_vertex_count,
            "selected_vertices": state.selected_vertices.tolist(),
        }

    if action == "probe_release":
        controller_id = str(params.get("controller_id") or params.get("probe_id") or "box_ee_0")
        if controller_id not in session.controllers and len(session.controllers) == 1:
            controller_id = next(iter(session.controllers))
        controller = session.controllers.get(controller_id)
        if controller is None:
            raise GenesisLiveError("unknown_controller", f"unknown controller: {controller_id}")
        state = controller.release()
        session.controllers.pop(controller_id, None)
        return {"action": action, "controller_id": controller_id, "controller_state": state.to_dict()}

    raise GenesisLiveError("unsupported_action", f"unsupported probe action: {action}")
