from __future__ import annotations

from typing import Any

import genesis as gs

from genesis.engine.controllers.box_end_effector import BoxEndEffectorController

from .protocol import GenesisLiveError


BOX_EE_NATIVE_FEM_COMPLIANT_STIFFNESS = 0.095
BOX_EE_IPC_SURFACE_COMPLIANT_STRENGTH_RATE = 100.0
BOX_EE_COMPLIANCE_OVERRIDE_KEYS = frozenset(
    {
        "stiffness",
        "controller_stiffness",
        "strength",
        "strength_rate",
        "constraint_strength",
        "soft_constraint",
        "is_soft_constraint",
    }
)


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


def _reject_compliance_override_keys(params: dict[str, Any], controller_spec: dict[str, Any]) -> None:
    offenders = sorted(
        {f"params.{key}" for key in BOX_EE_COMPLIANCE_OVERRIDE_KEYS if key in params}
        | {f"controllers[0].{key}" for key in BOX_EE_COMPLIANCE_OVERRIDE_KEYS if key in controller_spec}
    )
    if offenders:
        raise GenesisLiveError(
            "invalid_controller_request",
            "BoxEE compliance controls are backend-owned; remove override field(s): " + ", ".join(offenders),
        )


def _box_ee_compliant_policy(entity) -> tuple[bool, float]:
    from genesis.engine.couplers import IPCCoupler
    from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial

    if isinstance(entity.sim.coupler, IPCCoupler) and isinstance(entity.material, ClothMaterial):
        # Existing FEMEntity API transports IPC surface strength-rate through the stiffness keyword.
        return True, BOX_EE_IPC_SURFACE_COMPLIANT_STRENGTH_RATE
    return True, BOX_EE_NATIVE_FEM_COMPLIANT_STIFFNESS


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
        _reject_compliance_override_keys(params, controller_spec)
        controller_id = str(controller_spec.get("controller_id") or params.get("controller_id") or "box_ee_0")
        aabb_box = controller_spec.get("aabb_box")
        if aabb_box is None:
            raise GenesisLiveError("invalid_controller_request", "box controller requires aabb_box")
        distance_scale = controller_spec.get("distance_scale")
        if distance_scale is None:
            raise GenesisLiveError("invalid_controller_request", "box controller requires distance_scale")
        duration_steps = int(params.get("duration_steps", controller_spec.get("duration_steps", 1)))
        box_tolerance = None
        if isinstance(aabb_box, dict):
            box_tolerance = aabb_box.get("selection_tolerance", aabb_box.get("tolerance", aabb_box.get("atol")))
        selection_tolerance = controller_spec.get(
            "selection_tolerance",
            controller_spec.get(
                "tolerance",
                controller_spec.get("atol", params.get("selection_tolerance", params.get("atol", box_tolerance))),
            ),
        )

        controller = BoxEndEffectorController(entity, controller_id=controller_id)
        frame = aabb_box.get("frame", "env_local") if isinstance(aabb_box, dict) else "env_local"
        is_soft_constraint, compliance_value = _box_ee_compliant_policy(entity)
        try:
            state = controller.grasp_and_move_positive_y(
                aabb_box,
                frame=frame,
                distance_scale=float(distance_scale),
                duration_steps=duration_steps,
                speed=float(controller_spec.get("speed", params.get("speed", 0.6))),
                max_distance_scale=float(
                    controller_spec.get("max_distance_scale", params.get("max_distance_scale", 1.0))
                ),
                is_soft_constraint=is_soft_constraint,
                stiffness=compliance_value,
                selection_tolerance=None if selection_tolerance is None else float(selection_tolerance),
            )
        except gs.GenesisException as exc:
            message = str(exc)
            code = "probe_selection_failed" if "selected no FEM vertices" in message else "probe_action_failed"
            raise GenesisLiveError(
                code,
                message,
                details={"action": action, "entity": entity_name, "controller_id": controller_id},
            ) from exc
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
