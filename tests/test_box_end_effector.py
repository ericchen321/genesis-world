import igl
import numpy as np
import pytest

import genesis as gs
from genesis.engine.controllers.box_end_effector import BoxEndEffectorController, apply_static_box_anchors
from genesis.utils.spatial_selection import aabb_to_env_local, select_vertices_in_aabb
from genesis.utils.misc import tensor_to_array


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


def _write_tet_mesh(path):
    igl.writeMESH(str(path), _TWO_TET_VERTS, _TWO_TETS, np.empty((0, 3), dtype=np.int64))


def _build_two_tet_scene(tmp_path):
    mesh_path = tmp_path / "two_tets.mesh"
    _write_tet_mesh(mesh_path)

    scene = gs.Scene(
        show_viewer=False,
        fem_options=gs.options.FEMOptions(enable_vertex_constraints=True),
    )
    entity = scene.add_entity(
        morph=gs.morphs.TetMesh(file=str(mesh_path)),
        material=gs.materials.FEM.Elastic(),
    )
    scene.build()
    return scene, entity


def _constraint_flags(scene, entity):
    flags = scene.fem_solver.vertex_constraints.is_constrained.to_numpy()
    return flags[entity.v_start : entity.v_start + entity.n_vertices, 0]


def _constraint_target(scene, entity, vertex_idx):
    targets = scene.fem_solver.vertex_constraints.target_pos.to_numpy()
    return targets[entity.v_start + vertex_idx, 0]


def test_aabb_selection_supports_explicit_frames():
    positions = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [1.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(select_vertices_in_aabb(positions, [1, 2, 3, 1, 2, 3]), np.array([0]))
    np.testing.assert_array_equal(
        select_vertices_in_aabb(positions, [0, 0, 0, 0, 0, 0], frame="object_local", object_to_env=[1, 2, 3]),
        np.array([0]),
    )
    np.testing.assert_array_equal(
        select_vertices_in_aabb(positions, [2, 3, 4, 2, 3, 4], frame="world", world_to_env=[-1, -1, -1]),
        np.array([0]),
    )

    object_to_env = np.eye(4, dtype=np.float32)
    object_to_env[:3, 3] = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        aabb_to_env_local([0, 0, 0, 0, 0, 0], frame="object_local", object_to_env=object_to_env),
        np.array([1, 2, 3, 1, 2, 3], dtype=np.float32),
    )


def test_aabb_selection_rejects_bad_box_and_frame():
    positions = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(gs.GenesisException, match="shape"):
        select_vertices_in_aabb(positions, [0, 0, 0])
    with pytest.raises(gs.GenesisException, match="Unsupported"):
        select_vertices_in_aabb(positions, [0, 0, 0, 1, 1, 1], frame="camera")


def test_static_box_anchor_constrains_expected_vertices(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)

    records = apply_static_box_anchors(
        entity,
        [{"anchor_id": "origin_pin", "frame": "env_local", "box": [-0.05, -0.05, -0.05, 0.05, 0.05, 0.05]}],
    )

    assert records[0].selected_vertex_count == 1
    np.testing.assert_array_equal(records[0].selected_vertices, np.array([0], dtype=gs.np_int))
    flags = _constraint_flags(scene, entity)
    assert flags[0]
    assert not flags[1]


def test_box_ee_grasp_selects_current_positions(tmp_path):
    _scene, entity = _build_two_tet_scene(tmp_path)
    current = tensor_to_array(entity.get_state().pos[0])
    moved = current + np.array([10.0, 0.0, 0.0], dtype=np.float32)
    entity.set_position(moved)

    controller = BoxEndEffectorController(entity)
    state = controller.grasp([9.95, -0.05, -0.05, 10.05, 0.05, 0.05])

    np.testing.assert_array_equal(state.selected_vertices, np.array([0], dtype=gs.np_int))
    np.testing.assert_allclose(state.target_positions[0], moved[0], atol=1e-6)


def test_box_ee_move_updates_constraint_targets_deterministically(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)
    controller.grasp([-0.05, -0.05, 0.95, 0.05, 0.05, 1.05])

    state = controller.move_positive_y(distance_scale=0.5, duration_frames=12, speed=0.6)

    np.testing.assert_array_equal(state.selected_vertices, np.array([3], dtype=gs.np_int))
    np.testing.assert_allclose(state.displacement, np.array([0.0, 0.05, 0.0], dtype=gs.np_float), atol=1e-6)
    np.testing.assert_allclose(_constraint_target(scene, entity, 3), np.array([0.0, 0.05, 1.0]), atol=1e-6)
    assert state.duration_frames == 12
    assert state.speed == pytest.approx(0.6)
    assert state.distance == pytest.approx(0.05)
    assert state.moved_distance == pytest.approx(0.05)
    assert state.estimated_motion_frames == 9
    assert not state.motion_active


def test_box_ee_motion_uses_speed_and_duration_frames(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)
    controller.grasp([-0.05, -0.05, -0.05, 1.05, 0.05, 0.05])

    state = controller.move_positive_y(distance_scale=0.5, duration_frames=2, speed=0.5)

    expected_distance = 0.55
    expected_moved = 0.5 * scene.dt * 2
    assert state.distance == pytest.approx(expected_distance)
    assert state.moved_distance == pytest.approx(expected_moved)
    assert state.estimated_motion_frames == 110
    assert state.motion_active
    np.testing.assert_allclose(_constraint_target(scene, entity, 0), np.array([0.0, expected_moved, 0.0]), atol=1e-6)
    np.testing.assert_allclose(_constraint_target(scene, entity, 1), np.array([1.0, expected_moved, 0.0]), atol=1e-6)

    state = controller.advance_motion(frames=200)

    assert state.moved_distance == pytest.approx(expected_distance)
    assert not state.motion_active
    np.testing.assert_allclose(_constraint_target(scene, entity, 1), np.array([1.0, expected_distance, 0.0]), atol=1e-6)


def test_box_ee_release_preserves_unrelated_static_anchor(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)
    apply_static_box_anchors(
        entity,
        [{"anchor_id": "bottom_pin", "frame": "env_local", "box": [-0.05, -0.05, -1.05, 0.05, 0.05, -0.95]}],
    )
    controller = BoxEndEffectorController(entity)
    controller.grasp([-0.05, -0.05, 0.95, 0.05, 0.05, 1.05])

    controller.release()

    flags = _constraint_flags(scene, entity)
    assert not flags[3]
    assert flags[4]


def test_box_ee_release_restores_overlapping_static_anchor(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)
    apply_static_box_anchors(
        entity,
        [{"anchor_id": "top_pin", "frame": "env_local", "box": [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05]}],
    )
    original_target = _constraint_target(scene, entity, 3).copy()
    controller = BoxEndEffectorController(entity)
    controller.grasp([-0.05, -0.05, 0.95, 0.05, 0.05, 1.05])
    controller.move_positive_y(distance_scale=0.5, duration_frames=12)

    controller.release()

    flags = _constraint_flags(scene, entity)
    assert flags[3]
    np.testing.assert_allclose(_constraint_target(scene, entity, 3), original_target, atol=1e-6)


def test_box_ee_empty_selection_fails_unless_optional(tmp_path):
    _scene, entity = _build_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)

    with pytest.raises(gs.GenesisException, match="selected no FEM vertices"):
        controller.grasp([10, 10, 10, 11, 11, 11])

    state = controller.grasp([10, 10, 10, 11, 11, 11], optional=True)
    assert state.selected_vertex_count == 0
    assert not state.active


def test_box_ee_rejects_out_of_range_distance_scale(tmp_path):
    _scene, entity = _build_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)
    controller.grasp([-0.05, -0.05, 0.95, 0.05, 0.05, 1.05])

    with pytest.raises(gs.GenesisException, match="distance_scale"):
        controller.move_positive_y(distance_scale=1.5, duration_frames=1)
    with pytest.raises(gs.GenesisException, match="distance_scale"):
        controller.move_positive_y(distance_scale=0.0, duration_frames=1)
