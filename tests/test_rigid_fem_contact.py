from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pytest
import genesis as gs
from genesis.engine.couplers.sap_coupler import (
    _build_rigid_fem_face_whitelist,
    _rigid_fem_contact_sort_order,
)


def _receipt(enabled=True):
    entry = gs.RigidFEMWhitelistEntry(
        rigid_entity_idx=1,
        rigid_entity_name="robot",
        collision_enabled=enabled,
        requested_link_names=None,
        resolved_link_indices=(2,) if enabled else (),
        resolved_link_names=("finger",) if enabled else (),
        resolved_geom_indices=(3,) if enabled else (),
        enabled_face_count=1 if enabled else 0,
        total_face_count=1,
    )
    return gs.RigidFEMWhitelistReceipt(
        entries=(entry,),
        enabled_face_count=1 if enabled else 0,
        total_face_count=1,
        face_enabled_by_index={0: enabled},
    )


def _batch(**overrides):
    payload = dict(
        env_idx=np.array([0], dtype=np.int64),
        rigid_entity_idx=np.array([1], dtype=np.int64),
        rigid_link_idx=np.array([2], dtype=np.int64),
        rigid_geom_idx=np.array([3], dtype=np.int64),
        fem_entity_idx=np.array([4], dtype=np.int64),
        fem_element_idx_local=np.array([5], dtype=np.int64),
        rigid_entity_names=("robot",),
        rigid_link_names=("finger",),
        fem_entity_names=("soft",),
        point_m=np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        normal_world=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        signed_gap_m=np.array([-0.01], dtype=np.float64),
        penetration_m=np.array([0.01], dtype=np.float64),
        normal_impulse_ns=np.array([0.5], dtype=np.float64),
        tangential_impulse_world_ns=np.array([[0.1, 0.2, 0.0]], dtype=np.float64),
        relative_tangential_velocity_world_mps=np.array([[0.01, 0.02, 0.0]], dtype=np.float64),
        modes=(gs.RigidFEMContactMode.STICK,),
        completed_scene_step=7,
        completed_substep=15,
        dt_s=0.001,
        whitelist_receipt=_receipt(),
    )
    payload.update(overrides)
    return gs.RigidFEMContactBatch(**payload)


def test_public_batch_is_strict_fresh_and_immutable():
    source = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    batch = _batch(point_m=source)
    assert len(batch) == 1
    assert batch.point_m.dtype == np.float64
    assert batch.env_idx.dtype == np.int64
    assert batch.modes == (gs.RigidFEMContactMode.STICK,)
    assert batch.whitelist_receipt.face_enabled_by_index == {0: True}
    assert not batch.point_m.flags.writeable
    assert not batch.env_idx.flags.writeable
    source[:] = 99.0
    assert np.array_equal(batch.point_m, [[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError):
        batch.point_m[0, 0] = 1.0
    with pytest.raises(TypeError):
        batch.whitelist_receipt.face_enabled_by_index[0] = False


def test_public_build_ownership_receipt_is_available_before_step():
    ownership = gs.RigidFEMContactOwnershipReceipt(
        whitelist_receipt=_receipt(),
        rigid_fem_contact_enabled=True,
        floor_tet_contact_enabled=True,
        floor_height_m=0.0,
    )
    scene = object.__new__(gs.Scene)
    scene._is_built = True
    scene._sim = SimpleNamespace(
        _coupler=SimpleNamespace(get_rigid_fem_contact_ownership=lambda: ownership),
        destroy=lambda: None,
    )
    assert gs.Scene.get_rigid_fem_contact_ownership(scene) is ownership
    assert ownership.floor_tet_contact_enabled
    assert ownership.whitelist_receipt.enabled_face_count == 1


@pytest.mark.parametrize(
    "mode,normal_impulse,tangential_impulse",
    [
        (gs.RigidFEMContactMode.STICK, 0.5, (0.1, 0.2, 0.0)),
        (gs.RigidFEMContactMode.SLIDE, 0.5, (0.1, 0.2, 0.0)),
        (gs.RigidFEMContactMode.NO_CONTACT, 0.0, (0.0, 0.0, 0.0)),
    ],
)
def test_public_contact_modes_have_exact_impulse_semantics(mode, normal_impulse, tangential_impulse):
    batch = _batch(
        modes=(mode,),
        normal_impulse_ns=np.array([normal_impulse], dtype=np.float64),
        tangential_impulse_world_ns=np.array([tangential_impulse], dtype=np.float64),
    )
    assert batch.modes == (mode,)


def _fake_solver(requested=("left",), enabled=True):
    def geom(index, face_start, n_faces):
        return SimpleNamespace(idx=index, face_start=face_start, face_end=face_start + n_faces, n_faces=n_faces)

    left = SimpleNamespace(name="left", idx=2, geoms=(geom(4, 0, 2),))
    right = SimpleNamespace(name="right", idx=3, geoms=(geom(5, 2, 3),))
    material = SimpleNamespace(coup_collision_links=requested, enable_coup_collision=enabled)
    entity = SimpleNamespace(
        idx=1,
        name="robot",
        links=(left, right),
        material=material,
        n_faces=5,
    )
    return SimpleNamespace(n_faces=5, entities=(entity,))


def _write_two_finger_urdf(tmp_path):
    path = tmp_path / "two_finger.urdf"
    path.write_text(
        """<robot name="two_finger">
  <link name="base_link"><inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
  <link name="g2_left_link"><inertial><mass value="0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial><collision><geometry><box size="0.02 0.02 0.02"/></geometry></collision><visual><geometry><box size="0.02 0.02 0.02"/></geometry></visual></link>
  <link name="g2_right_link"><inertial><mass value="0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial><collision><geometry><box size="0.02 0.02 0.02"/></geometry></collision><visual><geometry><box size="0.02 0.02 0.02"/></geometry></visual></link>
  <joint name="g2_left_joint" type="prismatic"><parent link="base_link"/><child link="g2_left_link"/><origin xyz="0.02 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper="0.02" effort="20" velocity="1"/></joint>
  <joint name="g2_right_joint" type="prismatic"><parent link="base_link"/><child link="g2_right_link"/><origin xyz="-0.02 0 0" rpy="0 0 0"/><axis xyz="-1 0 0"/><limit lower="0" upper="0.02" effort="20" velocity="1"/></joint>
</robot>
""",
        encoding="utf-8",
    )
    return path


def test_build_whitelist_resolves_positive_links_without_mutating_geoms():
    solver = _fake_solver()
    mask, receipt = _build_rigid_fem_face_whitelist(solver)
    assert np.array_equal(mask, [True, True, False, False, False])
    assert receipt.entries[0].requested_link_names == ("left",)
    assert receipt.entries[0].resolved_link_indices == (2,)
    assert receipt.entries[0].resolved_geom_indices == (4,)
    assert receipt.enabled_face_count == 2
    assert solver.entities[0].links[1].geoms[0].n_faces == 3


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_disabled_ordinary_collider_preserves_named_sap_finger_whitelist(tmp_path, show_viewer, precision):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=2, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True, enable_qualification_safety_extrema=True),
        coupler_options=gs.options.SAPCouplerOptions(
            enable_rigid_fem_contact=True,
            fem_floor_contact_type="tet",
            rigid_floor_contact_type="none",
            rigid_rigid_contact_type="none",
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(file=str(_write_two_finger_urdf(tmp_path)), fixed=True, merge_fixed_links=False),
        material=gs.materials.Rigid(
            enable_coup_collision=True,
            coup_collision_links=("g2_left_link", "g2_right_link"),
        ),
        name="two_finger_robot",
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.0), radius=0.01),
        material=gs.materials.FEM.Elastic(model="linear_corotated"),
        name="soft",
    )
    scene.build()

    rigid_solver = scene.sim.rigid_solver
    assert rigid_solver.collider._ordinary_collision_enabled is False
    assert rigid_solver.collider._sdf is None
    assert rigid_solver.collider._gjk is None
    assert rigid_solver.collider._support_field is None
    assert robot.get_link(name="g2_left_link").n_geoms > 0
    assert robot.get_link(name="g2_right_link").n_geoms > 0
    with pytest.raises(gs.GenesisException, match="enable_collision=False"):
        robot.detect_collision()

    scene.step(update_visualizer=False)
    health = scene.get_last_completed_solver_health()
    assert len(health.substeps) == 2
    assert all(substep.fem_safety_extrema is not None for substep in health.substeps)
    batch = scene.get_rigid_fem_contacts()
    entry = next(entry for entry in batch.whitelist_receipt.entries if entry.rigid_entity_name == "two_finger_robot")
    assert entry.resolved_link_names == ("g2_left_link", "g2_right_link")
    assert entry.enabled_face_count > 0


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_robot_disabled_scene_query_is_ready_and_empty_after_completed_substep(show_viewer, precision):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True),
        coupler_options=gs.options.SAPCouplerOptions(enable_rigid_fem_contact=True),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), fixed=False),
        material=gs.materials.Rigid(enable_coup_collision=False),
        name="disabled_rigid",
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(radius=0.02),
        material=gs.materials.FEM.Elastic(model="linear_corotated"),
        name="soft",
    )
    scene.build()
    with pytest.raises(gs.RigidFEMContactNotReadyError):
        scene.get_rigid_fem_contacts()
    scene.step(update_visualizer=False)
    batch = scene.get_rigid_fem_contacts()
    assert isinstance(batch, gs.RigidFEMContactBatch)
    assert len(batch) == 0
    assert batch.completed_scene_step == 0
    assert batch.completed_substep == 0
    assert batch.whitelist_receipt.entries[0].collision_enabled is False


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_enabled_scene_exports_nonempty_deterministic_last_substep_batch(show_viewer, precision):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=1, gravity=(0.0, 0.0, -9.81)),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True),
        coupler_options=gs.options.SAPCouplerOptions(enable_rigid_fem_contact=True),
        show_viewer=show_viewer,
    )
    support = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, -0.05), size=(0.12, 0.1, 0.1), fixed=True),
        material=gs.materials.Rigid(coup_collision_links=None),
        name="support",
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=(1.0, 0.0, 0.0), size=(0.02, 0.02, 0.02), fixed=False),
        material=gs.materials.Rigid(enable_coup_collision=False),
        name="free_dummy",
    )
    soft_positive = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.025, 0.0, 0.016), size=(0.03, 0.03, 0.03)),
        material=gs.materials.FEM.Elastic(model="linear_corotated"),
        name="soft_positive",
    )
    soft_negative = scene.add_entity(
        morph=gs.morphs.Box(pos=(-0.025, 0.0, 0.016), size=(0.03, 0.03, 0.03)),
        material=gs.materials.FEM.Elastic(model="linear_corotated"),
        name="soft_negative",
    )
    ordinary_bits = tuple((geom.contype, geom.conaffinity) for geom in support.geoms)
    scene.build()
    assert tuple((geom.contype, geom.conaffinity) for geom in support.geoms) == ordinary_bits

    with pytest.raises(gs.RigidFEMContactNotReadyError):
        scene.get_rigid_fem_contacts()
    expected_fem_names = {soft_positive.name, soft_negative.name}
    for completed_step in range(40):
        scene.step(update_visualizer=False)
        batch = scene.get_rigid_fem_contacts()
        if expected_fem_names.issubset(batch.fem_entity_names):
            break
    else:
        raise AssertionError("both FEM entities must produce public support contacts within 40 steps")

    assert np.all(batch.env_idx == 0)
    assert set(batch.rigid_entity_names) == {support.name}
    assert set(batch.rigid_link_names) == {support.links[0].name}
    assert expected_fem_names.issubset(batch.fem_entity_names)
    fem_by_name = {entity.name: entity for entity in (soft_positive, soft_negative)}
    for name, element_local in zip(batch.fem_entity_names, batch.fem_element_idx_local, strict=True):
        assert 0 <= element_local < fem_by_name[name].n_elements
    assert set(batch.rigid_geom_idx).issubset({geom.idx for geom in support.geoms})
    assert np.all(batch.normal_impulse_ns >= 0.0)
    assert all(isinstance(mode, gs.RigidFEMContactMode) for mode in batch.modes)
    assert batch.point_m.shape == (len(batch), 3)
    assert batch.signed_gap_m.shape == (len(batch),)
    assert batch.completed_scene_step == completed_step
    assert batch.completed_substep == completed_step

    fresh = scene.get_rigid_fem_contacts()
    assert fresh is not batch
    assert np.array_equal(fresh.point_m, batch.point_m)
    scene.step(update_visualizer=False)
    next_batch = scene.get_rigid_fem_contacts()
    assert next_batch.completed_scene_step == completed_step + 1
    assert next_batch.completed_substep == completed_step + 1
    scene.reset()
    with pytest.raises(gs.RigidFEMContactNotReadyError):
        scene.get_rigid_fem_contacts()
