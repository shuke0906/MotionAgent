from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from motion_agent.constraints import (
    ConditionStore,
    ConstraintRequest,
    ConstraintTarget,
    FeatureMaskBuilder,
    GEMConstraintEncoder,
    SMPLJointRegistry,
    compose_conditions,
    compile_constraint,
)
from motion_agent.constraints.gem_features import GEMFeatureMapper
from motion_agent.constraints.kinematics import SimpleKinematics
from motion_agent.constraints.schemas import HardMotionCondition, TimeSpec
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.schemas import CandidateMetadata, SamplerMetadata
from motion_agent.keyframes import KeyframeRequest, build_keyframe
from motion_agent.keyframes.store import KeyframeStore
from motion_agent.state.reducer import StateReducer
from motion_agent.state.schemas import MotionSegmentSummary


def prepared_state(total_frames: int = 180):
    reducer = StateReducer()
    state = reducer.create_run("walk forward then sit", run_id="phase6_unit", fps=30, total_frames=total_frames)
    return reducer.commit_motion_plan(
        state,
        motion_spec_id="spec_fixture",
        gem_text_condition_id="text_fixture",
        segments=[MotionSegmentSummary(segment_id=0, start_s=0.0, end_s=total_frames / 30, action="walk")],
    )


def stores(root: Path):
    return ConditionStore(root / "conditions"), KeyframeStore(root / "keyframes")


class Phase6ConstraintTests(unittest.TestCase):
    def test_joint_target_hard_constraint_uses_ik(self):
        state = prepared_state()
        kinematics = SimpleKinematics()
        target = kinematics.fk()[ "right_wrist"] + torch.tensor([0.05, 0.02, 0.01])
        with tempfile.TemporaryDirectory() as tmp:
            store = ConditionStore(Path(tmp))
            result = compile_constraint(
                state,
                ConstraintRequest(
                    target_segment=0,
                    constraint_type="joint_target",
                    requested_strength="hard",
                    time_spec=TimeSpec(type="point", time_s=3.0),
                    body_parts=["right_wrist"],
                    target=ConstraintTarget(target_type="point", values={"xyz": target.tolist()}),
                ),
                store=store,
            )
            self.assertEqual(result.status, "compiled")
            self.assertEqual(result.constraint.effective_mode, "hard_condition")
            condition = store.load_hard_condition(result.constraint.hard_condition_handle)
            self.assertGreater(condition.mask[90].sum().item(), 0)
            self.assertEqual(result.constraint.verification_spec.metric_type, "joint_position_error")

    def test_body_part_pose_masks_only_requested_body_part(self):
        state = prepared_state()
        pose = torch.zeros(21, 6)
        pose[18, 0] = 0.25
        with tempfile.TemporaryDirectory() as tmp:
            store = ConditionStore(Path(tmp))
            result = compile_constraint(
                state,
                ConstraintRequest(
                    target_segment=0,
                    constraint_type="body_part_pose",
                    requested_strength="hard",
                    time_spec=TimeSpec(type="point", time_s=2.0),
                    body_parts=["right_arm"],
                    target=ConstraintTarget(target_type="reference_pose", values={"body_pose": pose.tolist()}),
                ),
                store=store,
            )
            condition = store.load_hard_condition(result.constraint.hard_condition_handle)
            mapper = GEMFeatureMapper()
            betas = slice(*mapper.get_group_slice("betas"))
            self.assertEqual(condition.mask[60, betas].sum().item(), 0.0)
            self.assertGreater(condition.mask[60].sum().item(), 0.0)

    def test_sparse_feature_mask_only_hits_target_slice_and_frame(self):
        mask = FeatureMaskBuilder(120).for_body_joints(["right_elbow"], [90])
        mapper = GEMFeatureMapper()
        elbow_slice = slice(*mapper.get_body_joint_slice("right_elbow"))
        self.assertEqual(mask[:90].sum().item() + mask[91:].sum().item(), 0.0)
        self.assertEqual(mask[90, elbow_slice].sum().item(), 6.0)
        self.assertEqual(mask[90].sum().item(), 6.0)

    def test_root_trajectory_contact_and_fixed_joint_compile_to_reusable_specs(self):
        state = prepared_state()
        trajectory = compile_constraint(
            state,
            ConstraintRequest(
                target_segment=0,
                constraint_type="root_trajectory",
                time_spec=TimeSpec(type="window", start_s=0.0, end_s=2.0),
                target=ConstraintTarget(target_type="trajectory", values={"waypoints": [[0, 0, 0], [1, 0, 0]]}),
            ),
        )
        contact = compile_constraint(
            state,
            ConstraintRequest(
                target_segment=0,
                constraint_type="contact",
                time_spec=TimeSpec(type="point", time_s=1.0),
                body_parts=["right_foot"],
                target=ConstraintTarget(target_type="plane", values={"normal": [0, 1, 0], "offset": 0.0}),
            ),
        )
        fixed = compile_constraint(
            state,
            ConstraintRequest(
                target_segment=0,
                constraint_type="fixed_joint",
                time_spec=TimeSpec(type="window", start_s=1.0, end_s=2.0),
                body_parts=["left_foot"],
            ),
        )
        self.assertEqual(trajectory.constraint.reward_spec.reward_type, "trajectory_rmse")
        self.assertEqual(contact.constraint.verification_spec.metric_type, "contact_distance_velocity")
        self.assertEqual(fixed.constraint.verification_spec.metric_type, "joint_drift")

    def test_ik_success_and_failure_paths(self):
        state = prepared_state()
        base = SimpleKinematics().fk()["right_wrist"]
        success = compile_constraint(
            state,
            ConstraintRequest(
                target_segment=0,
                constraint_type="joint_target",
                requested_strength="hard",
                time_spec=TimeSpec(type="point", time_s=1.0),
                body_parts=["right_wrist"],
                target=ConstraintTarget(target_type="point", values={"xyz": (base + torch.tensor([0.01, 0, 0])).tolist()}),
            ),
        )
        failure = compile_constraint(
            state,
            ConstraintRequest(
                target_segment=0,
                constraint_type="joint_target",
                requested_strength="hard",
                time_spec=TimeSpec(type="point", time_s=1.0),
                body_parts=["right_wrist"],
                target=ConstraintTarget(target_type="point", values={"xyz": [100.0, 100.0, 100.0]}),
            ),
        )
        self.assertEqual(success.status, "compiled")
        self.assertEqual(failure.status, "compiled_soft")
        self.assertIn("reach", failure.warnings[0])

    def test_gem_encode_decode_consistency(self):
        encoder = GEMConstraintEncoder()
        smpl = encoder.canonical_sequence(12)
        smpl["body_pose"][:, 0] = torch.linspace(0.0, 1.0, 12)
        encoded = encoder.encode_sequence(smpl)
        decoded = encoder.decode_sequence(encoded)
        torch.testing.assert_close(decoded["body_pose"], smpl["body_pose"])
        torch.testing.assert_close(decoded["betas"], smpl["betas"])

    def test_fk_ik_numerical_residual(self):
        kinematics = SimpleKinematics()
        base_pose = kinematics.canonical_body_pose()
        wrist = kinematics.fk(base_pose)["right_wrist"]
        target = wrist + torch.tensor([0.08, -0.02, 0.03])
        solved = kinematics.solve_ik(
            base_pose=base_pose,
            target_joint="right_wrist",
            target_position=target,
            tolerance_m=1e-5,
        )
        residual = torch.linalg.vector_norm(kinematics.fk(solved.body_pose)["right_wrist"] - target)
        self.assertLessEqual(float(residual), 1e-5)

    def test_conflicting_hard_conditions_block_generation(self):
        a = HardMotionCondition(values=torch.zeros(100, 151), mask=torch.zeros(100, 151), source_constraint_id="constraint_a")
        b = HardMotionCondition(values=torch.zeros(100, 151), mask=torch.zeros(100, 151), source_constraint_id="constraint_b")
        a.mask[90, 0:6] = 1
        b.mask[90, 0:6] = 1
        b.values[90, 0:6] = 1
        bundle = compose_conditions([a, b], total_frames=100)
        self.assertTrue(bundle.generation_blocked)
        self.assertEqual(bundle.conflicts[0].frame_indices, [90])

    def test_missing_scene_geometry_never_fabricates_xyz(self):
        state = prepared_state()
        result = compile_constraint(
            state,
            ConstraintRequest(
                target_segment=0,
                constraint_type="contact",
                requested_strength="hard",
                time_spec=TimeSpec(type="point", time_s=1.0),
                body_parts=["right_wrist"],
                target=ConstraintTarget(target_type="scene_anchor", coordinate_frame="scene_local", values={"label": "table"}),
            ),
        )
        self.assertEqual(result.status, "needs_geometry")
        self.assertIsNone(result.constraint)


class Phase6KeyframeTests(unittest.TestCase):
    def test_add_replace_remove_and_explicit_pose(self):
        state = prepared_state()
        pose = {"body_pose": torch.zeros(21, 6).tolist(), "global_orient": [0, 0, 0, 1, 0, 0]}
        with tempfile.TemporaryDirectory() as tmp:
            condition_store, keyframe_store = stores(Path(tmp))
            added = build_keyframe(
                state,
                KeyframeRequest(
                    target_segment=0,
                    time=1.0,
                    description="explicit standing pose",
                    source_preference="explicit",
                    explicit_pose=pose,
                ),
                condition_store=condition_store,
                keyframe_store=keyframe_store,
            )
            replaced = build_keyframe(
                state,
                KeyframeRequest(
                    mode="replace",
                    keyframe_id=added.keyframe.keyframe_id,
                    target_segment=0,
                    time=1.0,
                    description="replacement standing pose",
                    source_preference="explicit",
                    explicit_pose=pose,
                ),
                condition_store=condition_store,
                keyframe_store=keyframe_store,
            )
            removed = build_keyframe(
                state,
                KeyframeRequest(
                    mode="remove",
                    keyframe_id=added.keyframe.keyframe_id,
                    target_segment=0,
                    time=1.0,
                    description="remove",
                ),
            )
            self.assertEqual(added.status, "compiled")
            self.assertEqual(replaced.keyframe.keyframe_id, added.keyframe.keyframe_id)
            self.assertEqual(removed.status, "compiled")

    def test_fixture_retrieved_pose_and_motion_frame_sources(self):
        state = prepared_state()
        with tempfile.TemporaryDirectory() as tmp:
            condition_store, keyframe_store = stores(Path(tmp))
            pose_result = build_keyframe(
                state,
                KeyframeRequest(
                    target_segment=0,
                    time=2.0,
                    description="fixture seated pose",
                    source_preference="retrieval",
                    retrieved_references=[{"ref_id": "ref_pose", "pose_handle": "fixture_pose", "retrieval_score": 0.9, "metadata": {}}],
                ),
                condition_store=condition_store,
                keyframe_store=keyframe_store,
            )
            motion_result = build_keyframe(
                state,
                KeyframeRequest(
                    target_segment=0,
                    time=2.5,
                    description="fixture motion frame",
                    source_preference="retrieval",
                    retrieved_references=[{"ref_id": "ref_motion", "motion_handle": "fixture_motion", "retrieval_score": 0.8, "metadata": {"frame": 75}}],
                ),
                condition_store=condition_store,
                keyframe_store=keyframe_store,
            )
            self.assertEqual(pose_result.keyframe.source_type, "retrieval_pose")
            self.assertEqual(motion_result.keyframe.source_type, "retrieval_motion")
            self.assertTrue(any("DEFERRED_PHASE5_DATA" in warning for warning in pose_result.warnings))

    def test_no_reference_behavior(self):
        result = build_keyframe(
            prepared_state(),
            KeyframeRequest(target_segment=0, time=1.0, description="stable seated pose", source_preference="retrieval"),
        )
        self.assertEqual(result.status, "needs_reference")

    def test_existing_candidate_frame_source(self):
        state = prepared_state()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CandidateStore(root / "candidates")
            sampler = SamplerMetadata(seed=1, checkpoint_version="fake")
            metadata = CandidateMetadata(
                seed=1,
                generation_id="gen_phase6",
                condition_fingerprint="cond",
                candidate_fingerprint="phase6_candidate",
                checkpoint_version="fake",
                sampler=sampler,
                scope="full",
                postprocess_policy="none",
                runtime_ms=1.0,
                frame_count=180,
                motion_dim=151,
                technical_valid=True,
            )
            record = store.save(
                tensors={
                    "motion_repr": torch.zeros(180, 151),
                    "body_params_global": {
                        "body_pose": torch.zeros(180, 63),
                        "global_orient": torch.zeros(180, 3),
                        "transl": torch.zeros(180, 3),
                        "betas": torch.zeros(180, 10),
                    },
                },
                metadata=metadata,
            )
            result = build_keyframe(
                state,
                KeyframeRequest(
                    target_segment=0,
                    time=1.0,
                    description="candidate pose",
                    source_preference="generation",
                    source_candidate_id=record.candidate.candidate_id,
                ),
                condition_store=ConditionStore(root / "conditions"),
                keyframe_store=KeyframeStore(root / "keyframes"),
                candidate_store_root=root / "candidates",
            )
            self.assertEqual(result.keyframe.source_type, "candidate")

    def test_keyframe_ik_success_and_failure(self):
        state = prepared_state()
        kinematics = SimpleKinematics()
        wrist = kinematics.fk()["right_wrist"]
        pose = {"body_pose": torch.zeros(21, 6).tolist()}
        success = build_keyframe(
            state,
            KeyframeRequest(
                target_segment=0,
                time=1.0,
                description="ik pose",
                source_preference="explicit",
                explicit_pose=pose,
                pose_targets=[{"body_part": "right_wrist", "target_type": "position", "values": {"xyz": (wrist + torch.tensor([0.02, 0, 0])).tolist()}}],
            ),
        )
        failure = build_keyframe(
            state,
            KeyframeRequest(
                target_segment=0,
                time=1.0,
                description="ik pose",
                source_preference="explicit",
                explicit_pose=pose,
                pose_targets=[{"body_part": "right_wrist", "target_type": "position", "values": {"xyz": [99, 99, 99]}}],
            ),
        )
        self.assertEqual(success.keyframe.source_type, "ik_refined")
        self.assertEqual(failure.status, "needs_reference")

    def test_frame_boundary_root_orientation_body_shape_and_verification_spec(self):
        state = prepared_state()
        with tempfile.TemporaryDirectory() as tmp:
            condition_store, keyframe_store = stores(Path(tmp))
            result = build_keyframe(
                state,
                KeyframeRequest(
                    target_segment=0,
                    time=(state.task.total_frames - 1) / state.task.fps,
                    description="final pose",
                    source_preference="explicit",
                    explicit_pose={"body_pose": torch.zeros(21, 6).tolist()},
                    control_root_orientation=True,
                ),
                condition_store=condition_store,
                keyframe_store=keyframe_store,
            )
            condition = condition_store.load_hard_condition(result.keyframe.hard_condition_handle)
            mapper = GEMFeatureMapper()
            betas = slice(*mapper.get_group_slice("betas"))
            root = slice(*mapper.get_group_slice("global_orient"))
            self.assertEqual(result.keyframe.target_frame, state.task.total_frames - 1)
            self.assertEqual(condition.mask[result.keyframe.target_frame, betas].sum().item(), 0.0)
            self.assertEqual(condition.mask[result.keyframe.target_frame, root].sum().item(), 6.0)
            self.assertIn("root_orientation_error", result.keyframe.verification_spec.metrics)

    def test_multiple_keyframes_and_constraint_keyframe_conflict(self):
        state = prepared_state(total_frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            condition_store, keyframe_store = stores(Path(tmp))
            first = build_keyframe(
                state,
                KeyframeRequest(target_segment=0, time=1.0, description="first", source_preference="explicit", explicit_pose={"body_pose": torch.zeros(21, 6).tolist()}),
                condition_store=condition_store,
                keyframe_store=keyframe_store,
            )
            second = build_keyframe(
                state,
                KeyframeRequest(target_segment=0, time=2.0, description="second", source_preference="explicit", explicit_pose={"body_pose": torch.ones(21, 6).tolist()}),
                condition_store=condition_store,
                keyframe_store=keyframe_store,
            )
            compatible = compose_conditions(
                [
                    condition_store.load_hard_condition(first.keyframe.hard_condition_handle),
                    condition_store.load_hard_condition(second.keyframe.hard_condition_handle),
                ],
                total_frames=100,
                store=condition_store,
            )
            self.assertFalse(compatible.generation_blocked)
            conflicting_constraint = HardMotionCondition(
                values=torch.ones(100, 151),
                mask=FeatureMaskBuilder(100).for_whole_body_keyframe(30),
                source_constraint_id="constraint_conflict",
            )
            conflict = compose_conditions(
                [
                    condition_store.load_hard_condition(first.keyframe.hard_condition_handle),
                    conflicting_constraint,
                ],
                total_frames=100,
                store=condition_store,
            )
            self.assertTrue(conflict.generation_blocked)


if __name__ == "__main__":
    unittest.main()
