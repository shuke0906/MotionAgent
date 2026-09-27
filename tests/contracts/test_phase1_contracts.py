from __future__ import annotations

import unittest

from motion_agent.common.fingerprints import (
    candidate_fingerprint,
    condition_fingerprint,
    evidence_fingerprint,
    failure_signature,
    repair_signature,
    retrieval_signature,
)
from motion_agent.state.schemas import ArtifactHandle, MotionAgentState


class Phase1ContractTests(unittest.TestCase):
    def test_canonical_schema_is_single_source_importable(self):
        self.assertIn("state_version", MotionAgentState.model_fields)
        self.assertIn("artifact_id", ArtifactHandle.model_fields)

    def test_fingerprints_are_stable_and_namespaced(self):
        payload = {"b": [2, 1], "a": {"x": "y"}}
        self.assertEqual(
            condition_fingerprint(payload),
            condition_fingerprint({"a": {"x": "y"}, "b": [2, 1]}),
        )
        self.assertNotEqual(retrieval_signature(payload), condition_fingerprint(payload))
        self.assertNotEqual(candidate_fingerprint(payload), evidence_fingerprint(payload))
        self.assertNotEqual(failure_signature(payload), repair_signature(payload))

