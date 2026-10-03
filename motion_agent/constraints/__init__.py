"""Phase 6 constraint and shared hard-condition infrastructure."""

from motion_agent.constraints.compiler import compile_constraint
from motion_agent.constraints.composer import ConstraintComposer, compose_conditions, compose_constraints
from motion_agent.constraints.feature_mask import FeatureMaskBuilder
from motion_agent.constraints.gem_encoder import GEMConstraintEncoder
from motion_agent.constraints.joints import SMPLJointRegistry
from motion_agent.constraints.schemas import (
    CompiledConstraint,
    ConstraintBundle,
    ConstraintCompileResult,
    ConstraintRequest,
    ConstraintTarget,
    HardMotionCondition,
    RewardSpec,
    TimeSpec,
    VerificationSpec,
)
from motion_agent.constraints.store import ConditionStore

__all__ = [
    "CompiledConstraint",
    "ConstraintBundle",
    "ConstraintCompileResult",
    "ConstraintComposer",
    "ConstraintRequest",
    "ConstraintTarget",
    "ConditionStore",
    "FeatureMaskBuilder",
    "GEMConstraintEncoder",
    "HardMotionCondition",
    "RewardSpec",
    "SMPLJointRegistry",
    "TimeSpec",
    "VerificationSpec",
    "compile_constraint",
    "compose_conditions",
    "compose_constraints",
]
