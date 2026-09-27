"""Typed errors shared by the MotionAgent harness."""


class MotionAgentError(Exception):
    """Base class for MotionAgent errors."""


class StateConflictError(MotionAgentError):
    """Raised when an optimistic-concurrency commit uses a stale version."""


class StateInvariantError(MotionAgentError):
    """Raised when canonical state invariants are violated."""


class MissingArtifactError(StateInvariantError):
    """Raised when state references an artifact that is not in the artifact store."""


class ArtifactStoreError(MotionAgentError):
    """Raised when artifact persistence fails."""


class CheckpointError(MotionAgentError):
    """Raised when graph checkpoint persistence fails."""

