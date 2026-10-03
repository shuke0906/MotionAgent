"""Physical verifier functions."""

from motion_agent.verification.physical.foot_skating import verify_foot_skating
from motion_agent.verification.physical.penetration import verify_ground_penetration
from motion_agent.verification.physical.smoothness import verify_smoothness

__all__ = ["verify_foot_skating", "verify_ground_penetration", "verify_smoothness"]
