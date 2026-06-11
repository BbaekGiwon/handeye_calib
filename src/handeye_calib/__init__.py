"""AprilTag-based hand-eye calibration for a Franka FR3.

Pipeline:
    1. ``record_snapshots``    -- collect AprilTag pose + FR3 joint snapshots.
    2. ``estimate_extrinsic``  -- solve the base->camera extrinsic.

See the ``scripts/`` entry points and the README for usage.
"""

from .fk import DEFAULT_URDF_PATH, FR3ForwardKinematics, URDFForwardKinematics
from .robots import ROBOTS, RobotConfig, get_robot

__all__ = [
    "URDFForwardKinematics",
    "FR3ForwardKinematics",
    "DEFAULT_URDF_PATH",
    "RobotConfig",
    "ROBOTS",
    "get_robot",
    "__version__",
]

__version__ = "0.2.0"
