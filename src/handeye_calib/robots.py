"""Robot presets — everything FK and the recorder need to differ per arm.

Add a new arm by registering another ``RobotConfig`` in ``ROBOTS`` (URDF path,
base/end link, ordered joint names matching the ROS ``/joint_states`` message).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "urdf"


@dataclass(frozen=True)
class RobotConfig:
    name: str
    urdf_path: Path
    base_link: str
    end_link: str
    joint_names: list[str]              # ordered, as named in /joint_states
    joint_topic: str = "/joint_states"
    joint_message_type: str = "joint_state"
    snapshot_key: str = "robot"         # JSON key the recorder writes joints under

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


ROBOTS: dict[str, RobotConfig] = {
    # Franka FR3 — 7-axis, fr3_link0 -> fr3_link8.
    "fr3": RobotConfig(
        name="fr3",
        urdf_path=_ASSETS / "fr3.urdf",
        base_link="fr3_link0",
        end_link="fr3_link8",
        joint_names=[f"fr3_joint{i}" for i in range(1, 8)],
        joint_topic="/franka/joint_position",
        joint_message_type="float64_multi_array",
    ),
    # Doosan A0509 — 6-axis collaborative arm, base_link -> link_6 (flange).
    "a0509": RobotConfig(
        name="a0509",
        urdf_path=_ASSETS / "a0509.urdf",
        base_link="base_link",
        end_link="link_6",
        joint_names=[f"joint_{i}" for i in range(1, 7)],
    ),
}

DEFAULT_ROBOT = "fr3"


def get_robot(name: str) -> RobotConfig:
    key = name.lower()
    if key not in ROBOTS:
        raise ValueError(
            f"Unknown robot '{name}'. Available: {sorted(ROBOTS)}"
        )
    return ROBOTS[key]
