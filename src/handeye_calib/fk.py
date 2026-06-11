"""Forward kinematics from a URDF kinematic chain (FR3, Doosan A0509, ...).

There is no usable ``franka`` / ``pinocchio`` / Doosan binding in this
environment, so FK is computed directly by walking the URDF chain from
``base_link`` to ``end_link`` and composing each joint's static origin with its
revolute angle. The number of joints is inferred from the chain, so the same
class serves the 7-axis FR3 and the 6-axis A0509.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import axis_angle_to_matrix, make_transform, rpy_to_matrix

# Bundled URDF shipped with this repo. Override with --urdf if you have another.
DEFAULT_URDF_PATH = Path(__file__).resolve().parents[2] / "assets" / "urdf" / "fr3.urdf"


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    joint_type: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def _parse_xyz(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=np.float64)
    return np.array([float(v) for v in text.split()], dtype=np.float64)


class URDFForwardKinematics:
    """Compute ``T_base_ee`` (base_link -> end_link) from N joint positions.

    N (``num_joints``) is the number of revolute joints on the resolved chain.
    """

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_URDF_PATH,
        end_link: str = "fr3_link8",
        base_link: str = "fr3_link0",
    ):
        self.urdf_path = Path(urdf_path)
        self.end_link = end_link
        self.base_link = base_link
        self._joint_chain = self._load_chain()
        self.num_joints = sum(1 for j in self._joint_chain if j.joint_type == "revolute")

    @classmethod
    def from_robot(cls, robot) -> "URDFForwardKinematics":
        """Build FK from a :class:`handeye_calib.robots.RobotConfig`."""
        return cls(urdf_path=robot.urdf_path, base_link=robot.base_link, end_link=robot.end_link)

    def _load_chain(self) -> list[JointSpec]:
        root = ET.parse(self.urdf_path).getroot()
        child_to_joint: dict[str, JointSpec] = {}
        all_links: set[str] = set()
        parent_links: set[str] = set()
        child_links: set[str] = set()

        for link_elem in root.findall("link"):
            all_links.add(link_elem.attrib["name"])

        for joint_elem in root.findall("joint"):
            name = joint_elem.attrib["name"]
            parent = joint_elem.find("parent").attrib["link"]
            child = joint_elem.find("child").attrib["link"]
            parent_links.add(parent)
            child_links.add(child)
            joint_type = joint_elem.attrib["type"]
            origin = joint_elem.find("origin")
            xyz = _parse_xyz(origin.attrib.get("xyz") if origin is not None else None)
            rpy = _parse_xyz(origin.attrib.get("rpy") if origin is not None else None)
            axis_elem = joint_elem.find("axis")
            axis = _parse_xyz(axis_elem.attrib.get("xyz") if axis_elem is not None else None)
            child_to_joint[child] = JointSpec(name, parent, child, joint_type, xyz, rpy, axis)

        resolved_base = self._resolve_link_name(
            requested=self.base_link,
            candidates=["fr3_link0", "base", "panda_link0"],
            available_links=all_links,
            fallback=sorted(parent_links - child_links),
        )
        resolved_end = self._resolve_link_name(
            requested=self.end_link,
            candidates=["fr3_link8", "link8", "panda_link8"],
            available_links=all_links,
            fallback=sorted(child_links - parent_links),
        )
        self.base_link = resolved_base
        self.end_link = resolved_end

        chain: list[JointSpec] = []
        current = self.end_link
        while current != self.base_link:
            if current not in child_to_joint:
                raise ValueError(f"Could not trace joint chain from {self.end_link} to {self.base_link}")
            joint = child_to_joint[current]
            chain.append(joint)
            current = joint.parent
        chain.reverse()
        return chain

    @staticmethod
    def _resolve_link_name(
        requested: str,
        candidates: list[str],
        available_links: set[str],
        fallback: list[str],
    ) -> str:
        if requested in available_links:
            return requested
        for candidate in candidates:
            if candidate in available_links:
                return candidate
        if len(fallback) == 1:
            return fallback[0]
        raise ValueError(
            f"Could not resolve link name '{requested}'. "
            f"Available example links: {sorted(list(available_links))[:10]}"
        )

    def compute(self, joint_positions) -> np.ndarray:
        q = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if q.size != self.num_joints:
            raise ValueError(
                f"FK chain {self.base_link}->{self.end_link} expects "
                f"{self.num_joints} joint positions, got {q.size}"
            )

        transform = np.eye(4, dtype=np.float64)
        revolute_index = 0

        for joint in self._joint_chain:
            static_tf = make_transform(rpy_to_matrix(*joint.rpy), joint.xyz)
            transform = transform @ static_tf
            if joint.joint_type == "revolute":
                if revolute_index >= q.size:
                    raise ValueError(
                        f"URDF revolute joint count exceeds provided joint positions at {joint.name}"
                    )
                angle = float(q[revolute_index])
                revolute_index += 1
                transform = transform @ make_transform(
                    axis_angle_to_matrix(joint.axis, angle), np.zeros(3)
                )
            elif joint.joint_type != "fixed":
                raise NotImplementedError(f"Unsupported joint type: {joint.joint_type}")

        if revolute_index != q.size:
            raise ValueError(
                f"FK used {revolute_index} revolute joints, but received {q.size} joint positions"
            )

        return transform

    def compute_pose_dict(self, joint_positions) -> dict:
        transform = self.compute(joint_positions)
        return {
            "base_link": self.base_link,
            "end_link": self.end_link,
            "translation_xyz": transform[:3, 3].tolist(),
            "rotation_matrix": transform[:3, :3].tolist(),
            "transform_matrix": transform.tolist(),
            "urdf_path": str(self.urdf_path),
        }


# Backward-compatible alias (the class is no longer FR3-specific).
FR3ForwardKinematics = URDFForwardKinematics


def _parse_joint_list(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected comma-separated joint positions.")
    return [float(v) for v in parts]


def main() -> None:
    from .robots import DEFAULT_ROBOT, ROBOTS, get_robot

    parser = argparse.ArgumentParser(description="Compute forward kinematics from joint positions.")
    parser.add_argument(
        "--robot", choices=sorted(ROBOTS), default=DEFAULT_ROBOT,
        help="Robot preset selecting URDF, base/end link.",
    )
    parser.add_argument(
        "--joints", type=_parse_joint_list, required=True,
        help="Comma-separated joint positions in radians (7 for fr3, 6 for a0509).",
    )
    parser.add_argument("--urdf", type=Path, default=None, help="Override URDF path.")
    parser.add_argument("--end_link", type=str, default=None, help="Override end-effector link name.")
    args = parser.parse_args()

    robot = get_robot(args.robot)
    fk = URDFForwardKinematics(
        urdf_path=args.urdf or robot.urdf_path,
        base_link=robot.base_link,
        end_link=args.end_link or robot.end_link,
    )
    print(json.dumps(fk.compute_pose_dict(args.joints), indent=2))


if __name__ == "__main__":
    main()
