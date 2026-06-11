#!/usr/bin/env python3
"""Collect AprilTag pose + FR3 joint snapshots for hand-eye calibration.

Runs a RealSense color stream with live AprilTag detection while subscribing to
ROS2 ``/joint_states``. Press ``s`` to capture a snapshot and ``q`` to write the
JSON consumed by ``estimate_extrinsic.py``.

Requires (typically in a dedicated conda env): pyrealsense2, pupil_apriltags,
opencv-python, numpy, and a sourced ROS2 (rclpy + sensor_msgs).

Example:
    python scripts/record_snapshots.py --output-dir data/snapshots --tag-size 0.04
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from handeye_calib.recorder import TAG_FAMILY, TAG_SIZE_M, run_recorder  # noqa: E402
from handeye_calib.robots import DEFAULT_ROBOT, ROBOTS, get_robot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AprilTag + robot joint snapshot recorder")
    parser.add_argument(
        "--robot", choices=sorted(ROBOTS), default=DEFAULT_ROBOT,
        help="Robot preset selecting joint names / joint count (fr3=7-axis, a0509=6-axis).",
    )
    parser.add_argument("--tag-size", type=float, default=TAG_SIZE_M, help="AprilTag side length (m)")
    parser.add_argument("--tag-family", default=TAG_FAMILY, help="AprilTag family, e.g. tag36h11")
    parser.add_argument("--no-smooth", action="store_true", help="Disable position smoothing")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "snapshots"),
        help="Directory to write the snapshot JSON into",
    )
    parser.add_argument("--output-prefix", default=None,
                        help="Output filename prefix (default: apriltag_<robot>_snapshots)")
    parser.add_argument("--joint-topic", default=None,
                        help="ROS2 JointState topic (default: the robot preset's topic)")
    args = parser.parse_args()

    robot = get_robot(args.robot)
    prefix = args.output_prefix or f"apriltag_{robot.name}_snapshots"

    run_recorder(
        joint_names=robot.joint_names,
        robot_name=robot.name,
        snapshot_key=robot.snapshot_key,
        tag_size=args.tag_size,
        tag_family=args.tag_family,
        smooth_alpha=1.0 if args.no_smooth else 0.6,
        output_dir=os.path.abspath(args.output_dir),
        output_prefix=prefix,
        joint_topic=args.joint_topic or robot.joint_topic,
    )


if __name__ == "__main__":
    main()
