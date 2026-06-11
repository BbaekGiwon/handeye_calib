#!/usr/bin/env python3
"""Estimate the base->camera extrinsic for an FR3.

Three input modes (pick exactly one):

  --apriltag_joint_samples_json   AprilTag pose + joint snapshots (hand-eye LS).
  --aruco_samples_json            EE-mounted ArUco images + joints (FK + solvePnP).
  --correspondences_json          Manual 3D(base)<->2D(image) points (solvePnP).

The ArUco and manual modes also require --camera_info_yaml (a ROS CameraInfo dump
with a 3x3 ``k`` and distortion ``d``).

Examples:
  # AprilTag hand-eye (default mode):
  python scripts/estimate_extrinsic.py \
      --apriltag_joint_samples_json data/snapshots/apriltag_franka_snapshots_*.json \
      --apriltag_id 1 --output_json data/results/extrinsic.json

  # Optionally constrain a known EE->tag mount and only solve its rotation:
  python scripts/estimate_extrinsic.py \
      --apriltag_joint_samples_json data/snapshots/snaps.json \
      --apriltag_ee_tag_xyz 0.0,0.0,0.05 --fix_apriltag_ee_tag_translation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from handeye_calib.fk import URDFForwardKinematics  # noqa: E402
from handeye_calib.geometry import pose_xyz_rpy_to_transform  # noqa: E402
from handeye_calib.robots import DEFAULT_ROBOT, ROBOTS, get_robot  # noqa: E402
from handeye_calib.solver import (  # noqa: E402
    build_ee_to_marker_transform,
    collect_aruco_correspondences,
    estimate_extrinsic_from_apriltag_snapshots,
    estimate_extrinsic_pnp,
    load_apriltag_joint_snapshots,
    load_aruco_samples,
    load_camera_info_yaml,
    load_manual_correspondences,
)


def _parse_vec3(text: str) -> np.ndarray:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated values")
    return np.asarray(vals, dtype=np.float64)


def _build_camera_ext_payload(result: dict) -> dict:
    """Minimal payload consumed by downstream nodes: just T_base_camera."""
    return {"T_base_camera": result["T_base_camera"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate base->camera extrinsic (FR3 / A0509)")
    parser.add_argument(
        "--robot", choices=sorted(ROBOTS), default=DEFAULT_ROBOT,
        help="Robot preset selecting URDF, base/end link and joint count.",
    )
    parser.add_argument("--camera_info_yaml", type=Path, help="ROS2 CameraInfo YAML (aruco/manual modes)")
    parser.add_argument("--urdf", type=Path, default=None, help="Override the preset URDF path for FK")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correspondences_json", type=Path,
                       help="JSON with points_3d_base and points_2d_image")
    group.add_argument("--aruco_samples_json", type=Path,
                       help="JSON with samples=[{image_path, joint_positions}]")
    group.add_argument("--apriltag_joint_samples_json", type=Path,
                       help="JSON with snapshots=[{april_tags, franka_right_arm}]")

    # ArUco-mode options
    parser.add_argument("--marker_id", type=int, default=0)
    parser.add_argument("--aruco_dict", type=str, default="4X4_50")
    parser.add_argument("--marker_size_m", type=float, default=0.04)
    parser.add_argument("--ee_to_marker_xyz", type=_parse_vec3, default=np.zeros(3))
    parser.add_argument("--ee_to_marker_rpy", type=_parse_vec3, default=np.zeros(3),
                        help="EE->marker roll,pitch,yaw in radians")

    # AprilTag-mode options
    parser.add_argument("--apriltag_id", type=int, default=1,
                        help="Target AprilTag ID (use -1 to take the first tag in each snapshot)")
    parser.add_argument("--apriltag_ee_tag_xyz", type=_parse_vec3, default=None,
                        help="Optional EE->tag translation x,y,z (m)")
    parser.add_argument("--apriltag_ee_tag_rpy_deg", type=_parse_vec3, default=None,
                        help="Optional EE->tag roll,pitch,yaw in degrees")
    parser.add_argument("--fix_apriltag_ee_tag", action="store_true",
                        help="Keep the provided EE->tag transform fixed")
    parser.add_argument("--fix_apriltag_ee_tag_translation", action="store_true",
                        help="Fix EE->tag translation, solve only its rotation")

    parser.add_argument("--output_json", type=Path, required=True,
                        help="Path to save the calibration result JSON")
    parser.add_argument("--full_output_json", type=Path, default=None,
                        help="Optional path to also save the full diagnostic result")
    args = parser.parse_args()

    robot = get_robot(args.robot)

    def build_fk() -> URDFForwardKinematics:
        return URDFForwardKinematics(
            urdf_path=args.urdf or robot.urdf_path,
            base_link=robot.base_link,
            end_link=robot.end_link,
        )

    if args.correspondences_json is not None:
        if args.camera_info_yaml is None:
            raise ValueError("--camera_info_yaml is required with --correspondences_json")
        camera_matrix, dist_coeffs, camera_meta = load_camera_info_yaml(args.camera_info_yaml)
        points_3d_base, points_2d_image = load_manual_correspondences(args.correspondences_json)
        result = estimate_extrinsic_pnp(camera_matrix, dist_coeffs, points_3d_base, points_2d_image)
        result.update({
            "mode": "manual_correspondences",
            "camera_info": camera_meta,
            "num_correspondences": int(len(points_3d_base)),
        })
    elif args.aruco_samples_json is not None:
        if args.camera_info_yaml is None:
            raise ValueError("--camera_info_yaml is required with --aruco_samples_json")
        camera_matrix, dist_coeffs, camera_meta = load_camera_info_yaml(args.camera_info_yaml)
        fk = build_fk()
        samples = load_aruco_samples(args.aruco_samples_json)
        ee_to_marker = build_ee_to_marker_transform(args.ee_to_marker_xyz, args.ee_to_marker_rpy)
        points_3d_base, points_2d_image, per_sample = collect_aruco_correspondences(
            samples=samples, fk=fk, marker_id=args.marker_id,
            marker_size_m=args.marker_size_m, ee_to_marker_transform=ee_to_marker,
            aruco_dict=args.aruco_dict,
        )
        result = estimate_extrinsic_pnp(camera_matrix, dist_coeffs, points_3d_base, points_2d_image)
        result.update({
            "mode": "aruco_auto_correspondences",
            "camera_info": camera_meta,
            "aruco_dict": args.aruco_dict,
            "marker_id": int(args.marker_id),
            "marker_size_m": float(args.marker_size_m),
            "num_correspondences": int(len(points_3d_base)),
            "samples": per_sample,
        })
    else:
        fk = build_fk()
        snapshots = load_apriltag_joint_snapshots(args.apriltag_joint_samples_json)
        initial_ee_tag = None
        if args.apriltag_ee_tag_xyz is not None or args.apriltag_ee_tag_rpy_deg is not None:
            xyz = args.apriltag_ee_tag_xyz if args.apriltag_ee_tag_xyz is not None else np.zeros(3)
            rpy = args.apriltag_ee_tag_rpy_deg if args.apriltag_ee_tag_rpy_deg is not None else np.zeros(3)
            initial_ee_tag = pose_xyz_rpy_to_transform(*xyz, *rpy, degrees=True)
        tag_id = None if args.apriltag_id < 0 else args.apriltag_id
        result = estimate_extrinsic_from_apriltag_snapshots(
            snapshots=snapshots, fk=fk, tag_id=tag_id,
            initial_ee_tag=initial_ee_tag,
            fix_ee_tag=args.fix_apriltag_ee_tag,
            fix_ee_tag_translation=args.fix_apriltag_ee_tag_translation,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(_build_camera_ext_payload(result), f, indent=2)
    result["output_json"] = str(args.output_json)

    if args.full_output_json is not None:
        args.full_output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.full_output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
