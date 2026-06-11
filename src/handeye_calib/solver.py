"""Base->camera extrinsic estimation for FR3.

Three input modes are supported:

1. ``manual``   - explicit 3D(base)<->2D(image) correspondences + solvePnP.
2. ``aruco``    - detect an ArUco marker rigidly mounted on the EE in each
                  image, build its corner points in base frame via FK, solvePnP.
3. ``apriltag`` - AprilTag pose + joint snapshots; solve a hand-eye style
                  least-squares for ``T_camera_base`` (and the EE->tag offset).

The AprilTag mode is the one used in practice. It minimizes the SE(3) residual
of ``T_camera_tag ~= T_camera_base @ T_base_ee @ T_ee_tag`` over all snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as SciRot

from .fk import FR3ForwardKinematics
from .geometry import (
    invert_transform,
    make_transform,
    matrix_to_rpy,
    pose_xyz_rpy_to_transform,
    rpy_to_matrix,
    se3_error,
    transform_points,
    transform_to_vec6,
    vec6_to_transform,
)


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load_camera_info_yaml(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    k = np.asarray(data["k"], dtype=np.float64).reshape(3, 3)
    d = np.asarray(data.get("d", []), dtype=np.float64).reshape(-1, 1)
    meta = {
        "width": int(data.get("width", 0)),
        "height": int(data.get("height", 0)),
        "distortion_model": data.get("distortion_model", ""),
        "frame_id": data.get("header", {}).get("frame_id", ""),
    }
    return k, d, meta


def load_manual_correspondences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points_3d = np.asarray(data["points_3d_base"], dtype=np.float64).reshape(-1, 3)
    points_2d = np.asarray(data["points_2d_image"], dtype=np.float64).reshape(-1, 2)
    validate_correspondences(points_3d, points_2d)
    return points_3d, points_2d


def validate_correspondences(points_3d: np.ndarray, points_2d: np.ndarray) -> None:
    if len(points_3d) != len(points_2d):
        raise ValueError("points_3d_base and points_2d_image must have the same length")
    if len(points_3d) < 4:
        raise ValueError("At least 4 correspondences are required for solvePnP")


def load_aruco_samples(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("aruco_samples_json must contain a non-empty list of samples")
    return samples


def load_apriltag_joint_snapshots(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    snapshots = data["snapshots"] if isinstance(data, dict) and "snapshots" in data else data
    if not isinstance(snapshots, list) or len(snapshots) == 0:
        raise ValueError("apriltag_joint_samples_json must contain a non-empty snapshots list")
    return snapshots


# Snapshot keys that may hold the robot's joint state, newest convention first.
_JOINT_STATE_KEYS = ("robot", "franka_right_arm", "arm", "robot_state")


def extract_joint_positions(sample: dict) -> np.ndarray:
    """Pull ``joint_positions`` from a snapshot regardless of which arm wrote it."""
    for key in _JOINT_STATE_KEYS:
        block = sample.get(key)
        if isinstance(block, dict) and "joint_positions" in block:
            return np.asarray(block["joint_positions"], dtype=np.float64).reshape(-1)
    if "joint_positions" in sample:  # flat fallback
        return np.asarray(sample["joint_positions"], dtype=np.float64).reshape(-1)
    return np.empty(0, dtype=np.float64)


# --------------------------------------------------------------------------
# ArUco detection
# --------------------------------------------------------------------------
def _get_aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise ImportError("This OpenCV build does not include cv2.aruco")
    key = f"DICT_{name.upper()}"
    if not hasattr(cv2.aruco, key):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key))


def detect_marker_corners(image_bgr: np.ndarray, marker_id: int, dictionary_name: str):
    dictionary = _get_aruco_dictionary(dictionary_name)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None:
        return None
    ids = ids.reshape(-1)
    for idx, found_id in enumerate(ids):
        if int(found_id) == int(marker_id):
            return np.asarray(corners[idx], dtype=np.float64).reshape(4, 2)
    return None


def marker_corners_in_marker_frame(marker_size_m: float) -> np.ndarray:
    half = marker_size_m * 0.5
    return np.asarray(
        [
            [-half, half, 0.0],   # top-left
            [half, half, 0.0],    # top-right
            [half, -half, 0.0],   # bottom-right
            [-half, -half, 0.0],  # bottom-left
        ],
        dtype=np.float64,
    )


def build_ee_to_marker_transform(translation_xyz: np.ndarray, rpy_rad: np.ndarray) -> np.ndarray:
    return make_transform(rpy_to_matrix(*rpy_rad), translation_xyz)


# --------------------------------------------------------------------------
# Hand-eye residuals (AprilTag mode)
# --------------------------------------------------------------------------
def _build_initial_handeye_guess(T_base_ee_list, T_camera_tag_list, T_ee_tag_0=None) -> np.ndarray:
    if T_ee_tag_0 is None:
        T_ee_tag_0 = np.eye(4, dtype=np.float64)
    T_camera_base_0 = T_camera_tag_list[0] @ invert_transform(T_base_ee_list[0] @ T_ee_tag_0)
    return np.hstack([transform_to_vec6(T_camera_base_0), transform_to_vec6(T_ee_tag_0)])


def _handeye_residuals(x, T_base_ee_list, T_camera_tag_list) -> np.ndarray:
    T_camera_base = vec6_to_transform(x[:6])
    T_ee_tag = vec6_to_transform(x[6:12])
    return np.concatenate([
        se3_error(T_camera_tag, T_camera_base @ T_base_ee @ T_ee_tag)
        for T_base_ee, T_camera_tag in zip(T_base_ee_list, T_camera_tag_list)
    ])


def _handeye_residuals_fixed_ee_tag(x, T_base_ee_list, T_camera_tag_list, T_ee_tag) -> np.ndarray:
    T_camera_base = vec6_to_transform(x[:6])
    return np.concatenate([
        se3_error(T_camera_tag, T_camera_base @ T_base_ee @ T_ee_tag)
        for T_base_ee, T_camera_tag in zip(T_base_ee_list, T_camera_tag_list)
    ])


def _handeye_residuals_fixed_ee_tag_translation(x, T_base_ee_list, T_camera_tag_list, ee_tag_translation):
    T_camera_base = vec6_to_transform(x[:6])
    ee_tag_rotation = SciRot.from_rotvec(np.asarray(x[6:9], dtype=np.float64)).as_matrix()
    T_ee_tag = make_transform(ee_tag_rotation, ee_tag_translation)
    return np.concatenate([
        se3_error(T_camera_tag, T_camera_base @ T_base_ee @ T_ee_tag)
        for T_base_ee, T_camera_tag in zip(T_base_ee_list, T_camera_tag_list)
    ])


def estimate_extrinsic_from_apriltag_snapshots(
    snapshots: list[dict],
    fk: FR3ForwardKinematics,
    tag_id: int | None,
    initial_ee_tag: np.ndarray | None = None,
    fix_ee_tag: bool = False,
    fix_ee_tag_translation: bool = False,
) -> dict:
    T_base_ee_list: list[np.ndarray] = []
    T_camera_tag_list: list[np.ndarray] = []
    used_samples: list[dict] = []

    expected_joints = fk.num_joints
    for sample_idx, sample in enumerate(snapshots):
        joints = extract_joint_positions(sample)
        if joints.size != expected_joints:
            used_samples.append({
                "sample_index": sample_idx, "used": False,
                "reason": f"joint_positions must contain {expected_joints} values, got {joints.size}",
            })
            continue

        tags = sample.get("april_tags", [])
        selected_tag = None
        if tag_id is None:
            if tags:
                selected_tag = tags[0]
        else:
            for tag in tags:
                if int(tag.get("tag_id", -1)) == int(tag_id):
                    selected_tag = tag
                    break

        if selected_tag is None:
            used_samples.append({
                "sample_index": sample_idx, "used": False,
                "reason": "requested tag not found in april_tags",
            })
            continue

        position = selected_tag["position_m"]
        orientation = selected_tag["orientation_deg"]
        T_base_ee = fk.compute(joints)
        T_camera_tag = pose_xyz_rpy_to_transform(
            position["x"], position["y"], position["z"],
            orientation["roll"], orientation["pitch"], orientation["yaw"],
            degrees=True,
        )

        T_base_ee_list.append(T_base_ee)
        T_camera_tag_list.append(T_camera_tag)
        used_samples.append({
            "sample_index": sample_idx, "used": True,
            "tag_id": int(selected_tag["tag_id"]),
            "joint_positions": joints.tolist(),
            "T_base_ee": T_base_ee.tolist(),
            "T_camera_tag": T_camera_tag.tolist(),
        })

    if len(T_base_ee_list) < 2:
        raise RuntimeError("At least 2 valid AprilTag+joint snapshots are required")

    x0_full = _build_initial_handeye_guess(T_base_ee_list, T_camera_tag_list, initial_ee_tag)
    method = "lm" if len(T_base_ee_list) * 6 >= 12 else "trf"

    if fix_ee_tag:
        if initial_ee_tag is None:
            raise ValueError("fix_ee_tag=True requires an explicit initial_ee_tag transform")
        result = least_squares(
            _handeye_residuals_fixed_ee_tag, x0_full[:6],
            args=(T_base_ee_list, T_camera_tag_list, initial_ee_tag),
            method=method, max_nfev=5000,
        )
        T_camera_base = vec6_to_transform(result.x[:6])
        T_ee_tag = initial_ee_tag.copy()
    elif fix_ee_tag_translation:
        if initial_ee_tag is None:
            raise ValueError("fix_ee_tag_translation=True requires an explicit initial_ee_tag transform")
        x0_partial = np.hstack([x0_full[:6], transform_to_vec6(initial_ee_tag)[3:6]])
        result = least_squares(
            _handeye_residuals_fixed_ee_tag_translation, x0_partial,
            args=(T_base_ee_list, T_camera_tag_list, initial_ee_tag[:3, 3].copy()),
            method=method, max_nfev=5000,
        )
        T_camera_base = vec6_to_transform(result.x[:6])
        T_ee_tag = make_transform(
            SciRot.from_rotvec(np.asarray(result.x[6:9], dtype=np.float64)).as_matrix(),
            initial_ee_tag[:3, 3].copy(),
        )
    else:
        result = least_squares(
            _handeye_residuals, x0_full,
            args=(T_base_ee_list, T_camera_tag_list),
            method=method, max_nfev=5000,
        )
        T_camera_base = vec6_to_transform(result.x[:6])
        T_ee_tag = vec6_to_transform(result.x[6:12])

    T_base_camera = invert_transform(T_camera_base)

    trans_err_mm, rot_err_deg = [], []
    for T_base_ee, T_camera_tag in zip(T_base_ee_list, T_camera_tag_list):
        err6 = se3_error(T_camera_tag, T_camera_base @ T_base_ee @ T_ee_tag)
        trans_err_mm.append(float(np.linalg.norm(err6[:3]) * 1000.0))
        rot_err_deg.append(float(np.linalg.norm(err6[3:]) * 180.0 / np.pi))

    return {
        "mode": "apriltag_joint_snapshots",
        "num_valid_samples": len(T_base_ee_list),
        "ee_tag_fixed": bool(fix_ee_tag),
        "ee_tag_translation_fixed": bool(fix_ee_tag_translation),
        "sample_details": used_samples,
        "T_camera_base": T_camera_base.tolist(),
        "T_base_camera": T_base_camera.tolist(),
        "T_ee_tag": T_ee_tag.tolist(),
        "translation_base_camera_xyz": T_base_camera[:3, 3].tolist(),
        "rpy_base_camera_rad": matrix_to_rpy(T_base_camera[:3, :3]).tolist(),
        "rpy_base_camera_deg": np.rad2deg(matrix_to_rpy(T_base_camera[:3, :3])).tolist(),
        "handeye_translation_error_mm": {
            "mean": float(np.mean(trans_err_mm)),
            "max": float(np.max(trans_err_mm)),
            "per_sample": trans_err_mm,
        },
        "handeye_rotation_error_deg": {
            "mean": float(np.mean(rot_err_deg)),
            "max": float(np.max(rot_err_deg)),
            "per_sample": rot_err_deg,
        },
        "optimization": {
            "success": bool(result.success),
            "cost": float(result.cost),
            "message": str(result.message),
            "nfev": int(result.nfev),
        },
    }


# --------------------------------------------------------------------------
# ArUco / PnP modes
# --------------------------------------------------------------------------
def collect_aruco_correspondences(
    samples: list[dict],
    fk: FR3ForwardKinematics,
    marker_id: int,
    marker_size_m: float,
    ee_to_marker_transform: np.ndarray,
    aruco_dict: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    marker_corners_marker = marker_corners_in_marker_frame(marker_size_m)
    all_points_3d, all_points_2d, per_sample = [], [], []

    for sample_idx, sample in enumerate(samples):
        image_path = Path(sample["image_path"])
        joint_positions = np.asarray(sample["joint_positions"], dtype=np.float64).reshape(-1)
        if joint_positions.size != fk.num_joints:
            raise ValueError(
                f"Sample {sample_idx}: joint_positions must contain {fk.num_joints} values"
            )

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        corners_2d = detect_marker_corners(image_bgr, marker_id=marker_id, dictionary_name=aruco_dict)
        if corners_2d is None:
            per_sample.append({
                "sample_index": sample_idx, "image_path": str(image_path),
                "used": False, "reason": f"marker_id {marker_id} not detected",
            })
            continue

        T_base_ee = fk.compute(joint_positions)
        T_base_marker = T_base_ee @ ee_to_marker_transform
        corners_3d_base = transform_points(T_base_marker, marker_corners_marker)

        all_points_3d.append(corners_3d_base)
        all_points_2d.append(corners_2d)
        per_sample.append({
            "sample_index": sample_idx, "image_path": str(image_path), "used": True,
            "joint_positions": joint_positions.tolist(),
            "corners_2d_image": corners_2d.tolist(),
            "corners_3d_base": corners_3d_base.tolist(),
            "T_base_ee": T_base_ee.tolist(),
            "T_base_marker": T_base_marker.tolist(),
        })

    if not all_points_3d:
        raise RuntimeError("No valid ArUco detections were collected")

    points_3d = np.concatenate(all_points_3d, axis=0)
    points_2d = np.concatenate(all_points_2d, axis=0)
    validate_correspondences(points_3d, points_2d)
    return points_3d, points_2d, per_sample


def estimate_extrinsic_pnp(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    points_3d_base: np.ndarray,
    points_2d_image: np.ndarray,
) -> dict:
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=points_3d_base, imagePoints=points_2d_image,
        cameraMatrix=camera_matrix, distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("cv2.solvePnP failed to estimate extrinsic")

    rotation_camera_base, _ = cv2.Rodrigues(rvec)
    T_camera_base = make_transform(rotation_camera_base, tvec.reshape(3))
    T_base_camera = np.linalg.inv(T_camera_base)

    reprojected, _ = cv2.projectPoints(
        objectPoints=points_3d_base, rvec=rvec, tvec=tvec,
        cameraMatrix=camera_matrix, distCoeffs=dist_coeffs,
    )
    reproj_err = np.linalg.norm(reprojected.reshape(-1, 2) - points_2d_image, axis=1)
    rpy_base_camera = matrix_to_rpy(T_base_camera[:3, :3])

    return {
        "T_base_camera": T_base_camera.tolist(),
        "T_camera_base": T_camera_base.tolist(),
        "translation_base_camera_xyz": T_base_camera[:3, 3].tolist(),
        "rpy_base_camera_rad": rpy_base_camera.tolist(),
        "rpy_base_camera_deg": np.rad2deg(rpy_base_camera).tolist(),
        "mean_reprojection_error_px": float(np.mean(reproj_err)),
        "max_reprojection_error_px": float(np.max(reproj_err)),
        "per_point_reprojection_error_px": reproj_err.tolist(),
    }
