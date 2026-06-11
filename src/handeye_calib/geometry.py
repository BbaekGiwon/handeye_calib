"""SE(3) / rotation helpers shared across FK, detection and the solver.

All transforms are 4x4 homogeneous matrices in column-vector convention
(``p_out = T @ p_in``). Rotations follow the URDF / ROS ``xyz`` RPY convention
(intrinsic roll about x, then pitch about y, then yaw about z).
"""

from __future__ import annotations

import math

import numpy as np

try:  # SciPy is only needed for the rotvec helpers used by the solver.
    from scipy.spatial.transform import Rotation as SciRot
except ImportError:  # pragma: no cover - allow FK/recorder use without SciPy
    SciRot = None


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a rotation matrix from intrinsic xyz roll/pitch/yaw (radians)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Recover intrinsic xyz roll/pitch/yaw (radians) from a rotation matrix."""
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about ``axis`` by ``angle`` radians."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    v = 1.0 - c
    return np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ],
        dtype=np.float64,
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid 4x4 transform."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ translation
    return out


def transform_points(transform: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an ``(N, 3)`` array of points."""
    points_xyz = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((len(points_xyz), 1), dtype=np.float64)
    hom = np.concatenate([points_xyz, ones], axis=1)
    out = (transform @ hom.T).T
    return out[:, :3]


def pose_xyz_rpy_to_transform(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    *,
    degrees: bool,
) -> np.ndarray:
    """Build a transform from a translation and intrinsic xyz Euler angles."""
    if SciRot is None:
        rpy = np.deg2rad([roll, pitch, yaw]) if degrees else np.array([roll, pitch, yaw])
        rotation = rpy_to_matrix(*rpy)
    else:
        rotation = SciRot.from_euler("xyz", [roll, pitch, yaw], degrees=degrees).as_matrix()
    return make_transform(rotation, np.array([x, y, z], dtype=np.float64))


def _require_scipy() -> None:
    if SciRot is None:
        raise ImportError("scipy is required for rotvec-based helpers (pip install scipy)")


def transform_to_vec6(transform: np.ndarray) -> np.ndarray:
    """Pack a transform into ``[tx, ty, tz, rotvec_x, rotvec_y, rotvec_z]``."""
    _require_scipy()
    rotvec = SciRot.from_matrix(transform[:3, :3]).as_rotvec()
    translation = transform[:3, 3]
    return np.hstack([translation, rotvec])


def vec6_to_transform(vec6: np.ndarray) -> np.ndarray:
    _require_scipy()
    translation = np.asarray(vec6[:3], dtype=np.float64)
    rotvec = np.asarray(vec6[3:6], dtype=np.float64)
    rotation = SciRot.from_rotvec(rotvec).as_matrix()
    return make_transform(rotation, translation)


def se3_error(measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """6-vector residual (translation + rotvec) between two transforms."""
    _require_scipy()
    transform_err = invert_transform(measured) @ predicted
    translation_err = transform_err[:3, 3]
    rotation_err = SciRot.from_matrix(transform_err[:3, :3]).as_rotvec()
    return np.hstack([translation_err, rotation_err])
