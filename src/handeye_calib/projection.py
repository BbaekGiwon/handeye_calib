"""Pixel <-> camera <-> base projection helpers for verifying a calibration.

These are used by the click test: deproject a clicked pixel with its depth into
the camera optical frame, then map it into the robot base frame with the
``T_base_camera`` produced by ``estimate_extrinsic.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_base_to_camera_transform(calibration_result_path: str | Path) -> np.ndarray:
    """Load the 4x4 ``T_base_camera`` from a solver result JSON."""
    with open(calibration_result_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    transform = np.asarray(data["T_base_camera"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"T_base_camera must be 4x4, got {transform.shape}")
    return transform


def backproject_pixel_to_camera_xyz(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """Deproject pixel ``(u, v)`` at ``depth_m`` to a point in the camera frame."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (float(u) - cx) * depth_m / fx
    y = (float(v) - cy) * depth_m / fy
    return np.array([x, y, depth_m], dtype=np.float64)


def camera_xyz_to_base_xyz(camera_xyz: np.ndarray, T_base_camera: np.ndarray) -> np.ndarray:
    """Map a camera-frame point into the robot base frame."""
    camera_xyz_h = np.ones(4, dtype=np.float64)
    camera_xyz_h[:3] = np.asarray(camera_xyz, dtype=np.float64).reshape(3)
    return (T_base_camera @ camera_xyz_h)[:3]


def estimate_depth_m(depth_m: np.ndarray, u: int, v: int, radius: int) -> float:
    """Median of valid depths in a ``(2*radius+1)`` patch around ``(u, v)``."""
    h, w = depth_m.shape[:2]
    x0, x1 = max(0, u - radius), min(w, u + radius + 1)
    y0, y1 = max(0, v - radius), min(h, v + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def make_depth_vis(depth_raw: np.ndarray) -> np.ndarray:
    """Colorize a raw uint16 depth image for display."""
    import cv2

    depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
