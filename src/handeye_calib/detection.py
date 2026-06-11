"""AprilTag pose tracking and visualization helpers.

Wraps ``pupil_apriltags`` detection results into a smoothed, drawable
per-tag state. The pose returned by the detector is the tag expressed in the
camera optical frame (``T_camera_tag``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .geometry import matrix_to_rpy

DEFAULT_POSITION_SMOOTH_ALPHA = 0.6

TAG_COLORS = [
    (0, 255, 0),
    (255, 165, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (128, 0, 255),
]


def get_tag_color(tag_id: int) -> tuple[int, int, int]:
    return TAG_COLORS[tag_id % len(TAG_COLORS)]


@dataclass
class TagPositionState:
    """Smoothed pose of a single tag in the camera optical frame."""

    tag_id: int
    smooth_alpha: float = DEFAULT_POSITION_SMOOTH_ALPHA
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    tracking: bool = False
    _smoothed: Optional[np.ndarray] = None

    def update(self, pose_t: np.ndarray, pose_R: np.ndarray) -> None:
        self.tracking = True
        translation = np.array(pose_t, dtype=float).reshape(3)
        rotation = np.array(pose_R, dtype=float).reshape(3, 3)

        if self._smoothed is None:
            self._smoothed = translation.copy()
        else:
            self._smoothed = (
                (1.0 - self.smooth_alpha) * self._smoothed
                + self.smooth_alpha * translation
            )

        self.x, self.y, self.z = (float(v) for v in self._smoothed)

        roll, pitch, yaw = matrix_to_rpy(rotation)
        self.roll_deg = math.degrees(roll)
        self.pitch_deg = math.degrees(pitch)
        self.yaw_deg = math.degrees(yaw)

    def lost(self) -> None:
        self.tracking = False

    def to_dict(self) -> dict:
        return {
            "tag_id": self.tag_id,
            "position_m": {"x": self.x, "y": self.y, "z": self.z},
            "orientation_deg": {
                "roll": self.roll_deg,
                "pitch": self.pitch_deg,
                "yaw": self.yaw_deg,
            },
            "tracking": self.tracking,
        }


def draw_tag_and_position(image: np.ndarray, detection, state: TagPositionState) -> None:
    corners = np.array(detection.corners, dtype=int).reshape(4, 2)
    color = get_tag_color(state.tag_id)
    for idx in range(4):
        cv2.line(image, tuple(corners[idx]), tuple(corners[(idx + 1) % 4]), color, 2)

    center = corners.mean(axis=0).astype(int)
    cv2.circle(image, tuple(center), 5, (0, 255, 255), -1)

    text1 = f"ID{state.tag_id} x:{state.x:.3f} y:{state.y:.3f} z:{state.z:.3f} m"
    text2 = f"yaw:{state.yaw_deg:.1f} pitch:{state.pitch_deg:.1f} roll:{state.roll_deg:.1f} deg"
    cv2.putText(image, text1, (corners[0][0], corners[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(image, text2, (corners[0][0], corners[0][1] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
