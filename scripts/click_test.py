#!/usr/bin/env python3
"""Interactive click test to verify a hand-eye calibration.

Click any pixel in the live RealSense color (or depth) view; the tool deprojects
it with the measured depth into the camera frame, then maps it into the robot
base frame using ``T_base_camera`` from a solver result JSON. Touch the clicked
base xyz against a known point on the robot/workspace to judge the calibration.

Robot-agnostic: works with any ``estimate_extrinsic.py`` output (FR3 or A0509),
since it only consumes ``T_base_camera``.

Requires pyrealsense2 + opencv (the data-collection env).

Example:
    python scripts/click_test.py --calibration_result_path data/results/extrinsic.json
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from handeye_calib.projection import (  # noqa: E402
    backproject_pixel_to_camera_xyz,
    camera_xyz_to_base_xyz,
    estimate_depth_m,
    load_base_to_camera_transform,
    make_depth_vis,
)


@dataclass
class ClickResult:
    pixel_xy: tuple[int, int]
    depth_m: float
    camera_xyz: np.ndarray
    base_xyz: np.ndarray


def _annotate_click(image: np.ndarray, result: ClickResult | None) -> np.ndarray:
    canvas = image.copy()
    if result is None:
        return canvas
    u, v = result.pixel_xy
    cv2.circle(canvas, (u, v), 6, (0, 255, 255), -1)
    cv2.circle(canvas, (u, v), 12, (0, 0, 0), 2)
    cv2.putText(canvas, f"({u}, {v}) z={result.depth_m:.3f}m", (u + 10, max(25, v - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _build_info_panel(width: int, result: ClickResult | None, status: str) -> np.ndarray:
    panel_h = 180
    panel = np.full((panel_h, width, 3), 28, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 24
    for line in ["Click a pixel to convert it into camera/base coordinates.",
                 "Use stable points on a real surface. Press q to quit."]:
        cv2.putText(panel, line, (10, y), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        y += 24

    if result is None:
        cv2.putText(panel, "No click yet.", (10, y + 12), font, 0.55, (0, 180, 255), 1, cv2.LINE_AA)
    else:
        c, b = result.camera_xyz, result.base_xyz
        cv2.putText(panel, f"pixel={result.pixel_xy} depth={result.depth_m:.4f} m",
                    (10, y + 12), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"camera xyz = [{c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}] m",
                    (10, y + 40), font, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, f"base xyz   = [{b[0]:+.4f}, {b[1]:+.4f}, {b[2]:+.4f}] m",
                    (10, y + 68), font, 0.52, (255, 220, 120), 1, cv2.LINE_AA)

    cv2.putText(panel, status, (10, panel_h - 14), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive click test: pixel -> camera xyz -> robot base xyz using RealSense depth."
    )
    parser.add_argument("--calibration_result_path", required=True,
                        help="Solver result JSON containing T_base_camera.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--serial_number", default="", help="Optional RealSense serial number.")
    parser.add_argument("--depth_patch_radius", type=int, default=2,
                        help="Median depth patch radius around the clicked pixel.")
    return parser


def main() -> None:
    args = build_parser().parse_args()  # parse first so --help works without hardware libs
    import pyrealsense2 as rs

    T_base_camera = load_base_to_camera_transform(args.calibration_result_path)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial_number:
        config.enable_device(args.serial_number)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    state = {
        "rgb_bgr": None, "depth_m": None, "K": None, "result": None,
        "status": f"Calibration: {Path(args.calibration_result_path).resolve()}",
    }

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        rgb_bgr, depth_m, K = state["rgb_bgr"], state["depth_m"], state["K"]
        if rgb_bgr is None or depth_m is None or K is None:
            state["status"] = "Frame not ready yet."
            return

        depth_value_m = estimate_depth_m(depth_m, x, y, radius=max(0, int(args.depth_patch_radius)))
        if depth_value_m <= 0.0:
            state["status"] = f"No valid depth near pixel ({x}, {y})."
            state["result"] = None
            return

        camera_xyz = backproject_pixel_to_camera_xyz(x, y, depth_value_m, K)
        base_xyz = camera_xyz_to_base_xyz(camera_xyz, T_base_camera)
        state["result"] = ClickResult((int(x), int(y)), depth_value_m, camera_xyz, base_xyz)
        state["status"] = (
            f"pixel=({x}, {y}) depth={depth_value_m:.4f}m "
            f"base=[{base_xyz[0]:+.4f}, {base_xyz[1]:+.4f}, {base_xyz[2]:+.4f}] m"
        )
        print(f"pixel: ({x}, {y})  depth_m: {depth_value_m:.6f}")
        print(f"camera_xyz_m: [{camera_xyz[0]:+.6f}, {camera_xyz[1]:+.6f}, {camera_xyz[2]:+.6f}]")
        print(f"base_xyz_m:   [{base_xyz[0]:+.6f}, {base_xyz[1]:+.6f}, {base_xyz[2]:+.6f}]\n")

    window_name = "Hand-eye Click Test (pixel -> base)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        for _ in range(max(0, int(args.warmup_frames))):
            align.process(pipeline.wait_for_frames())

        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            rgb_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                         dtype=np.float64)

            state["rgb_bgr"], state["depth_m"], state["K"] = rgb_bgr, depth_m, K

            color_view = _annotate_click(rgb_bgr, state["result"])
            depth_view = _annotate_click(make_depth_vis(depth_raw), state["result"])
            top_row = np.hstack([color_view, depth_view])
            info_panel = _build_info_panel(top_row.shape[1], state["result"], state["status"])
            cv2.imshow(window_name, np.vstack([top_row, info_panel]))

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
