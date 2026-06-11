"""RealSense + AprilTag + Franka joint-state snapshot recorder.

Live-detects AprilTag poses with ``pupil_apriltags`` over a RealSense color
stream while subscribing to ROS2 ``/joint_states`` for the FR3 7-axis joint
positions. Press ``s`` to capture a snapshot (tag poses + joints) and ``q`` to
write all snapshots to a timestamped JSON the solver can consume.

ROS2 (``rclpy``), ``pyrealsense2`` and ``pupil_apriltags`` are imported lazily so
the rest of the package (FK, solver) stays importable without them.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from .detection import TagPositionState, draw_tag_and_position

# --- defaults -------------------------------------------------------------
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.04
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30
INFO_PANEL_H = 130
FR3_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]


def _import_ros():
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    return rclpy, MultiThreadedExecutor, Node, qos_profile_sensor_data, JointState


def make_franka_subscriber(joint_names: list[str], topic: str = "/joint_states"):
    """Build a ROS2 node subscribing to ``topic`` for the named FR3 joints."""
    rclpy, _, Node, qos_profile_sensor_data, JointState = _import_ros()

    class FrankaJointStateSubscriber(Node):
        def __init__(self) -> None:
            super().__init__("handeye_franka_state_subscriber")
            self._lock = threading.Lock()
            self._joint_names = list(joint_names)
            self._joint_positions = np.zeros(len(joint_names), dtype=np.float64)
            self._joint_torques = np.zeros(len(joint_names), dtype=np.float64)
            self._received = False
            self._recv_count = 0
            self._last_stamp_sec: Optional[int] = None
            self._last_stamp_nanosec: Optional[int] = None

            self.create_subscription(JointState, topic, self._callback, qos_profile_sensor_data)
            self.get_logger().info(f"Subscribe: {topic}")

        def _callback(self, msg) -> None:
            joint_map = {name: idx for idx, name in enumerate(msg.name)}
            if any(name not in joint_map for name in self._joint_names):
                return
            with self._lock:
                self._joint_positions[:] = [
                    float(msg.position[joint_map[name]]) for name in self._joint_names
                ]
                if len(msg.effort) > 0:
                    self._joint_torques[:] = [
                        float(msg.effort[joint_map[name]]) if joint_map[name] < len(msg.effort) else 0.0
                        for name in self._joint_names
                    ]
                else:
                    self._joint_torques[:] = 0.0
                self._received = True
                self._recv_count += 1
                self._last_stamp_sec = int(msg.header.stamp.sec)
                self._last_stamp_nanosec = int(msg.header.stamp.nanosec)

        def latest_state(self) -> dict:
            with self._lock:
                return {
                    "received": self._received,
                    "receive_count": self._recv_count,
                    "arm_id": 0,
                    "joint_names": list(self._joint_names),
                    "joint_positions": self._joint_positions.tolist(),
                    "joint_torques": self._joint_torques.tolist(),
                    "source_topic": topic,
                    "source_stamp": {
                        "sec": self._last_stamp_sec,
                        "nanosec": self._last_stamp_nanosec,
                    },
                }

    return FrankaJointStateSubscriber()


def draw_info_panel(canvas, fps, sample_count, latest_franka, last_status, width) -> None:
    height = canvas.shape[0]
    panel_top = height - INFO_PANEL_H
    cv2.rectangle(canvas, (0, panel_top), (width, height), (30, 30, 30), -1)
    cv2.line(canvas, (0, panel_top), (width, panel_top), (100, 100, 100), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y0 = panel_top + 20
    cv2.putText(canvas, f"FPS: {fps:.1f}", (10, y0), font, 0.5, (220, 220, 220), 1)
    cv2.putText(canvas, f"Saved samples: {sample_count}", (110, y0), font, 0.5, (220, 220, 220), 1)

    franka_ok = latest_franka["received"]
    franka_text = "Franka: connected" if franka_ok else "Franka: waiting for joint_states"
    franka_color = (0, 220, 0) if franka_ok else (0, 180, 255)
    cv2.putText(canvas, franka_text, (10, y0 + 24), font, 0.45, franka_color, 1)
    if franka_ok:
        joint_text = "q: [" + ", ".join(f"{v:.3f}" for v in latest_franka["joint_positions"]) + "]"
        cv2.putText(canvas, joint_text, (10, y0 + 46), font, 0.38, (200, 200, 200), 1)

    cv2.putText(canvas, "Keys: s=save snapshot, q=save json and quit",
                (10, y0 + 68), font, 0.45, (180, 180, 180), 1)
    cv2.putText(canvas, last_status, (10, y0 + 92), font, 0.42, (255, 255, 0), 1)


def build_snapshot(tag_states, robot_state, snapshot_key) -> dict:
    now = time.time()
    tracked_tags = [
        tag_states[tag_id].to_dict()
        for tag_id in sorted(tag_states.keys())
        if tag_states[tag_id].tracking
    ]
    return {
        "saved_at_unix": now,
        "saved_at_iso": datetime.fromtimestamp(now).isoformat(),
        "april_tags": tracked_tags,
        snapshot_key: robot_state,
    }


def save_records_as_json(records, output_dir, output_prefix, tag_size, tag_family, robot_name, joint_topic) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{output_prefix}_{timestamp}.json")
    payload = {
        "created_at_iso": datetime.now().isoformat(),
        "robot": robot_name,
        "camera": {"frame_width": FRAME_WIDTH, "frame_height": FRAME_HEIGHT, "frame_fps": FRAME_FPS},
        "apriltag": {"family": tag_family, "tag_size_m": tag_size},
        "ros_topic": joint_topic,
        "snapshot_count": len(records),
        "snapshots": records,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return output_path


def run_recorder(
    *,
    joint_names: list[str] = None,
    robot_name: str = "fr3",
    snapshot_key: str = "robot",
    tag_size: float = TAG_SIZE_M,
    tag_family: str = TAG_FAMILY,
    smooth_alpha: float = 0.6,
    output_dir: str,
    output_prefix: str = "apriltag_franka_snapshots",
    joint_topic: str = "/joint_states",
) -> Optional[str]:
    """Run the interactive recorder loop. Returns the written JSON path."""
    import pyrealsense2 as rs
    from pupil_apriltags import Detector

    if joint_names is None:
        joint_names = FR3_JOINT_NAMES

    rclpy, MultiThreadedExecutor, _, _, _ = _import_ros()

    rclpy.init()
    ros_node = make_franka_subscriber(joint_names, topic=joint_topic)
    executor = MultiThreadedExecutor()
    executor.add_node(ros_node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    detector = Detector(
        families=tag_family,
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=0,
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FRAME_FPS)
    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_params = (intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

    tag_states: dict[int, TagPositionState] = {}
    saved_records: list[dict] = []
    prev_time = time.time()
    fps_alpha = 0.2
    fps_smoothed = float(FRAME_FPS)
    last_status = "Waiting... press 's' to save snapshot."
    output_path: Optional[str] = None

    print(f"[INFO] RealSense: {FRAME_WIDTH}x{FRAME_HEIGHT}@{FRAME_FPS}fps")
    print(f"[INFO] AprilTag family={tag_family}, tag_size={tag_size}m")
    print("[INFO] Keys: 's'=save current snapshot, 'q'=save json and quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0.0:
                fps_smoothed = (1.0 - fps_alpha) * fps_smoothed + fps_alpha * (1.0 / dt)

            detections = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=tag_size,
            )

            detected_ids = set()
            for detection in detections:
                tag_id = int(detection.tag_id)
                if tag_id not in tag_states:
                    tag_states[tag_id] = TagPositionState(tag_id=tag_id, smooth_alpha=smooth_alpha)
                tag_states[tag_id].update(detection.pose_t, detection.pose_R)
                draw_tag_and_position(image, detection, tag_states[tag_id])
                detected_ids.add(tag_id)

            for tag_id in list(tag_states.keys()):
                if tag_id not in detected_ids:
                    tag_states[tag_id].lost()

            panel = np.full((INFO_PANEL_H, FRAME_WIDTH, 3), (30, 30, 30), dtype=np.uint8)
            canvas = np.vstack([image, panel])
            latest_franka = ros_node.latest_state()
            draw_info_panel(canvas, fps_smoothed, len(saved_records), latest_franka, last_status, FRAME_WIDTH)

            cv2.imshow("AprilTag + Franka Snapshot Recorder", canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                snapshot = build_snapshot(tag_states, latest_franka, snapshot_key)
                saved_records.append(snapshot)
                last_status = (
                    f"Saved #{len(saved_records)} | tags={len(snapshot['april_tags'])} "
                    f"| joints_received={latest_franka['received']}"
                )
                print(f"[SAVE] {last_status}")
            elif key == ord("q"):
                output_path = save_records_as_json(
                    saved_records, output_dir, output_prefix, tag_size, tag_family, robot_name, joint_topic
                )
                print(f"[INFO] JSON saved: {output_path}")
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        executor.shutdown()
        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=1.0)
        if output_path is None:
            output_path = save_records_as_json(
                saved_records, output_dir, output_prefix, tag_size, tag_family, robot_name, joint_topic
            )
            print(f"[INFO] JSON saved on shutdown: {output_path}")

    return output_path
