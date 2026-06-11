# handeye_calib

AprilTag-based **hand-eye calibration** for a Franka **FR3** (7-axis) and a
Doosan **A0509** (6-axis) — from data collection to the base→camera extrinsic
(`T_base_camera`). Pick the arm with `--robot {fr3,a0509}`; add more arms by
registering a preset in [`src/handeye_calib/robots.py`](src/handeye_calib/robots.py).

A planar tag (AprilTag `tag36h11`) is rigidly mounted on the robot's end
effector. As the arm moves, a fixed RealSense camera observes the tag while
ROS2 `/joint_states` is logged. Forward kinematics gives `T_base_ee` at every
pose; the tag detector gives `T_camera_tag`. We then solve the hand-eye
relation

```
T_camera_tag  ≈  T_camera_base · T_base_ee · T_ee_tag
```

for the unknown camera mounting `T_camera_base` (and the EE→tag offset
`T_ee_tag`) via a nonlinear SE(3) least-squares fit, and report
`T_base_camera = inv(T_camera_base)`.

## Pipeline

```
  ┌─────────────────────────┐  JSON  ┌────────────────────────────┐  JSON  ┌──────────────────────────┐
  │ 1. record_snapshots.py  │ ─────► │ 2. estimate_extrinsic.py   │ ─────► │ 3. click_test.py         │
  │  RealSense + AprilTag    │        │  FK + least-squares /      │        │  click a pixel → verify  │
  │  + ROS2 /joint_states    │        │  solvePnP → T_base_camera  │        │  base xyz vs reality     │
  └─────────────────────────┘        └────────────────────────────┘        └──────────────────────────┘
```

## Layout

```
handeye_calib/
├── scripts/
│   ├── record_snapshots.py      # 1. data collection (RealSense + AprilTag + ROS2)
│   ├── estimate_extrinsic.py    # 2. extrinsic solver (3 modes)
│   └── click_test.py            # 3. click-a-pixel calibration verification
├── src/handeye_calib/
│   ├── robots.py                # robot presets (fr3, a0509): URDF, links, joints
│   ├── geometry.py              # SE(3) / rotation helpers (shared)
│   ├── fk.py                    # URDFForwardKinematics (joint count from URDF)
│   ├── detection.py             # AprilTag pose tracking + drawing
│   ├── recorder.py              # interactive recorder loop
│   ├── solver.py                # apriltag / aruco / manual estimation
│   └── projection.py            # pixel <-> camera <-> base helpers (click test)
├── assets/urdf/fr3.urdf         # bundled FR3 URDF (7-axis)
├── assets/urdf/a0509.urdf       # bundled Doosan A0509 URDF (6-axis)
├── configs/camera_info.example.yaml
├── data/sample/                 # example snapshot JSON
├── requirements.txt
└── pyproject.toml
```

## Install

The **solver** is pure Python and runs in any environment:

```bash
pip install -r requirements.txt        # numpy, scipy, opencv-python, pyyaml
# or: pip install -e .
```

**Data collection** needs hardware/ROS deps that usually live in a dedicated
conda env (e.g. a `realsense_ros2` env):

- `pyrealsense2` — RealSense color stream
- `pupil-apriltags` — tag detection
- a sourced **ROS2** install providing `rclpy` + `sensor_msgs` (`source /opt/ros/<distro>/setup.bash`)

These are imported lazily, so FK and the solver import fine without them.

## 1. Collect snapshots

Mount the tag on the EE, point the camera at the workspace, then:

```bash
# Franka FR3 (subscribes to fr3_joint1..7 on /joint_states)
python scripts/record_snapshots.py --robot fr3 \
    --output-dir data/snapshots --tag-size 0.04 --tag-family tag36h11

# Doosan A0509 (subscribes to joint_1..6 on /joint_states)
python scripts/record_snapshots.py --robot a0509 \
    --output-dir data/snapshots --tag-size 0.04
```

`--robot` picks the joint names and joint count (FR3 = 7-axis, A0509 = 6-axis).
If your Doosan publishes joint states on a namespaced topic, pass
`--joint-topic /dsr01/joint_states`.

In the OpenCV window:

- **`s`** — save a snapshot (current tag poses + 7 FR3 joint positions)
- **`q`** — write all snapshots to `data/snapshots/apriltag_franka_snapshots_<ts>.json` and quit

Aim for **10–20+ snapshots** spanning varied arm orientations and tag depths;
diverse rotations make the hand-eye fit well-conditioned. Camera intrinsics are
read live from the RealSense device, so no `camera_info.yaml` is needed here.

Snapshot JSON schema (one entry per `s` press):

```json
{
  "snapshots": [
    {
      "april_tags": [
        {"tag_id": 1,
         "position_m": {"x": 0.1, "y": -0.0, "z": 0.4},
         "orientation_deg": {"roll": 3.0, "pitch": -1.0, "yaw": 90.0}}
      ],
      "franka_right_arm": {"joint_positions": [j1, j2, j3, j4, j5, j6, j7]}
    }
  ]
}
```

## 2. Estimate the extrinsic

**AprilTag mode (default / recommended):**

```bash
# FR3
python scripts/estimate_extrinsic.py --robot fr3 \
    --apriltag_joint_samples_json data/sample/apriltag_franka_snapshots_example.json \
    --apriltag_id 1 \
    --output_json data/results/extrinsic.json \
    --full_output_json data/results/extrinsic_full.json

# A0509 (same flags, just --robot a0509)
python scripts/estimate_extrinsic.py --robot a0509 \
    --apriltag_joint_samples_json data/snapshots/apriltag_a0509_snapshots_*.json \
    --apriltag_id 1 --output_json data/results/extrinsic_a0509.json
```

The solver derives the joint count from the selected robot's URDF, so the same
command works for the 7-axis FR3 and the 6-axis A0509.

`--output_json` holds the minimal `{ "T_base_camera": [[...]] }` for downstream
nodes. `--full_output_json` (optional) keeps the per-sample residuals and the
fitted `T_ee_tag`. Useful options:

- `--apriltag_id -1` — use the first tag found in each snapshot (single-tag setups).
- `--apriltag_ee_tag_xyz X,Y,Z --apriltag_ee_tag_rpy_deg R,P,Y` — seed a known
  EE→tag mount.
- `--fix_apriltag_ee_tag` — keep that EE→tag fixed and solve only `T_camera_base`.
- `--fix_apriltag_ee_tag_translation` — fix the EE→tag translation, solve its rotation.

Check the printed `handeye_translation_error_mm` / `handeye_rotation_error_deg`:
low mean residuals (a few mm / sub-degree) indicate a good fit. Large residuals
usually mean too few or too-similar poses, a wrong `--tag-size`, or a bad tag
mount.

**ArUco mode** — an ArUco marker on the EE, captured images + joints, solved by
FK + `solvePnP` (needs `--camera_info_yaml`):

```bash
python scripts/estimate_extrinsic.py \
    --aruco_samples_json data/aruco_samples.json \
    --camera_info_yaml configs/camera_info.example.yaml \
    --marker_id 0 --marker_size_m 0.04 --aruco_dict 4X4_50 \
    --ee_to_marker_xyz 0,0,0.05 \
    --output_json data/results/extrinsic.json
```

**Manual mode** — explicit 3D(base)↔2D(image) correspondences + `solvePnP`:

```bash
python scripts/estimate_extrinsic.py \
    --correspondences_json data/correspondences.json \
    --camera_info_yaml configs/camera_info.example.yaml \
    --output_json data/results/extrinsic.json
```

## 3. Verify with the click test

Sanity-check the result on the live camera: click a pixel and read off where it
lands in the **robot base frame**, then compare against a known point (e.g. a
corner of the table, the robot base, a ruler mark).

```bash
python scripts/click_test.py \
    --calibration_result_path data/results/extrinsic.json
```

- Streams aligned color + depth; left-click any pixel.
- The clicked pixel is deprojected with its measured depth into the camera
  frame, then mapped to base via `T_base_camera`; both `camera xyz` and
  `base xyz` print to the console and the on-screen panel.
- `--depth_patch_radius` median-filters depth around the click (default 2 px);
  `--serial_number` selects a specific RealSense; `q` quits.

Robot-agnostic — it only consumes `T_base_camera`, so the same tool verifies FR3
and A0509 results. Needs `pyrealsense2` + `opencv` (the collection env).

## Conventions

- Transforms are 4×4 homogeneous matrices, column-vector convention
  (`p_out = T @ p_in`).
- Rotations use the URDF/ROS intrinsic `xyz` roll-pitch-yaw convention.
- `T_camera_tag` from `pupil_apriltags` is the tag in the **camera optical
  frame**.
- FK runs the selected robot's `base_link` → `end_link` (FR3: `fr3_link0` →
  `fr3_link8`; A0509: `base_link` → `link_6`). Override the URDF with `--urdf`,
  or register a new preset in `robots.py`.

## Notes

- The bundled `assets/urdf/fr3.urdf` drives FK. Override with `--urdf` if your
  robot uses a modified chain.
- `data/snapshots/` and `data/results/` are git-ignored; `data/sample/` is kept
  for reference.
