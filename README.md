# TE3002B — UAV Swarm Control with ArUco Tracking

**Course:** TE3002B — Implementación de Robótica Inteligente  
**Institution:** ITESM Campus Monterrey  
**Professor:** Dr. Herman Castañeda Cuevas

| Name | ID |
|---|---|
| David Gilberto Lomelí Leal | A01571193 |
| Abraham de Jesús Maldonado Mata | A00838581 |
| David Alejandro Soni Cuevas | A01571777 |

---

## Overview

This repository contains two integrated components:

1. **Camera calibration + standalone ArUco detection** — calibrate a DJI Tello camera and test single-drone ArUco tracking without ROS2.
2. **ROS2 swarm control** (`ros2_ws/src/tello_vicon`) — three Tello drones flying in formation, with Vicon motion capture for state estimation, a Kalman filter for smoothing, a PD position controller, and ArUco-based leader tracking.

```
Leader drone detects ArUco marker → follows it at standoff distance
Follower drones maintain geometric formation relative to leader (Vicon)
```

---

## Repository Structure

```
TE3002B_uavSwarm/
│
├── arucos/                        # ArUco utilities
│   ├── arucocreation.py           # Generate ArUco markers (DICT_4X4_50)
│   ├── calibracion.py             # Camera calibration from checkerboard frames
│   ├── detect_aruco.py            # Standalone ArUco detection (no ROS2)
│   └── test_aruco/                # Test images
│
├── calib_frames/                  # Checkerboard frames for calibration (43 images)
├── calibration/                   # Calibration output files
│
├── python_scripts/                # Standalone utilities
│   ├── battery_status.py          # Check Tello battery over WiFi
│   ├── find_tellos.py             # Scan network for Tello IPs
│   ├── square_motion.py           # Simple square flight test
│   ├── tello_router.py            # Multi-drone WiFi router helper
│   ├── metrics.py                 # Extract flight metrics from ROS2 bags
│   ├── metrics_report.txt         # Last generated metrics report
│   └── metrics_plots.png          # Last generated metrics plots
│
├── ros2_ws/                       # ROS2 workspace
│   └── src/
│       └── tello_vicon/           # Main ROS2 package (see below)
│
├── checkerboard.py                # Generate checkerboard pattern for calibration
├── captureframes.py               # Capture calibration frames from Tello camera
├── checkerboard_9x6.png           # Generated checkerboard (ready to print)
├── requirements.txt               # Python dependencies
├── .gitmodules
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
# or manually:
pip install opencv-contrib-python djitellopy numpy matplotlib
```

> **Important:** use `opencv-contrib-python`, not `opencv-python` — the `contrib` build includes ArUco support.

For the ROS2 package:
```bash
sudo apt install ros-humble-desktop
pip3 install djitellopy
```

---

## Part 1 — Camera Calibration

### Step 0 — Generate patterns (once)

```bash
# Checkerboard for calibration (10×7 squares, 80 px/square)
python checkerboard.py

# ArUco markers (DICT_4X4_50, ID 0 and 1)
python arucos/arucocreation.py
```

Print `checkerboard_9x6.png` and measure the real side of each square in metres — set that value as `SQUARE_SIZE` in `arucos/calibracion.py`.

### Step 1 — Capture frames

```bash
python captureframes.py
```

| Key | Action |
|---|---|
| `s` | Save current frame to `calib_frames/` |
| `q` | Quit |

Capture **at least 20–30 frames** varying tilt, rotation, and distance. The drone must be connected to its own WiFi AP.

### Step 2 — Calibrate

```bash
python arucos/calibracion.py
```

Processes all frames in `calib_frames/`, shows detected corners, and saves `camera_params.npz`.

- RMS reprojection error < 1.0 px → valid calibration.

### Calibration output

| Variable | Description |
|---|---|
| `K` | 3×3 intrinsic camera matrix |
| `dist` | Distortion coefficients (k1, k2, p1, p2, k3) |

```python
import numpy as np
data = np.load("calibration/camera_params.npz")
K, dist = data["K"], data["dist"]
```

---

## Part 2 — Standalone ArUco Tracking (no ROS2)

Located in the root and `arucos/`. Connects directly to one Tello drone via WiFi — no Vicon, no ROS2 required.

### Run

```bash
python arucos/detect_aruco.py
```

Tracks **ArUco ID=1** (DICT_4X4_50). The marker must be printed with a real side length of **20.8 cm**.

| Key | Action |
|---|---|
| `q` | Land and exit |

### Architecture

Three concurrent threads + display loop on the main thread:

```
Main thread  →  display loop (OpenCV imshow)
Thread 1     →  ArucoDetector  — detection + solvePnP
Thread 2     →  RCController   — proportional RC commands every 50 ms
```

### Control parameters

| Parameter | Value | Description |
|---|---|---|
| `TARGET_DIST` | 1.2 m | Desired distance to marker |
| `DEAD_ZONE_D` | 0.08 m | Distance dead zone |
| `Kp_dist` | 60 | Forward/back proportional gain |
| `MAX_VEL_FB` | 30 % | Max forward/back speed |
| `DEAD_ZONE_PX` | 60 px | Pixel dead zone for yaw/up-down |
| `LOST_TIMEOUT` | 0.5 s | Hover if marker not seen for this long |

RC channels: lateral is always 0. Forward/back proportional to distance error. Up/down and yaw proportional to pixel error from frame center.

---

## Part 3 — ROS2 Swarm Control (`ros2_ws/src/tello_vicon`)

### Package structure

```
tello_vicon/
├── scripts/
│   ├── vicon_kf_node.py           # Kalman filter — all drones in one process
│   ├── tello_controller_node.py   # PD position controller — all drones in one process
│   ├── swarm_bridge_node.py       # djitellopy bridge — one UDP socket for all drones
│   ├── formation_controller_node.py  # formation geometry + ArUco reference
│   ├── aruco_node.py              # ArUco detection + world-frame transform + HUD
│   ├── video_recorder_node.py     # compressed video recording
│   ├── kalman_filter.py           # discrete Kalman filter (12-state)
│   └── vicon_viz_node.py          # Foxglove / RViz visualization
├── launch/
│   ├── swarm.launch.py            # main launch — full 3-drone swarm
│   ├── single_drone.launch.py     # single drone testing
│   ├── kf_only.launch.py          # KF only (bag replay)
│   ├── record_bag.launch.py       # record all topics
│   └── viz.launch.py              # Foxglove + RViz
└── config/
    ├── params.yaml                # all tunable parameters
    ├── camera_params.npz          # Tello camera calibration
    └── foxglove_layouts/          # Foxglove Studio panel layouts
```

### Hardware

| Component | Details |
|---|---|
| Drones | DJI Tello × 3 |
| Motion capture | Vicon Tracker |
| Subjects | `tello_soni1`, `tello_soni2`, `tello_soni3` |
| IPs | soni1: `192.168.0.100`, soni2: `192.168.0.101`, soni3: `192.168.0.102` |
| ArUco marker | ID=1, DICT_4X4_50, 20.8 cm side |

The Vicon driver publishes `/vicon/tello_soniN/tello_soniN` — the launch file remaps this automatically to `/vicon/tello_soniN/pose`.

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select tello_vicon
source install/setup.bash
```

### Launch

```bash
# Full swarm (uses default IPs from params.yaml)
ros2 launch tello_vicon swarm.launch.py

# Custom IPs or formation
ros2 launch tello_vicon swarm.launch.py \
  drones:="tello_soni1:192.168.0.100:tello_soni2:192.168.0.101:tello_soni3:192.168.0.102" \
  formation:=V

# Mock mode — no real drones (for bag replay)
ros2 launch tello_vicon swarm.launch.py mock:=true
ros2 bag play <bag_path>   # in another terminal
```

### Launch arguments

| Argument | Default | Description |
|---|---|---|
| `drones` | `tello_soni1:192.168.0.100:...` | Colon-separated `subject:ip` pairs. First = leader. |
| `mock` | `false` | Skip real drones |
| `formation` | `LINE` | `V` \| `LINE` \| `COLUMN` \| `PANORAMIC` \| `RECONSTRUCTION` |
| `record_mode` | `leader` | Video: `leader` \| `all` |
| `output_dir` | `/home/kfcnef/videos` | Video output directory |

### Formations

| Name | S1 offset | S2 offset |
|---|---|---|
| `LINE` | (0, +1.5, 0) m | (0, −1.5, 0) m |
| `V` | (−1.0, +1.0, 0) m | (−1.0, −1.0, 0) m |
| `COLUMN` | (−1.2, 0, +0.4) m | (−2.4, 0, +0.8) m |
| `PANORAMIC` | (0, +1.5, 0) yaw+45° | (0, −1.5, 0) yaw−45° |
| `RECONSTRUCTION` | (−0.8, +1.2, +0.4) m | (−0.8, −1.2, +0.4) m |

Change at runtime:
```bash
ros2 param set /formation_controller formation V
```

### Key topics

| Topic | Type | Description |
|---|---|---|
| `/tello_soniN/kf_state` | `Float64MultiArray` | KF state [px,vx,py,vy,pz,vz,roll,vroll,pitch,vpitch,yaw,vyaw] |
| `/tello_soniN/reference` | `PoseStamped` | Position setpoint |
| `/tello_soniN/rc_cmd` | `Int32MultiArray` | RC command [lr, fb, ud, yaw] |
| `/tello_soniN/image_raw/compressed` | `CompressedImage` | JPEG video (leader only) |
| `/aruco/pose` | `PoseStamped` | Marker pose in Vicon world frame |
| `/tello_soni1/aruco_detected` | `Bool` | True while marker visible |
| `/tello_soni1/aruco_distance` | `Float32` | Distance to marker [m] |

### ArUco behavior

```
ArUco visible   →  leader follows marker at standoff distance (default 1.0 m)
ArUco lost      →  leader hovers freely after 0.5 s timeout
ArUco reappears →  leader resumes tracking immediately
Followers       →  always track formation offset from leader Vicon position
```

### Video recording

```bash
# Switch to recording all drones at runtime
ros2 service call /video/set_mode std_srvs/srv/SetBool "data: true"

# Switch back to leader only
ros2 service call /video/set_mode std_srvs/srv/SetBool "data: false"
```

Videos saved as `<ns>_YYYYMMDD_HHMMSS.mp4` in `output_dir`.

### Visualization

```bash
# Foxglove Studio
ros2 run foxglove_bridge foxglove_bridge
# Open config/foxglove_layouts/swarm.json → connect ws://localhost:8765

# RViz
ros2 launch tello_vicon viz.launch.py
```

---

## Flight Metrics

Extract position and formation error from a recorded bag:

```bash
source /opt/ros/humble/setup.bash

python3 python_scripts/metrics.py ~/ros2_ws/bags/flight_01 \
  --trim-start 5 \
  --trim-end 3 \
  --formation LINE
```

Outputs `metrics_report.txt` and `metrics_plots.png` with:
- RMS, max, P95 position error per drone
- Formation convergence time and error
- ArUco detection stability (streak analysis)
- Distance consistency when marker is visible

---

## Utility Scripts (`python_scripts/`)

| Script | Description |
|---|---|
| `find_tellos.py` | Scan local network and print IPs of all connected Tellos |
| `battery_status.py` | Print battery % for each drone |
| `square_motion.py` | Fly a square pattern — quick sanity check after setup |
| `tello_router.py` | Helper to connect multiple Tellos through a WiFi router |

---

## Common Issues

| Symptom | Fix |
|---|---|
| `Address already in use` (port 8889) | Use `swarm_bridge` — one process owns the UDP socket |
| `Waiting for leader kf_state` | Check Vicon remap in `swarm.launch.py` |
| Followers fly to wrong position | Verify `leader_ns/s1_ns/s2_ns` passed to `formation_controller` in launch |
| Video publish latency ~170ms | Already fixed — uses CompressedImage (JPEG, ~11KB vs 2MB raw) |
| High CPU on vicon_kf (>50%) | Use merged node with `drone_subjects` param (one process for all KFs) |
| Leader keeps moving after ArUco lost | Set `timeout_s: 0.5` in `params.yaml` tello_controller section |
| `opencv-python` has no ArUco | Install `opencv-contrib-python` instead |

---

## License

Apache-2.0