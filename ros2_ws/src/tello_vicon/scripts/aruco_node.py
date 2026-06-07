"""aruco_node.py — ArUco detection + world-frame transform in one node.

Merges aruco_detection_node and aruco_world_bridge_node.
Runs under the leader drone namespace (e.g. /tello0).

Subscribes
----------
image_raw   (sensor_msgs/Image)        relative → /<ns>/image_raw  from tello_bridge
kf_state    (std_msgs/Float64MultiArray) relative → /<ns>/kf_state   from vicon_kf_node

Publishes
---------
aruco_detected  (std_msgs/Bool)          true while marker is visible
aruco_distance  (std_msgs/Float32)       distance to marker [m]
/aruco/pose     (geometry_msgs/PoseStamped)  marker pose in Vicon world frame
                                             consumed by formation_controller

Pipeline (all inside one process)
----------------------------------
image_raw → [detect thread] → solvePnP → camera-frame tvec/rvec
           + kf_state (leader pos + yaw)
           → R_cam_to_body · R_body_to_world · p_cam + p_leader
           → /aruco/pose  (world frame, ready for formation_controller)

Parameters
----------
marker_size         float   ArUco marker side length [m]       (default 0.185)
marker_id           int     ArUco ID to track                  (default 1)
detect_scale        float   downscale factor for detection     (default 0.4)
camera_params_file  str     path to .npz with K and dist       (required)
publish_rate        float   output publish rate [Hz]           (default 30.0)
image_topic         str     image topic name (relative)        (default image_raw)
target_z_override   float   fix world-Z to this height [m];
                            -1.0 = use detected Z              (default -1.0)
target_distance     float   standoff: leader hovers this far
                            in front of the marker [m]         (default 1.0)
min_distance        float   hard floor — leader never gets
                            closer than this to the marker [m] (default 0.4)
"""

import math
import threading

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float64MultiArray

# ── Camera → body frame rotation ────────────────────────────────────────────
# Tello camera axes:  +Z forward, +X right,  +Y down
# Drone body axes:    +X forward, +Y left,   +Z up
#
# Verify with a bench test before flying:
#   hold drone still facing Vicon +X (yaw=0), place marker 0.5 m in front
#   → /aruco/pose should read  x ≈ drone_x+0.5, y ≈ drone_y, z ≈ drone_z
# If axes are swapped, exchange rows; if a sign is wrong, negate that row.
R_CAM_TO_BODY = np.array([
    [ 0,  0,  1],   # body +X (fwd)  = camera +Z (fwd)
    [-1,  0,  0],   # body +Y (left) = camera -X (right)
    [ 0, -1,  0],   # body +Z (up)   = camera -Y (down)
], dtype=float)

# KF state vector indices (must match vicon_kf_node output)
IDX_PX, IDX_PY, IDX_PZ, IDX_YAW = 0, 2, 4, 10


def _rvec_to_quat(rvec: np.ndarray):
    """Rodrigues rotation vector → (qw, qx, qy, qz)."""
    R, _ = cv2.Rodrigues(rvec)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return 0.25 / s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    if R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
    return (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s


class ArucoNode(Node):

    def __init__(self):
        super().__init__('aruco_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('marker_size',        0.185)
        self.declare_parameter('marker_id',          1)
        self.declare_parameter('detect_scale',       0.4)
        self.declare_parameter('camera_params_file', 'calibration/camera_params.npz')
        self.declare_parameter('publish_rate',       20.0)
        self.declare_parameter('image_topic',        'image_raw')
        self.declare_parameter('target_z_override',  -1.0)
        self.declare_parameter('target_distance',    1.0)   # standoff [m]
        self.declare_parameter('min_distance',       0.4)   # hard floor [m]

        marker_size = self.get_parameter('marker_size').value
        self._mid   = self.get_parameter('marker_id').value
        self._scale = self.get_parameter('detect_scale').value
        cam_file    = self.get_parameter('camera_params_file').value
        pub_rate    = self.get_parameter('publish_rate').value
        img_topic   = self.get_parameter('image_topic').value
        self._fixed_z        = self.get_parameter('target_z_override').value
        self._target_dist    = self.get_parameter('target_distance').value
        self._min_dist       = self.get_parameter('min_distance').value

        # ── Camera calibration ────────────────────────────────────────
        try:
            data = np.load(cam_file)
            self._K    = data['K']
            self._dist = data['dist']
            self.get_logger().info(f'Calibration loaded: {cam_file}')
        except Exception as e:
            self.get_logger().error(f'Cannot load camera params: {e}')
            raise

        # ── ArUco detector ────────────────────────────────────────────
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters())

        hs = marker_size / 2.0
        self._obj_pts = np.array([
            [-hs,  hs, 0], [ hs,  hs, 0],
            [ hs, -hs, 0], [-hs, -hs, 0],
        ], dtype=np.float32)

        # ── Shared state ──────────────────────────────────────────────
        self._latest_frame: np.ndarray | None = None
        self._frame_lock   = threading.Lock()

        # Detection result: (rvec, tvec, distance) | None
        self._det_result   = None
        self._det_lock     = threading.Lock()
        self._hit_count    = 0

        # Latest leader KF state
        self._kf_state: np.ndarray | None = None
        self._kf_lock  = threading.Lock()

        self._stop = threading.Event()

        # ── Subscribers ───────────────────────────────────────────────
        # Both topics are relative → resolved under /<leader_ns>/
        self.create_subscription(Image,              img_topic,  self._cb_image, 10)
        self.create_subscription(Float64MultiArray,  'kf_state', self._cb_kf,    10)

        # ── Publishers ────────────────────────────────────────────────
        # aruco_detected / aruco_distance are relative (under leader namespace)
        # /aruco/pose is absolute — consumed by formation_controller globally
        self._pub_detected = self.create_publisher(Bool,        'aruco_detected', 10)
        self._pub_distance = self.create_publisher(Float32,     'aruco_distance', 10)
        self._pub_world    = self.create_publisher(PoseStamped, '/aruco/pose',    10)

        # ── Detection thread + publish timer ──────────────────────────
        threading.Thread(target=self._detect_loop, daemon=True).start()
        self.create_timer(1.0 / pub_rate, self._publish)

        self.get_logger().info(
            f'aruco_node ready  ID={self._mid}  size={marker_size} m  '
            f'img={img_topic}  rate={pub_rate} Hz')

    # ── Callbacks ─────────────────────────────────────────────────────

    def _cb_image(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8) \
                  .reshape((msg.height, msg.width, 3)).copy()
        with self._frame_lock:
            self._latest_frame = frame

    def _cb_kf(self, msg: Float64MultiArray):
        if len(msg.data) >= 12:
            with self._kf_lock:
                self._kf_state = np.array(msg.data)

    # ── Detection loop (background thread) ───────────────────────────

    def _detect_loop(self):
        last_frame_id = None
        while not self._stop.is_set():
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None or id(frame) == last_frame_id:
                self._stop.wait(0.002)
                continue
            last_frame_id = id(frame)

            small = cv2.resize(frame, None,
                               fx=self._scale, fy=self._scale,
                               interpolation=cv2.INTER_NEAREST)
            corners_s, ids, _ = self._detector.detectMarkers(
                cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

            result = None
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    if mid != self._mid:
                        continue
                    corners = (corners_s[i] / self._scale).astype(np.float32)
                    ok, rvec, tvec = cv2.solvePnP(
                        self._obj_pts, corners[0],
                        self._K, self._dist,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        result = (rvec, tvec, float(np.linalg.norm(tvec)))
                    break

            # Debounce
            self._hit_count = min(self._hit_count + 1, 3) if result else \
                              max(self._hit_count - 1, 0)

            with self._det_lock:
                if self._hit_count >= 2:
                    self._det_result = result
                elif self._hit_count == 0:
                    self._det_result = None

    # ── Publish timer ─────────────────────────────────────────────────

    def _publish(self):
        with self._det_lock:
            det = self._det_result
        with self._kf_lock:
            kf = self._kf_state

        detected = det is not None
        self._pub_detected.publish(Bool(data=detected))

        if not detected:
            return

        rvec, tvec, distance = det

        # ── /aruco_distance (diagnostic) ─────────────────────────────
        self._pub_distance.publish(Float32(data=distance))

        if kf is None:
            self.get_logger().warn('No KF state yet — cannot publish world pose',
                                   throttle_duration_sec=2.0)
            return

        # ── Camera frame → world frame ────────────────────────────────
        # 1. camera tvec → drone body frame
        p_cam  = tvec.flatten()
        p_body = R_CAM_TO_BODY @ p_cam

        # 2. body frame → Vicon world frame (yaw rotation only; Tello stays level)
        yaw    = float(kf[IDX_YAW])
        cy, sy = math.cos(yaw), math.sin(yaw)
        R_b2w  = np.array([[cy, -sy, 0],
                            [sy,  cy, 0],
                            [ 0,   0, 1]])

        leader_pos = np.array([kf[IDX_PX], kf[IDX_PY], kf[IDX_PZ]])
        p_world    = leader_pos + R_b2w @ p_body

        # 3. Optional fixed-altitude override (useful for floor markers)
        if self._fixed_z >= 0.0:
            p_world[2] = self._fixed_z

        # 4. Standoff: shift the target backward so the leader hovers
        #    target_distance metres in front of the marker, not on top of it.
        #
        #    Direction vector = leader → marker (unit vector in XY plane)
        #    Target = marker_pos - unit_vec * target_distance
        #
        #    Also enforce min_distance: if the raw detected distance is already
        #    below the floor, clamp so the leader never moves closer.
        marker_pos   = p_world.copy()
        leader_to_marker = marker_pos - leader_pos   # world-frame vector
        horiz_dist   = float(np.linalg.norm(leader_to_marker[:2]))  # XY only

        if horiz_dist > 1e-3:   # avoid divide-by-zero when marker is directly above
            unit_xy = leader_to_marker[:2] / horiz_dist

            # How far to stand off — never closer than min_distance
            standoff = max(self._target_dist, self._min_dist)

            # Target XY = marker XY − standoff * unit direction
            p_world[0] = marker_pos[0] - unit_xy[0] * standoff
            p_world[1] = marker_pos[1] - unit_xy[1] * standoff
            # Z stays as detected (or fixed override already applied above)

        # 5. Orientation: rvec → quaternion, rotated into world frame
        qw, qx, qy, qz = _rvec_to_quat(rvec)

        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(p_world[0])
        msg.pose.position.y = float(p_world[1])
        msg.pose.position.z = float(p_world[2])
        msg.pose.orientation.w = float(qw)
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        self._pub_world.publish(msg)

    # ── Cleanup ───────────────────────────────────────────────────────

    def destroy_node(self):
        self._stop.set()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()