"""Formation controller node — leader + two followers (S1, S2).

Architecture
------------
This node sits *above* the per-drone tello_controller nodes.
It does NOT control drones directly; it publishes ``/telloN/reference``
(geometry_msgs/PoseStamped) for each drone's existing PD controller to track.

                 ┌─────────────────────────────────────┐
                 │   formation_controller_node          │
                 │                                      │
  /tello0/kf_state ──► leader pose                     │
  /tello1/kf_state ──► s1 pose     ──► offset math ──► │ ──► /tello0/reference
  /tello2/kf_state ──► s2 pose                         │ ──► /tello1/reference
  /aruco/pose      ──► aruco pose  ──► leader ref  ──► │ ──► /tello2/reference
                 └─────────────────────────────────────┘

Formations  (offsets in the leader's body frame, metres)
---------
  V              -> classic V (S1 right-back, S2 left-back)
  LINE           -> side-by-side line
  COLUMN         -> single file with altitude step
  PANORAMIC      -> side-by-side with yaw spread
  RECONSTRUCTION -> triangular with inward yaw for 3-D reconstruction

Parameters
----------
  formation       : str   -> active formation key (default "V")
  leader_ns       : str   -> ROS namespace of the leader drone (default "tello0")
  s1_ns           : str   -> namespace of follower 1               (default "tello1")
  s2_ns           : str   -> namespace of follower 2               (default "tello2")
  aruco_topic     : str   -> topic where ArUco pose arrives
  aruco_timeout_s : float -> zero leader vel if no ArUco for this long (default 0.5)
  rate_hz         : float -> reference publish rate (default 20.0)

Topics published
----------------
  /<leader_ns>/reference   (PoseStamped)
  /<s1_ns>/reference       (PoseStamped)
  /<s2_ns>/reference       (PoseStamped)

Topics subscribed
-----------------
  /<leader_ns>/kf_state    (Float64MultiArray)  [px,vx,py,vy,pz,vz,...,yaw,vyaw]
  /<s1_ns>/kf_state        (Float64MultiArray)
  /<s2_ns>/kf_state        (Float64MultiArray)
  <aruco_topic>            (PoseStamped)
  /swarm/ready             (Bool)               published by swarm_bridge when all
                                                drones are airborne
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray

# ─────────────────────────────────────────────────────────────────
# Formation table  (offsets in leader body frame, metres & degrees)
# ─────────────────────────────────────────────────────────────────
# Each entry: {"dx": forward, "dy": left, "dz": up, "dyaw": deg}
FORMATIONS: dict[str, dict[str, dict]] = {
    "V": {
        "S1": {"dx": -0.50, "dy":  1.00, "dz": 0.0, "dyaw": 0.0},
        "S2": {"dx": -0.50, "dy": -1.00, "dz": 0.0, "dyaw": 0.0},
    },
    "LINE": {
        "S1": {"dx":  0.0,  "dy":  1.00, "dz": 0.0, "dyaw": 0.0},
        "S2": {"dx":  0.0,  "dy": -1.00, "dz": 0.0, "dyaw": 0.0},
    },
    "COLUMN": {
        "S1": {"dx": -0.60, "dy":  0.0,  "dz":  0.20, "dyaw": 0.0},
        "S2": {"dx": -1.20, "dy":  0.0,  "dz":  0.40, "dyaw": 0.0},
    },
    "PANORAMIC": {
        "S1": {"dx":  0.0,  "dy":  0.80, "dz": 0.0, "dyaw":  45.0},
        "S2": {"dx":  0.0,  "dy": -0.80, "dz": 0.0, "dyaw": -45.0},
    },
    "RECONSTRUCTION": {
        "S1": {"dx": -0.40, "dy":  0.70, "dz":  0.20, "dyaw": -30.0},
        "S2": {"dx": -0.40, "dy": -0.70, "dz":  0.20, "dyaw":  30.0},
    },
}

# KF state vector indices (matches vicon_kf_node.py output)
IDX_PX, IDX_VX = 0, 1
IDX_PY, IDX_VY = 2, 3
IDX_PZ, IDX_VZ = 4, 5
IDX_YAW        = 10


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _angle_diff(target: float, current: float) -> float:
    """Shortest-path signed angular difference (radians)."""
    d = target - current
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def _yaw_to_quat(yaw: float):
    """Return (x, y, z, w) quaternion for a pure yaw rotation."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return 0.0, 0.0, sy, cy


def _make_reference(x: float, y: float, z: float, yaw: float,
                    frame_id: str = "world") -> PoseStamped:
    """Build a PoseStamped from world-frame position + yaw."""
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    qx, qy, qz, qw = _yaw_to_quat(yaw)
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


def _body_offset_to_world(leader_pos: np.ndarray, leader_yaw: float,
                           offset: dict) -> tuple[float, float, float, float]:
    """
    Convert a body-frame offset (dx forward, dy left, dz up) to a world-frame
    target position and absolute yaw for a follower.

    Returns (x_w, y_w, z_w, yaw_w).
    """
    cy, sy = math.cos(leader_yaw), math.sin(leader_yaw)
    dx_w = offset["dx"] * cy - offset["dy"] * sy
    dy_w = offset["dx"] * sy + offset["dy"] * cy

    x_w = leader_pos[0] + dx_w
    y_w = leader_pos[1] + dy_w
    z_w = leader_pos[2] + offset["dz"]

    yaw_w = leader_yaw + math.radians(offset["dyaw"])
    return x_w, y_w, z_w, yaw_w


# ─────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────

class FormationControllerNode(Node):

    def __init__(self):
        super().__init__("formation_controller")

        # ── Parameters ───────────────────────────────────────────
        self.declare_parameter("formation",       "V")
        self.declare_parameter("leader_ns",       "tello0")
        self.declare_parameter("s1_ns",           "tello1")
        self.declare_parameter("s2_ns",           "tello2")
        self.declare_parameter("aruco_topic",     "/aruco/pose")
        self.declare_parameter("aruco_timeout_s", 0.5)
        self.declare_parameter("rate_hz",         20.0)

        formation_key = self.get_parameter("formation").value.upper()
        if formation_key not in FORMATIONS:
            self.get_logger().warn(
                f"Unknown formation '{formation_key}', falling back to 'V'.")
            formation_key = "V"
        self._formation = FORMATIONS[formation_key]
        self.get_logger().info(f"Active formation: {formation_key}")

        ns_leader = self.get_parameter("leader_ns").value
        ns_s1     = self.get_parameter("s1_ns").value
        ns_s2     = self.get_parameter("s2_ns").value
        aruco_top = self.get_parameter("aruco_topic").value
        self._aruco_timeout = self.get_parameter("aruco_timeout_s").value

        # ── State ─────────────────────────────────────────────────
        self._state: dict[str, np.ndarray | None] = {
            "leader": None, "s1": None, "s2": None
        }

        # ArUco: initialise last_t to NOW so aruco_fresh starts False
        # correctly instead of being True at t=0 (unix epoch bug)
        self._aruco_pos      = np.zeros(3)
        self._aruco_yaw      = 0.0
        self._aruco_last_t   = self.get_clock().now().nanoseconds * 1e-9
        self._aruco_received = False   # True only after first real detection

        # Ready gate: do not publish references until swarm_bridge
        # confirms every drone is airborne via /swarm/ready.
        # Can also be set manually:
        #   ros2 topic pub /swarm/ready std_msgs/Bool "data: true" --once


        # ── Subscribers ───────────────────────────────────────────
        def _make_kf_sub(ns: str, key: str):
            return self.create_subscription(
                Float64MultiArray,
                f"/{ns}/kf_state",
                lambda msg, k=key: self._cb_kf(msg, k),
                10,
            )

        _make_kf_sub(ns_leader, "leader")
        _make_kf_sub(ns_s1, "s1")
        _make_kf_sub(ns_s2, "s2")

        self.create_subscription(PoseStamped, aruco_top, self._cb_aruco, 10)

        # ── Publishers ────────────────────────────────────────────
        self._pub_leader = self.create_publisher(
            PoseStamped, f"/{ns_leader}/reference", 10)
        self._pub_s1 = self.create_publisher(
            PoseStamped, f"/{ns_s1}/reference", 10)
        self._pub_s2 = self.create_publisher(
            PoseStamped, f"/{ns_s2}/reference", 10)

        # ── Timer ─────────────────────────────────────────────────
        dt = 1.0 / self.get_parameter("rate_hz").value
        self.create_timer(dt, self._tick)

        self.get_logger().info("formation_controller started")

    # ── Callbacks ─────────────────────────────────────────────────

    def _cb_kf(self, msg: Float64MultiArray, key: str):
        if len(msg.data) >= 12:
            self._state[key] = np.array(msg.data)


    def _cb_aruco(self, msg: PoseStamped):
        """Store the latest ArUco pose (world frame)."""
        self._aruco_pos[0] = msg.pose.position.x
        self._aruco_pos[1] = msg.pose.position.y
        self._aruco_pos[2] = msg.pose.position.z
        q = msg.pose.orientation
        self._aruco_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self._aruco_last_t   = self.get_clock().now().nanoseconds * 1e-9
        self._aruco_received = True

    # ── Main tick ─────────────────────────────────────────────────

    def _tick(self):
        # Single guard — wait until leader KF state is available.
        ldr = self._state["leader"]
        if ldr is None:
            self.get_logger().warn(
                'Waiting for leader kf_state — check Vicon remap',
                throttle_duration_sec=2.0)
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # ── 1. Leader reference ───────────────────────────────────
        # The leader receives NO reference until ArUco is detected for
        # the first time. Before detection the leader hovers freely
        # under the Tello's own internal stabilization — the controller
        # stays in RC=0 (timeout/no-ref mode). Once ArUco is detected,
        # the leader tracks the marker. If ArUco is lost after being
        # seen, the leader holds its last Vicon position.
        aruco_fresh = (self._aruco_received and
                       (now - self._aruco_last_t) < self._aruco_timeout)

        if not self._aruco_received:
            # ArUco never seen yet — do NOT publish leader reference.
            # tello_controller stays in RC=0 (hover) for the leader.
            # Still publish follower references so they can form up
            # relative to the leader's current real position.
            self.get_logger().info(
                'Leader waiting for ArUco detection — hovering freely',
                throttle_duration_sec=3.0)
        elif aruco_fresh:
            leader_ref = _make_reference(
                self._aruco_pos[0],
                self._aruco_pos[1],
                self._aruco_pos[2],
                self._aruco_yaw,
            )
            leader_ref.header.stamp = self.get_clock().now().to_msg()
            self._pub_leader.publish(leader_ref)
        else:
            # ArUco was seen before but is now lost — hold last Vicon position
            leader_ref = _make_reference(
                ldr[IDX_PX], ldr[IDX_PY], ldr[IDX_PZ], ldr[IDX_YAW])
            leader_ref.header.stamp = self.get_clock().now().to_msg()
            self._pub_leader.publish(leader_ref)

        # ── 2. Followers: offset from leader's real position ──────
        leader_pos = np.array([ldr[IDX_PX], ldr[IDX_PY], ldr[IDX_PZ]])
        leader_yaw = ldr[IDX_YAW]
        now_msg    = self.get_clock().now().to_msg()

        for pub, key in ((self._pub_s1, "S1"), (self._pub_s2, "S2")):
            offset = self._formation[key]
            xw, yw, zw, yaw_w = _body_offset_to_world(
                leader_pos, leader_yaw, offset)
            ref = _make_reference(xw, yw, zw, yaw_w)
            ref.header.stamp = now_msg
            pub.publish(ref)


# ─────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = FormationControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()