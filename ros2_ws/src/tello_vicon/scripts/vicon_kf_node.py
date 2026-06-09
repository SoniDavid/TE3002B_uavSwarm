"""vicon_kf_node.py — ONE node, all drones, one process.

CPU optimizations:
  1. Single process — 1 Python interpreter instead of 3
  2. KF step cached P update — only recomputes gain every 10 steps
  3. Sparse C matrix exploited directly (no full matrix multiply)
  4. kf_state published at publish_hz (default 30 Hz) not at Vicon rate (100 Hz)
     The controller only runs at 30 Hz so publishing faster wastes CPU on
     message serialization. KF still runs at full 100 Hz for accuracy.
  5. No x.copy() — returns view, caller must not modify
  6. Vectorized angle unwrap — no Python loop
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from tello_vicon_scripts.kalman_filter import ViconKF

_ZERO_SQ = 1e-12


def _quat_to_euler(qx, qy, qz, qw):
    roll  = math.atan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
    sinp  = max(-1.0, min(1.0, 2.0*(qw*qy - qz*qx)))
    pitch = math.asin(sinp)
    yaw   = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    return roll, pitch, yaw


class _DroneKF:

    def __init__(self, ns: str, node: Node, kf_kwargs: dict,
                 max_path: int, publish_hz: float):
        self.ns          = ns
        self.kf          = ViconKF()
        self.kf.init(**kf_kwargs)
        self.kf_kwargs   = dict(kf_kwargs)
        self.dt          = kf_kwargs['dt']
        self.last_stamp  = None
        self.initialized = False

        # Publish decimation — run KF at 100 Hz, publish at publish_hz
        self._publish_hz      = publish_hz
        self._publish_period  = 1.0 / publish_hz
        self._last_pub_t      = 0.0

        # Latest KF state (updated every step, published at lower rate)
        self._x: np.ndarray | None = None

        # Pre-allocated messages
        self.state_msg      = Float64MultiArray()
        self.state_msg.data = [0.0] * 12
        self.pose_msg       = PoseStamped()
        self.pose_msg.header.frame_id = 'world'

        # Ring-buffer path (written at full rate, published by timer)
        self.max_path  = max_path
        self.path_buf  : list[PoseStamped] = []
        self.path_head = 0

        # Publishers — absolute topics
        self.pub_state = node.create_publisher(Float64MultiArray, f'/{ns}/kf_state', 10)
        self.pub_pose  = node.create_publisher(PoseStamped,       f'/{ns}/kf_pose',  10)
        self.pub_path  = node.create_publisher(Path,              f'/{ns}/kf_path',  10)

        topic = f'/vicon/{ns}/{ns}'
        node.create_subscription(
            PoseStamped, topic,
            lambda msg, d=self: d._cb(msg), 10)
        node.get_logger().info(f'[{ns}] KF → {topic}  publish {publish_hz:.0f} Hz')

    def _cb(self, msg: PoseStamped):
        p = msg.pose.position
        if p.x*p.x + p.y*p.y + p.z*p.z < _ZERO_SQ:
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_stamp is not None:
            dt = stamp - self.last_stamp
            if 0.0005 < dt < 1.0 and abs(dt - self.dt) > self.dt * 0.1:
                self.kf_kwargs['dt'] = dt
                self.kf.init(**self.kf_kwargs)
                self.dt = dt
        self.last_stamp = stamp

        q = msg.pose.orientation
        roll, pitch, yaw = _quat_to_euler(q.x, q.y, q.z, q.w)

        if not self.initialized:
            seed = np.zeros(ViconKF.NX)
            seed[0]=p.x; seed[2]=p.y; seed[4]=p.z
            seed[6]=roll; seed[8]=pitch; seed[10]=yaw
            self.kf._x = seed
            self.initialized = True

        # KF step — always runs at full Vicon rate for accuracy
        x = self.kf.step(np.array([p.x, p.y, p.z, roll, pitch, yaw]))
        self._x = x

        # ── Publish at reduced rate ───────────────────────────────
        # This is the main CPU saving: ROS2 Python publish() serializes
        # the message on every call (~50µs each). At 100 Hz × 3 drones
        # × 2 topics = 600 publish/sec. At 30 Hz = 180 publish/sec.
        if stamp - self._last_pub_t < self._publish_period:
            # Still update path buffer at full rate for smooth trajectory
            self._append_path(x, msg.header.stamp)
            return
        self._last_pub_t = stamp

        # State
        self.state_msg.data = x.tolist()
        self.pub_state.publish(self.state_msg)

        # Pose
        pm = self.pose_msg
        pm.header.stamp    = msg.header.stamp
        pm.pose.position.x = float(x[0])
        pm.pose.position.y = float(x[2])
        pm.pose.position.z = float(x[4])
        cy=math.cos(x[10]*.5); sy=math.sin(x[10]*.5)
        cr=math.cos(x[6] *.5); sr=math.sin(x[6] *.5)
        cp=math.cos(x[8] *.5); sp=math.sin(x[8] *.5)
        pm.pose.orientation.w = cr*cp*cy + sr*sp*sy
        pm.pose.orientation.x = sr*cp*cy - cr*sp*sy
        pm.pose.orientation.y = cr*sp*cy + sr*cp*sy
        pm.pose.orientation.z = cr*cp*sy - sr*sp*cy
        self.pub_pose.publish(pm)

        self._append_path(x, msg.header.stamp)

    def _append_path(self, x, stamp):
        if len(self.path_buf) < self.max_path:
            self.path_buf.append(PoseStamped())
        idx = self.path_head % self.max_path
        pb  = self.path_buf[idx]
        pb.header.stamp    = stamp
        pb.header.frame_id = 'world'
        pb.pose.position.x = float(x[0])
        pb.pose.position.y = float(x[2])
        pb.pose.position.z = float(x[4])
        pb.pose.orientation = self.pose_msg.pose.orientation
        self.path_head += 1

    def publish_path(self, stamp):
        if not self.path_buf:
            return
        path_msg = Path()
        path_msg.header.stamp    = stamp
        path_msg.header.frame_id = 'world'
        n   = min(self.path_head, self.max_path)
        idx = self.path_head % self.max_path
        path_msg.poses = (self.path_buf[:n] if self.path_head <= self.max_path
                          else self.path_buf[idx:] + self.path_buf[:idx])
        self.pub_path.publish(path_msg)


class ViconKFNode(Node):

    def __init__(self):
        super().__init__('vicon_kf_node')

        self.declare_parameter('drone_subjects', 'tello0')
        self.declare_parameter('dt',             0.01)
        self.declare_parameter('q_pos',          1e-3)
        self.declare_parameter('q_vel',          1e-1)
        self.declare_parameter('q_ang',          1e-4)
        self.declare_parameter('q_rate',         1e-2)
        self.declare_parameter('r_pos',          1e-6)
        self.declare_parameter('r_ang',          1e-5)
        self.declare_parameter('max_path_len',   2000)
        self.declare_parameter('path_rate_hz',   5.0)
        self.declare_parameter('publish_hz',     50.0)  

        subjects = [s.strip() for s in
                    self.get_parameter('drone_subjects').value.split(',') if s.strip()]

        kf_kwargs = {k: self.get_parameter(k).value
                     for k in ('dt','q_pos','q_vel','q_ang','q_rate','r_pos','r_ang')}

        max_path    = self.get_parameter('max_path_len').value
        publish_hz  = self.get_parameter('publish_hz').value

        self._drones = [
            _DroneKF(ns, self, kf_kwargs, max_path, publish_hz)
            for ns in subjects
        ]

        path_hz = self.get_parameter('path_rate_hz').value
        self.create_timer(1.0 / path_hz, self._publish_paths)

        self.get_logger().info(
            f'vicon_kf_node ready — {len(subjects)} drones, '
            f'KF@100Hz, publish@{publish_hz:.0f}Hz')

    def _publish_paths(self):
        stamp = self.get_clock().now().to_msg()
        for d in self._drones:
            d.publish_path(stamp)


def main(args=None):
    rclpy.init(args=args)
    node = ViconKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()