"""swarm_bridge_node.py — one process, all drones.

WHY THIS EXISTS
---------------
djitellopy uses a single global UDP socket bound to port 8889 (CONTROL_UDP_PORT).
The socket is created on the first Tello() call and reused by all subsequent ones
via a global `threads_initialized` flag. This works fine when all Tello objects
live in the same process — but if you run one tello_bridge per drone (separate
OS processes), each process tries to bind() port 8889 and only the first succeeds.
The others crash with: OSError: [Errno 98] Address already in use.

This node creates ALL Tello objects inside one process so the socket is shared
correctly. It subscribes to /<subject>/rc_cmd for each drone and routes commands
individually, preserving the per-drone ROS namespace structure.

Subscribes (per drone, absolute topics)
----------------------------------------
  /<ns>/rc_cmd   (Int32MultiArray)  [lr, fb, ud, yaw]
  /<ns>/land     (Bool)

Publishes (per drone, absolute topics)
---------------------------------------
  /<ns>/battery    (Float32)
  /<ns>/image_raw  (Image)    leader only

Parameters
----------
  drones_cfg   str    same colon-separated subject:ip string as the launch arg
  leader_ns    str    namespace of the leader (gets image_raw)
  image_rate   float  camera publish rate Hz  (default 30.0)
  mock         bool   skip real drones        (default false)
"""

import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32MultiArray

try:
    from djitellopy import Tello
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class SwarmBridgeNode(Node):

    def __init__(self):
        super().__init__('swarm_bridge')

        self.declare_parameter('drones_cfg', '')
        self.declare_parameter('leader_ns',  '')
        self.declare_parameter('image_rate', 30.0)
        self.declare_parameter('mock',       False)

        cfg        = self.get_parameter('drones_cfg').value
        leader_ns  = self.get_parameter('leader_ns').value
        img_rate   = self.get_parameter('image_rate').value
        self._mock = self.get_parameter('mock').value

        if not self._mock and not _AVAILABLE:
            self.get_logger().error('djitellopy not installed')
            raise RuntimeError('djitellopy not available')

        parts = [p.strip() for p in cfg.split(':') if p.strip()]
        if len(parts) % 2 != 0:
            raise ValueError(f'drones_cfg must be subject:ip pairs, got: {cfg!r}')

        self._drones:      dict[str, Tello] = {}
        self._frame_reads: dict[str, object] = {}
        self._flying:      dict[str, bool]   = {}
        self._lock = threading.Lock()

        self._pubs_bat:   dict[str, object] = {}
        self._pubs_image: dict[str, object] = {}

        # ── Connect all drones in sequence (same process = one UDP socket) ──
        for i in range(0, len(parts), 2):
            subject  = parts[i]
            drone_ip = parts[i + 1]

            if not self._mock:
                drone = Tello(host=drone_ip)
                drone.connect()
                bat = drone.get_battery()
                self.get_logger().info(
                    f'[{subject}] connected {drone_ip} — battery {bat}%')
                drone.takeoff()
                self._flying[subject] = True
                self.get_logger().info(f'[{subject}] airborne')

                if subject == leader_ns:
                    drone.streamon()
                    time.sleep(1.0)
                    self._frame_reads[subject] = drone.get_frame_read()
                    self.get_logger().info(f'[{subject}] camera stream started')

                self._drones[subject] = drone
            else:
                self.get_logger().warn(f'MOCK: skipping connection for {subject}')
                self._flying[subject] = False

            # Absolute topic names (no namespace on this node itself)
            self.create_subscription(
                Int32MultiArray, f'/{subject}/rc_cmd',
                lambda msg, s=subject: self._cb_rc(msg, s), 10)
            self.create_subscription(
                Bool, f'/{subject}/land',
                lambda msg, s=subject: self._cb_land(msg, s), 10)

            self._pubs_bat[subject] = self.create_publisher(
                Float32, f'/{subject}/battery', 10)

            if subject == leader_ns:
                self._pubs_image[subject] = self.create_publisher(
                    Image, f'/{subject}/image_raw', 10)

        # ── Timers ───────────────────────────────────────────────────
        self.create_timer(5.0, self._publish_batteries)
        if self._frame_reads:
            self.create_timer(1.0 / img_rate, self._publish_frames)

        self.get_logger().info(
            f'swarm_bridge ready — {len(parts)//2} drone(s), leader={leader_ns}')

    # ── RC ───────────────────────────────────────────────────────────
    def _cb_rc(self, msg: Int32MultiArray, subject: str):
        if len(msg.data) < 4:
            return
        lr, fb, ud, yaw = [int(v) for v in msg.data[:4]]
        if self._mock:
            return
        with self._lock:
            drone = self._drones.get(subject)
            if drone and self._flying.get(subject):
                drone.send_rc_control(lr, fb, ud, yaw)

    def _cb_land(self, _msg, subject: str):
        self._land_one(subject)

    # ── Image ────────────────────────────────────────────────────────
    def _publish_frames(self):
        for subject, fr in self._frame_reads.items():
            frame = fr.frame
            if frame is None:
                continue
            pub = self._pubs_image.get(subject)
            if pub is None:
                continue
            h, w = frame.shape[:2]
            msg = Image()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            msg.height, msg.width = h, w
            msg.encoding = 'bgr8'
            msg.step     = w * 3
            msg.data     = frame.tobytes()
            pub.publish(msg)

    # ── Battery ──────────────────────────────────────────────────────
    def _publish_batteries(self):
        if self._mock:
            return
        for subject, drone in self._drones.items():
            try:
                bat = float(drone.get_battery())
            except Exception:
                bat = -1.0
            self._pubs_bat[subject].publish(Float32(data=bat))
            if 0 <= bat < 15.0:
                self.get_logger().warn(f'[{subject}] low battery {bat}% — landing')
                self._land_one(subject)

    # ── Land ─────────────────────────────────────────────────────────
    def _land_one(self, subject: str):
        with self._lock:
            drone = self._drones.get(subject)
            if drone and self._flying.get(subject):
                drone.send_rc_control(0, 0, 0, 0)
                drone.land()
                self._flying[subject] = False
                self.get_logger().info(f'[{subject}] landed')

    def _land_all(self):
        for subject in list(self._drones.keys()):
            self._land_one(subject)

    # ── Cleanup ──────────────────────────────────────────────────────
    def destroy_node(self):
        self._land_all()
        for subject, drone in self._drones.items():
            if subject in self._frame_reads:
                try:
                    drone.streamoff()
                except Exception:
                    pass
            try:
                drone.end()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SwarmBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()