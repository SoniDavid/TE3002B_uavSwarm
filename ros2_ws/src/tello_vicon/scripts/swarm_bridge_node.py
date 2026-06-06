"""Single-process bridge for N Tello drones.

djitellopy binds UDP port 8889 once per OS process.  Spawning one
tello_bridge executable per drone means every process after the first
fails with EADDRINUSE.  This node manages all drones inside a single
process, sidestepping the collision entirely.

Parameters
----------
drone_subjects : str   comma-separated namespace list, e.g. "tello0,tello1"
drone_ips      : str   comma-separated IP list in the same order
leader_ns      : str   namespace of the leader (gets camera stream)
mock           : bool  skip real drones (default false)
image_rate     : float camera publish rate Hz (default 30.0)

Topics (per drone, under /<ns>/)
---------------------------------
Subscribed : rc_cmd  (Int32MultiArray)  [lr, fb, ud, yaw]
             land    (Bool)
Published  : battery (Float32)
             image_raw (Image)  — leader only
"""

import logging
import signal
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32MultiArray

try:
    from djitellopy import Tello
    _DJITELLOPY_AVAILABLE = True
except ImportError:
    _DJITELLOPY_AVAILABLE = False

logging.getLogger('djitellopy').setLevel(logging.WARNING)

# Base video-stream UDP port; each subsequent drone gets +1
_BASE_VS_PORT = 11111


class SwarmBridgeNode(Node):

    def __init__(self):
        super().__init__('swarm_bridge')

        self.declare_parameter('drone_subjects', 'tello0')
        self.declare_parameter('drone_ips',      '192.168.10.1')
        self.declare_parameter('leader_ns',      'tello0')
        self.declare_parameter('mock',           False)
        self.declare_parameter('image_rate',     30.0)

        subjects   = [s.strip() for s in
                      self.get_parameter('drone_subjects').value.split(',') if s.strip()]
        ips        = [ip.strip() for ip in
                      self.get_parameter('drone_ips').value.split(',')      if ip.strip()]
        leader_ns  = self.get_parameter('leader_ns').value
        self._mock = self.get_parameter('mock').value
        img_rate   = self.get_parameter('image_rate').value

        if len(subjects) != len(ips):
            raise ValueError(
                f'drone_subjects ({len(subjects)}) and drone_ips ({len(ips)}) '
                f'must have the same length')

        self._drones:      dict[str, Tello | None] = {}
        self._frame_reads: dict[str, object]        = {}
        self._flying:      dict[str, bool]          = {}
        self._locks:       dict[str, threading.Lock] = {}
        self._last_rc_t:   dict[str, float]          = {}

        self._pub_bat:   dict[str, object] = {}
        self._pub_image: dict[str, object] = {}

        for i, (ns, ip) in enumerate(zip(subjects, ips)):
            self._locks[ns]   = threading.Lock()
            self._flying[ns]  = False
            self._last_rc_t[ns] = time.time()

            # ── publishers ──────────────────────────────────────────
            self._pub_bat[ns] = self.create_publisher(
                Float32, f'/{ns}/battery', 10)
            self._pub_image[ns] = self.create_publisher(
                Image, f'/{ns}/image_raw', 10)

            # ── subscribers ─────────────────────────────────────────
            self.create_subscription(
                Int32MultiArray, f'/{ns}/rc_cmd',
                lambda msg, n=ns: self._cb_rc(msg, n), 10)
            self.create_subscription(
                Bool, f'/{ns}/land',
                lambda msg, n=ns: self._cb_land(msg, n), 10)

            # ── connect drone ────────────────────────────────────────
            if self._mock:
                self._drones[ns] = None
                self.get_logger().warn(f'[{ns}] MOCK MODE — no real drone')
                continue

            if not _DJITELLOPY_AVAILABLE:
                raise RuntimeError('djitellopy not installed; use mock:=true')

            try:
                drone = Tello(host=ip)
                drone.connect()
                bat = drone.get_battery()
                self.get_logger().info(f'[{ns}] connected {ip} — battery {bat}%')

                drone.takeoff()
                self._flying[ns] = True
                self.get_logger().info(f'[{ns}] airborne')

                is_leader = (ns == leader_ns)
                if is_leader:
                    vs_port = _BASE_VS_PORT + i
                    if vs_port != _BASE_VS_PORT:
                        drone.change_vs_udp(vs_port)
                    drone.streamon()
                    time.sleep(1.0)
                    self._frame_reads[ns] = drone.get_frame_read()
                    self.get_logger().info(f'[{ns}] camera stream started on port {vs_port}')

                self._drones[ns] = drone

            except Exception as exc:
                self.get_logger().error(
                    f'[{ns}] failed to connect/takeoff: {exc} — skipping this drone')
                self._drones[ns] = None

        # ── timers ──────────────────────────────────────────────────
        self.create_timer(5.0, self._publish_batteries)
        self.create_timer(4.0, self._keepalive)   # firmware auto-lands at 15 s

        if not self._mock:
            dt = 1.0 / img_rate
            self.create_timer(dt, self._publish_frames)

        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        self.get_logger().info('swarm_bridge ready')

    # ── RC ───────────────────────────────────────────────────────────

    def _cb_rc(self, msg: Int32MultiArray, ns: str):
        if len(msg.data) < 4:
            return
        lr, fb, ud, yaw = [int(v) for v in msg.data[:4]]
        if self._mock:
            self.get_logger().debug(
                f'[{ns}] RC lr={lr} fb={fb} ud={ud} yaw={yaw}')
            return
        drone = self._drones.get(ns)
        with self._locks[ns]:
            if drone and self._flying[ns]:
                drone.send_rc_control(lr, fb, ud, yaw)
                self._last_rc_t[ns] = time.time()

    def _keepalive(self):
        """Send rc 0 0 0 0 to any flying drone that hasn't received an RC command
        recently, preventing the Tello firmware's 15-second auto-land."""
        now = time.time()
        for ns, drone in self._drones.items():
            if drone is None or not self._flying[ns]:
                continue
            if (now - self._last_rc_t[ns]) > 3.0:
                with self._locks[ns]:
                    if self._flying[ns]:
                        drone.send_rc_control(0, 0, 0, 0)

    def _cb_land(self, msg: Bool, ns: str):
        self._do_land(ns)

    # ── Image ────────────────────────────────────────────────────────

    def _publish_frames(self):
        for ns, fr in self._frame_reads.items():
            frame = fr.frame
            if frame is None:
                continue
            h, w = frame.shape[:2]
            msg = Image()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            msg.height   = h
            msg.width    = w
            msg.encoding = 'bgr8'
            msg.step     = w * 3
            msg.data     = frame.tobytes()
            self._pub_image[ns].publish(msg)

    # ── Battery ──────────────────────────────────────────────────────

    def _publish_batteries(self):
        for ns, drone in self._drones.items():
            if drone is None:
                continue
            try:
                bat = float(drone.get_battery())
            except Exception:
                bat = -1.0
            self._pub_bat[ns].publish(Float32(data=bat))
            if bat >= 0 and bat < 15.0:
                self.get_logger().warn(f'[{ns}] low battery {bat}% — landing')
                self._do_land(ns)

    # ── Land / shutdown ──────────────────────────────────────────────

    def _do_land(self, ns: str):
        with self._locks[ns]:
            if self._mock:
                self._flying[ns] = False
                return
            drone = self._drones.get(ns)
            if drone and self._flying[ns]:
                drone.send_rc_control(0, 0, 0, 0)
                drone.land()
                self._flying[ns] = False
                self.get_logger().info(f'[{ns}] landed')

    def _shutdown_handler(self, *_):
        self.get_logger().info('Shutdown — landing all drones')
        for ns in list(self._drones.keys()):
            self._do_land(ns)
        for ns, drone in self._drones.items():
            if drone is None:
                continue
            if ns in self._frame_reads:
                drone.streamoff()
            drone.end()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = SwarmBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for ns in list(node._drones.keys()):
            node._do_land(ns)
        for ns, drone in node._drones.items():
            if drone is None:
                continue
            if ns in node._frame_reads:
                drone.streamoff()
            drone.end()
        node.destroy_node()
        rclpy.shutdown()
