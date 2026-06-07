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

Global published
----------------
/swarm/ready (Bool)  — fired once after ALL drones confirm airborne.
                       formation_controller waits for this before
                       sending any reference.  Can also be sent manually:
                         ros2 topic pub /swarm/ready std_msgs/Bool \
                           "data: true" --once
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
        self.declare_parameter('image_rate',     20.0)

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

        self._drones:      dict[str, Tello | None]   = {}
        self._frame_reads: dict[str, object]          = {}
        self._flying:      dict[str, bool]            = {}
        self._locks:       dict[str, threading.Lock] = {}
        self._last_rc_t:   dict[str, float]           = {}

        self._pub_bat:   dict[str, object] = {}
        self._pub_image: dict[str, object] = {}

        # /swarm/ready publisher — fired once after all drones airborne
        # Using queue_size=1 helps with persistence, but we'll also periodically re-publish
        self._pub_ready = self.create_publisher(Bool, '/swarm/ready', 1)
        self._ready_to_publish = False  # gate for periodic re-publishing

        # ── Phase 1: connect all drones sequentially ─────────────
        # djitellopy requires one handshake at a time (shared UDP socket).
        for i, (ns, ip) in enumerate(zip(subjects, ips)):
            self._locks[ns]     = threading.Lock()
            self._flying[ns]    = False
            self._last_rc_t[ns] = time.time()

            self._pub_bat[ns] = self.create_publisher(
                Float32, f'/{ns}/battery', 10)
            self._pub_image[ns] = self.create_publisher(
                Image, f'/{ns}/image_raw', 10)

            self.create_subscription(
                Int32MultiArray, f'/{ns}/rc_cmd',
                lambda msg, n=ns: self._cb_rc(msg, n), 10)
            self.create_subscription(
                Bool, f'/{ns}/land',
                lambda msg, n=ns: self._cb_land(msg, n), 10)

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
                self._drones[ns] = drone
            except Exception as exc:
                self.get_logger().error(
                    f'[{ns}] connection failed: {exc} — skipping')
                self._drones[ns] = None

        # ── Phase 2: take off ALL drones in parallel ──────────────
        # All Tello objects already share the same UDP socket (same process),
        # so parallel takeoff commands are safe.
        if not self._mock:
            takeoff_threads = []

            def _takeoff(ns: str, drone: Tello, index: int):
                try:
                    drone.takeoff()
                    with self._locks[ns]:
                        self._flying[ns] = True
                    self.get_logger().info(f'[{ns}] airborne')

                    # Camera stream — all drones get a stream on unique UDP ports
                    # so video_recorder_node can subscribe to any of them.
                    vs_port = _BASE_VS_PORT + index
                    if vs_port != _BASE_VS_PORT:
                        drone.change_vs_udp(vs_port)
                    drone.streamon()
                    time.sleep(0.5)
                    self._frame_reads[ns] = drone.get_frame_read()
                    self.get_logger().info(
                        f'[{ns}] camera stream on port {vs_port}')
                except Exception as exc:
                    self.get_logger().error(f'[{ns}] takeoff failed: {exc}')

            for i, (ns, drone) in enumerate(self._drones.items()):
                if drone is None:
                    continue
                t = threading.Thread(
                    target=_takeoff, args=(ns, drone, i), daemon=True)
                takeoff_threads.append(t)
                t.start()

            # Wait until every drone has confirmed airborne
            for t in takeoff_threads:
                t.join()

            self.get_logger().info('All drones airborne')

            # Signal formation_controller it can start sending references
            time.sleep(0.5)   # let subscribers connect
            self._ready_to_publish = True
            self._pub_ready.publish(Bool(data=True))
            self.get_logger().info('Published /swarm/ready')
            
            # Periodically re-publish to ensure late subscribers get the message
            self.create_timer(1.0, self._publish_ready_periodic)

        # ── Timers ────────────────────────────────────────────────
        self.create_timer(5.0, self._publish_batteries)
        self.create_timer(4.0, self._keepalive)   # firmware auto-lands at 15 s

        if not self._mock:
            dt = 1.0 / img_rate
            self.create_timer(dt, self._publish_frames)

        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        self.get_logger().info('swarm_bridge ready')

    # ── RC ────────────────────────────────────────────────────────

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
        """Send neutral RC if controller goes silent — prevents 15 s auto-land."""
        now = time.time()
        for ns, drone in self._drones.items():
            if drone is None or not self._flying[ns]:
                continue
            if (now - self._last_rc_t[ns]) > 3.0:
                with self._locks[ns]:
                    if self._flying[ns]:
                        drone.send_rc_control(0, 0, 0, 0)

    def _publish_ready_periodic(self):
        """Periodically re-publish /swarm/ready to handle late subscribers."""
        if self._ready_to_publish:
            self._pub_ready.publish(Bool(data=True))

    def _cb_land(self, msg: Bool, ns: str):
        self._do_land(ns)

    # ── Image ─────────────────────────────────────────────────────

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

    # ── Battery ───────────────────────────────────────────────────

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

    # ── Land / shutdown ───────────────────────────────────────────

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

    def _land_all_parallel(self):
        """Land all flying drones simultaneously using threads."""
        threads = []
        for ns in list(self._drones.keys()):
            if self._flying.get(ns):
                t = threading.Thread(target=self._do_land, args=(ns,), daemon=True)
                threads.append(t)
                t.start()
        for t in threads:
            t.join()
        self.get_logger().info('All drones landed')

    def _shutdown_handler(self, *_):
        self.get_logger().info('Shutdown — landing all drones simultaneously')
        self._land_all_parallel()
        for ns, drone in self._drones.items():
            if drone is None:
                continue
            if ns in self._frame_reads:
                try:
                    drone.streamoff()
                except Exception:
                    pass
            try:
                drone.end()
            except Exception:
                pass

    def destroy_node(self):
        self._land_all_parallel()
        for ns, drone in self._drones.items():
            if drone is None:
                continue
            if ns in self._frame_reads:
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