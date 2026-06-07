import signal
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32MultiArray

try:
    from djitellopy import Tello
    _DJITELLOPY_AVAILABLE = True
except ImportError:
    _DJITELLOPY_AVAILABLE = False


class TelloBridgeNode(Node):
    """Translate ROS2 RC commands into djitellopy calls and publish camera frames.

    Subscribes
    ----------
    rc_cmd  (Int32MultiArray)  [lr, fb, ud, yaw_rate]
    land    (Bool)             any message triggers landing

    Publishes
    ---------
    battery   (Float32)   battery percentage, every 5 s
    image_raw (Image)     BGR8 frames from the drone camera, ~30 Hz

    Parameters
    ----------
    mock        : bool  — skip real drone; publish no frames (default false)
    drone_ip    : str   — Tello IP (default 192.168.10.1)
    publish_image: bool — enable image_raw publishing (default true)
    image_rate  : float — frame publish rate in Hz (default 30.0)

    Notes
    -----
    Every tello_bridge instance (leader and followers) opens streamon() and
    publishes image_raw under its own namespace:
        /tello0/image_raw   ← aruco_detection_node subscribes here
        /tello1/image_raw   ← unused by default; available for other nodes
        /tello2/image_raw   ← unused by default; available for other nodes
    Only one djitellopy connection per drone IP is ever opened — there is no
    second connection from aruco_detection_node.
    """

    def __init__(self):
        super().__init__('tello_bridge')

        self.declare_parameter('mock',          False)
        self.declare_parameter('drone_ip',      '192.168.10.1')
        self.declare_parameter('publish_image', True)
        self.declare_parameter('image_rate',    20.0)

        self._mock          = self.get_parameter('mock').value
        drone_ip            = self.get_parameter('drone_ip').value
        self._pub_img_en    = self.get_parameter('publish_image').value
        image_rate          = self.get_parameter('image_rate').value

        self._drone      = None
        self._frame_read = None
        self._lock       = threading.Lock()
        self._flying     = False
        self._last_rc    = [0, 0, 0, 0]

        # ── Drone connection ────────────────────────────────────────
        if self._mock:
            self.get_logger().warn('TelloBridge: MOCK MODE — no real drone')
        else:
            if not _DJITELLOPY_AVAILABLE:
                self.get_logger().error('djitellopy not installed; use mock:=true')
                raise RuntimeError('djitellopy not available')

            self._drone = Tello(host=drone_ip)
            self._drone.connect()
            bat = self._drone.get_battery()
            self.get_logger().info(f'Tello connected at {drone_ip}. Battery: {bat}%')

            self._drone.takeoff()
            self._flying = True
            self.get_logger().info('Tello airborne')

            # Stream — opened once here, shared via image_raw topic
            if self._pub_img_en:
                self._drone.streamon()
                import time; time.sleep(1.0)
                self._frame_read = self._drone.get_frame_read()
                self.get_logger().info('Camera stream started')

        # ── Publishers / subscribers ────────────────────────────────
        self._pub_bat   = self.create_publisher(Float32, 'battery',   10)
        self._pub_image = self.create_publisher(Image,   'image_raw', 10)

        self._sub_rc   = self.create_subscription(
            Int32MultiArray, 'rc_cmd', self._cb_rc,   10)
        self._sub_land = self.create_subscription(
            Bool,            'land',   self._cb_land, 10)

        self._bat_timer = self.create_timer(5.0, self._publish_battery)

        if self._pub_img_en:
            dt = 1.0 / image_rate
            self._img_timer = self.create_timer(dt, self._publish_frame)

        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        self.get_logger().info('tello_bridge ready')

    # ── RC command ──────────────────────────────────────────────────
    def _cb_rc(self, msg: Int32MultiArray):
        if len(msg.data) < 4:
            return
        lr, fb, ud, yaw = [int(v) for v in msg.data[:4]]
        self._last_rc = [lr, fb, ud, yaw]

        if self._mock:
            self.get_logger().debug(f'RC cmd: lr={lr} fb={fb} ud={ud} yaw={yaw}')
            return

        with self._lock:
            if self._flying and self._drone is not None:
                self._drone.send_rc_control(lr, fb, ud, yaw)

    def _cb_land(self, msg: Bool):
        self._do_land()

    # ── Image publishing ─────────────────────────────────────────────
    def _publish_frame(self):
        if self._frame_read is None:
            return
        frame = self._frame_read.frame   # numpy BGR array from djitellopy
        if frame is None:
            return

        h, w = frame.shape[:2]
        msg = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.height   = h
        msg.width    = w
        msg.encoding = 'bgr8'
        msg.step     = w * 3
        msg.data     = frame.tobytes()
        self._pub_image.publish(msg)

    # ── Battery ──────────────────────────────────────────────────────
    def _publish_battery(self):
        if self._mock:
            return
        try:
            bat = float(self._drone.get_battery())
        except Exception:
            bat = -1.0
        msg = Float32(data=bat)
        self._pub_bat.publish(msg)
        if bat < 15.0:
            self.get_logger().warn(f'Low battery: {bat}% — landing')
            self._do_land()

    # ── Land / shutdown ──────────────────────────────────────────────
    def _do_land(self):
        with self._lock:
            if self._mock:
                self.get_logger().info('Mock land')
                self._flying = False
                return
            if self._drone is not None and self._flying:
                self._drone.send_rc_control(0, 0, 0, 0)
                self._drone.land()
                self._flying = False
                self.get_logger().info('Tello landed')

    def _shutdown_handler(self, *_):
        self.get_logger().info('Shutdown signal — landing')
        self._do_land()
        if self._drone is not None:
            if self._frame_read is not None:
                self._drone.streamoff()
            self._drone.end()


def main(args=None):
    rclpy.init(args=args)
    node = TelloBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._do_land()
        if node._drone is not None:
            if node._frame_read is not None:
                node._drone.streamoff()
            node._drone.end()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()