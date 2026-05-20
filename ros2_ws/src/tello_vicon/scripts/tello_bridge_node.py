import signal
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32MultiArray

# djitellopy import is deferred so the package can be imported without the
# library installed (e.g., when running unit tests or using mock mode).
try:
    from djitellopy import Tello
    _DJITELLOPY_AVAILABLE = True
except ImportError:
    _DJITELLOPY_AVAILABLE = False


class TelloBridgeNode(Node):
    """Translate ROS2 RC commands into djitellopy calls.

    Subscribes
    ----------
    /tello/rc_cmd  (Int32MultiArray)  [lr, fb, ud, yaw_rate]
    /tello/land    (Bool)             any message triggers landing

    Publishes
    ---------
    /tello/battery (Float32)         battery percentage, published every 5 s

    Parameters
    ----------
    mock        : bool   — if true, skip real drone; only log commands (default false)
    drone_ip    : str    — IP of the Tello (default 192.168.10.1)
    takeoff_alt : float  — not used directly; informational only (Tello auto-hovers)
    """

    def __init__(self):
        super().__init__('tello_bridge')

        self.declare_parameter('mock',     False)
        self.declare_parameter('drone_ip', '192.168.10.1')

        self._mock = self.get_parameter('mock').value
        drone_ip   = self.get_parameter('drone_ip').value

        self._drone = None
        self._lock  = threading.Lock()
        self._flying = False
        self._last_rc = [0, 0, 0, 0]

        #  Drone connection 
        if self._mock:
            self.get_logger().warn('TelloBridge: MOCK MODE — no real drone')
        else:
            if not _DJITELLOPY_AVAILABLE:
                self.get_logger().error('djitellopy not installed; switch to mock:=true')
                raise RuntimeError('djitellopy not available')

            self._drone = Tello(host=drone_ip)
            self._drone.connect()
            bat = self._drone.get_battery()
            self.get_logger().info(f'Tello connected. Battery: {bat}%')

            self._drone.takeoff()
            self._flying = True
            self.get_logger().info('Tello airborne')

        #  Publishers / subscribers 
        # All topics are relative — resolved under the node namespace so each
        # drone instance gets independent topics (e.g. /tello1/battery).
        self._pub_bat = self.create_publisher(Float32, 'battery', 10)

        self._sub_rc = self.create_subscription(
            Int32MultiArray, 'rc_cmd', self._cb_rc, 10)
        self._sub_land = self.create_subscription(
            Bool, 'land', self._cb_land, 10)

        self._bat_timer = self.create_timer(5.0, self._publish_battery)

        # Graceful shutdown on SIGINT/SIGTERM
        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        self.get_logger().info('tello_bridge ready')

    # 
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

    # 
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

    def _publish_battery(self):
        if self._mock:
            return
        try:
            bat = float(self._drone.get_battery())
        except Exception:
            bat = -1.0
        msg = Float32()
        msg.data = bat
        self._pub_bat.publish(msg)
        if bat < 15.0:
            self.get_logger().warn(f'Low battery: {bat}% — landing')
            self._do_land()

    def _shutdown_handler(self, *_):
        self.get_logger().info('Shutdown signal received — landing drone')
        self._do_land()
        if self._drone is not None:
            self._drone.end()
        rclpy.shutdown()


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
            node._drone.end()
        node.destroy_node()
        rclpy.shutdown()
