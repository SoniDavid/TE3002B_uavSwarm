"""tello_controller_node.py — ONE node, all drones, one process.

Manages one PD controller per drone in a single process, eliminating
two Python interpreters and two rclpy event loops compared to running
three separate tello_controller processes.

Parameters
----------
  drone_subjects  str    comma-separated namespace list  (default "tello0")
  Kp_xy / Kp_z / Kp_yaw / Kd_xy / Kd_z  — PD gains
  v_max_xy / v_max_z / yaw_max           — velocity limits
  rate_hz         float  control loop rate  (default 30.0)
  timeout_s       float  zero RC if no state for this long (default 0.5)

Topics per drone (absolute)
---------------------------
  Subscribes: /<ns>/kf_state   (Float64MultiArray)
              /<ns>/reference  (PoseStamped)
  Publishes:  /<ns>/rc_cmd     (Int32MultiArray)  [lr, fb, ud, yaw]
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32MultiArray

# State vector indices
_PX, _VX = 0, 1
_PY, _VY = 2, 3
_PZ, _VZ = 4, 5
_YAW     = 10

_TWO_PI = 2.0 * math.pi


def _angle_diff(target: float, current: float) -> float:
    d = (target - current) % _TWO_PI
    if d > math.pi:
        d -= _TWO_PI
    return d


class _DronePD:
    """Per-drone PD controller state."""

    def __init__(self, ns: str, node: Node, gains: dict,
                 rate_hz: float, timeout: float):
        self.ns      = ns
        self.gains   = gains
        self.timeout = timeout

        self._kf_state: np.ndarray | None = None
        self._ref_pos  = np.zeros(3)
        self._ref_yaw  = 0.0
        self._last_t   = 0.0
        self._ref_recv = False

        # Pre-allocated RC message
        self._rc_msg      = Int32MultiArray()
        self._rc_msg.data = [0, 0, 0, 0]

        self._pub_rc = node.create_publisher(Int32MultiArray, f'/{ns}/rc_cmd', 10)

        node.create_subscription(
            Float64MultiArray, f'/{ns}/kf_state',
            lambda msg, d=self: _DronePD._cb_state(d, msg), 10)
        node.create_subscription(
            PoseStamped, f'/{ns}/reference',
            lambda msg, d=self: _DronePD._cb_ref(d, msg), 10)

    def _cb_state(self, msg: Float64MultiArray):
        if len(msg.data) >= 12:
            self._kf_state = np.asarray(msg.data)
            self._last_t   = time.monotonic()

    def _cb_ref(self, msg: PoseStamped):
        self._ref_pos[0] = msg.pose.position.x
        self._ref_pos[1] = msg.pose.position.y
        self._ref_pos[2] = msg.pose.position.z
        q = msg.pose.orientation
        self._ref_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._ref_recv = True

    def tick(self):
        DEADBAND_XY = 0.10  # 10 centímetros
        DEADBAND_Z  = 0.10  # 10 centímetros
        DEADBAND_YAW = math.radians(10.0) # 10 grados
        
        g = self.gains

        if not self._ref_recv or self._kf_state is None or \
                (time.monotonic() - self._last_t) > self.timeout:
            self._rc_msg.data = [0, 0, 0, 0]
            self._pub_rc.publish(self._rc_msg)
            return

        x   = self._kf_state
        pos = x[[_PX, _PY, _PZ]]
        vel = x[[_VX, _VY, _VZ]]
        yaw = x[_YAW]
        
        e_pos_full = self._ref_pos - pos
        e_yaw_raw  = _angle_diff(self._ref_yaw, yaw)
        
        e_pos = np.array([
            e_pos_full[0] if abs(e_pos_full[0]) > DEADBAND_XY else 0.0,
            e_pos_full[1] if abs(e_pos_full[1]) > DEADBAND_XY else 0.0,
            e_pos_full[2] if abs(e_pos_full[2]) > DEADBAND_Z  else 0.0
        ])
        
        yaw_error = e_yaw_raw if abs(e_yaw_raw) > DEADBAND_YAW else 0.0

        e_pos = np.clip(e_pos, -0.5, 0.5)
        e_vel = -vel

        # XY world-frame velocity command
        vxy = g['Kp_xy'] * e_pos[:2] + g['Kd_xy'] * e_vel[:2]
        n   = math.hypot(vxy[0], vxy[1])
        if n > g['v_max_xy']:
            vxy *= g['v_max_xy'] / n

        # Z command
        vz = float(np.clip(
            g['Kp_z'] * e_pos[2] + g['Kd_z'] * e_vel[2],
            -g['v_max_z'], g['v_max_z']))

        # Rotate XY to body frame
        cy, sy  = math.cos(yaw), math.sin(yaw)
        v_fwd   =  vxy[0] * cy + vxy[1] * sy
        v_left  = -vxy[0] * sy + vxy[1] * cy

        # Yaw PD
        yaw_cmd = float(np.clip(
            g['Kp_yaw'] * yaw_error,
            -g['yaw_max'], g['yaw_max']))

        scale_xy  = 100.0 / g['v_max_xy']
        scale_z   = 100.0 / g['v_max_z']
        scale_yaw = 100.0 / g['yaw_max']

        self._rc_msg.data = [
            int(np.clip(-v_left  * scale_xy,  -30, 30)),
            int(np.clip( v_fwd   * scale_xy,  -30, 30)),
            int(np.clip( vz      * scale_z,   -50, 50)),
            int(np.clip( yaw_cmd * scale_yaw, -30, 30)),
        ]
        self._pub_rc.publish(self._rc_msg)


class TelloControllerNode(Node):

    def __init__(self):
        super().__init__('tello_controller')

        self.declare_parameter('drone_subjects', 'tello0')
        self.declare_parameter('Kp_xy',    0.5)
        self.declare_parameter('Kp_z',     0.6)
        self.declare_parameter('Kp_yaw',   0.4)
        self.declare_parameter('Kd_xy',    0.2)
        self.declare_parameter('Kd_z',     0.2)
        self.declare_parameter('v_max_xy', 0.5)
        self.declare_parameter('v_max_z',  0.3)
        self.declare_parameter('yaw_max',  30.0)
        self.declare_parameter('rate_hz',  50.0)
        self.declare_parameter('timeout_s', 0.5)

        subjects = [s.strip() for s in
                    self.get_parameter('drone_subjects').value.split(',')
                    if s.strip()]

        gains = {k: self.get_parameter(k).value
                 for k in ('Kp_xy','Kp_z','Kp_yaw','Kd_xy','Kd_z',
                            'v_max_xy','v_max_z','yaw_max')}
        rate    = self.get_parameter('rate_hz').value
        timeout = self.get_parameter('timeout_s').value

        self._drones = [
            _DronePD(ns, self, gains, rate, timeout)
            for ns in subjects
        ]

        self.create_timer(1.0 / rate, self._tick_all)
        self.get_logger().info(
            f'tello_controller ready — {len(subjects)} drone(s): {subjects}')

    def _tick_all(self):
        for d in self._drones:
            d.tick()


def main(args=None):
    rclpy.init(args=args)
    node = TelloControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()