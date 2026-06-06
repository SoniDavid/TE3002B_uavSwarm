import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32MultiArray


class TelloControllerNode(Node):
    """PD position controller: converts KF state + reference into Tello RC commands.

    Subscribes
    ----------
    /tello/kf_state   (Float64MultiArray)
        [px, vx, py, vy, pz, vz, roll, vroll, pitch, vpitch, yaw, vyaw]
    /tello/reference  (PoseStamped)  — desired position setpoint

    Publishes
    ---------
    /tello/rc_cmd  (Int32MultiArray)
        [lr, fb, ud, yaw_rate]  — values in [-100, 100] matching djitellopy convention
    """

    # State vector index helpers
    IDX_PX, IDX_VX = 0, 1
    IDX_PY, IDX_VY = 2, 3
    IDX_PZ, IDX_VZ = 4, 5
    IDX_YAW, IDX_VYAW = 10, 11

    def __init__(self):
        super().__init__('tello_controller')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('Kp_xy',    0.5)
        self.declare_parameter('Kp_z',     0.6)
        self.declare_parameter('Kp_yaw',   0.4)
        self.declare_parameter('Kd_xy',    0.2)
        self.declare_parameter('Kd_z',     0.2)
        self.declare_parameter('v_max_xy', 0.5)   # m/s
        self.declare_parameter('v_max_z',  0.3)   # m/s
        self.declare_parameter('yaw_max',  30.0)  # deg/s
        self.declare_parameter('rate_hz',  30.0)
        self.declare_parameter('timeout_s', 0.5)  # zero cmd if no state for this long

        self._Kp_xy   = self.get_parameter('Kp_xy').value
        self._Kp_z    = self.get_parameter('Kp_z').value
        self._Kp_yaw  = self.get_parameter('Kp_yaw').value
        self._Kd_xy   = self.get_parameter('Kd_xy').value
        self._Kd_z    = self.get_parameter('Kd_z').value
        self._v_max_xy = self.get_parameter('v_max_xy').value
        self._v_max_z  = self.get_parameter('v_max_z').value
        self._yaw_max  = self.get_parameter('yaw_max').value
        self._timeout  = self.get_parameter('timeout_s').value

        # ── State ─────────────────────────────────────────────────────
        self._kf_state: np.ndarray | None = None
        self._ref_pos = np.zeros(3)    # [x, y, z] in world frame
        self._ref_yaw = 0.0
        self._last_state_t = 0.0
        self._ref_received = False     # do not control until first reference arrives

        # ── Publishers / subscribers ──────────────────────────────────
        # Relative topic names — resolved under the node's ROS namespace.
        # kf_state and rc_cmd are siblings in the same namespace as the KF node.
        # reference is also relative, so each drone tracks its own setpoint.
        self._pub_rc = self.create_publisher(Int32MultiArray, 'rc_cmd', 10)

        self._sub_state = self.create_subscription(
            Float64MultiArray, 'kf_state', self._cb_state, 10)
        self._sub_ref = self.create_subscription(
            PoseStamped, 'reference', self._cb_ref, 10)

        dt = 1.0 / self.get_parameter('rate_hz').value
        self._timer = self.create_timer(dt, self._tick)

        self.get_logger().info('tello_controller started')

    # ------------------------------------------------------------------
    def _cb_state(self, msg: Float64MultiArray):
        if len(msg.data) < 12:
            return
        self._kf_state = np.array(msg.data)
        self._last_state_t = time.time()

    def _cb_ref(self, msg: PoseStamped):
        self._ref_pos[0] = msg.pose.position.x
        self._ref_pos[1] = msg.pose.position.y
        self._ref_pos[2] = msg.pose.position.z
        q = msg.pose.orientation
        self._ref_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self._ref_received = True

    # ------------------------------------------------------------------
    def _tick(self):
        # Safety: publish zeros if state is stale, unavailable, or no reference yet
        if (not self._ref_received
                or self._kf_state is None
                or (time.time() - self._last_state_t) > self._timeout):
            self._publish_rc(0, 0, 0, 0)
            return

        x = self._kf_state
        pos = np.array([x[self.IDX_PX], x[self.IDX_PY], x[self.IDX_PZ]])
        vel = np.array([x[self.IDX_VX], x[self.IDX_VY], x[self.IDX_VZ]])
        yaw = x[self.IDX_YAW]

        # PD control in world frame
        e_pos = self._ref_pos - pos
        e_vel = -vel                      # desired velocity = 0 (hold setpoint)

        v_world = (self._Kp_xy * e_pos[:2] + self._Kd_xy * e_vel[:2])
        vz_cmd  =  self._Kp_z  * e_pos[2] + self._Kd_z  * e_vel[2]

        # Clamp world-frame XY velocity
        v_norm = np.linalg.norm(v_world)
        if v_norm > self._v_max_xy:
            v_world = v_world / v_norm * self._v_max_xy
        vz_cmd = float(np.clip(vz_cmd, -self._v_max_z, self._v_max_z))

        # Rotate world-frame velocity to drone body frame using current yaw
        cy, sy = math.cos(yaw), math.sin(yaw)
        # Body frame: +fb = forward (drone nose), +lr = right
        v_forward = v_world[0] * cy + v_world[1] * sy
        v_right   = -v_world[0] * sy + v_world[1] * cy

        # Yaw PD
        e_yaw = self._angle_diff(self._ref_yaw, yaw)
        yaw_cmd = float(np.clip(self._Kp_yaw * e_yaw, -self._yaw_max, self._yaw_max))

        # Scale to djitellopy range [-100, 100]
        lr  = int(np.clip(v_right   / self._v_max_xy * 100, -100, 100))
        fb  = int(np.clip(v_forward / self._v_max_xy * 100, -100, 100))
        ud  = int(np.clip(vz_cmd    / self._v_max_z  * 100, -100, 100))
        yaw_rc = int(np.clip(yaw_cmd / self._yaw_max * 100, -100, 100))

        self._publish_rc(lr, fb, ud, yaw_rc)

    # ------------------------------------------------------------------
    def _publish_rc(self, lr, fb, ud, yaw):
        msg = Int32MultiArray()
        msg.data = [int(lr), int(fb), int(ud), int(yaw)]
        self._pub_rc.publish(msg)

    @staticmethod
    def _angle_diff(target, current):
        """Signed shortest-path angular difference (radians)."""
        d = target - current
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d


def main(args=None):
    rclpy.init(args=args)
    node = TelloControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
