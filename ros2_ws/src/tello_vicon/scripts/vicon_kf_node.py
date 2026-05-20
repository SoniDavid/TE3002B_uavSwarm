import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from tello_vicon_scripts.kalman_filter import ViconKF


def quat_to_euler(qx, qy, qz, qw):
    """Convert quaternion to (roll, pitch, yaw) in radians."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class ViconKFNode(Node):
    """Subscribe to Vicon pose, run Kalman filter, publish smoothed state.

    Subscribes
    ----------
    /vicon/<subject_name>/pose  (geometry_msgs/PoseStamped)

    Publishes
    ---------
    /tello/kf_state  (std_msgs/Float64MultiArray)
        Layout: [px, vx, py, vy, pz, vz, roll, vroll, pitch, vpitch, yaw, vyaw]
    /tello/kf_pose   (geometry_msgs/PoseStamped)  — smoothed pose for visualisation
    """

    def __init__(self):
        super().__init__('vicon_kf_node')

        #  Parameters 
        self.declare_parameter('subject_name', 'tello1')
        self.declare_parameter('dt', 0.01)          # nominal step (sec); overridden by timestamps
        self.declare_parameter('q_pos',  1e-3)
        self.declare_parameter('q_vel',  1e-1)
        self.declare_parameter('q_ang',  1e-4)
        self.declare_parameter('q_rate', 1e-2)
        self.declare_parameter('r_pos',  1e-6)
        self.declare_parameter('r_ang',  1e-5)
        self.declare_parameter('max_path_len', 2000)  # ~20 s at 100 Hz

        subject  = self.get_parameter('subject_name').value
        self._dt = self.get_parameter('dt').value

        #  Kalman filter 
        self._kf = ViconKF()
        self._kf.init(
            dt=self._dt,
            q_pos=self.get_parameter('q_pos').value,
            q_vel=self.get_parameter('q_vel').value,
            q_ang=self.get_parameter('q_ang').value,
            q_rate=self.get_parameter('q_rate').value,
            r_pos=self.get_parameter('r_pos').value,
            r_ang=self.get_parameter('r_ang').value,
        )

        self._last_stamp = None
        self._initialized = False
        self._max_path_len = self.get_parameter('max_path_len').value
        self._path_poses: deque = deque()

        #  Publishers 
        # Relative topic names — resolved under the node's ROS namespace,
        # e.g. /tello1/kf_state when launched with namespace:=/tello1
        self._pub_state = self.create_publisher(Float64MultiArray, 'kf_state', 10)
        self._pub_pose  = self.create_publisher(PoseStamped,       'kf_pose',  10)
        self._pub_path  = self.create_publisher(Path,              'kf_path',  10)

        #  Subscriber 
        topic = f'/vicon/{subject}/pose'
        self._sub = self.create_subscription(
            PoseStamped, topic, self._cb_pose, 10)

        self.get_logger().info(f'vicon_kf_node started, listening on {topic}')

    def _cb_pose(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        roll, pitch, yaw = quat_to_euler(q.x, q.y, q.z, q.w)

        # Recompute dt from message timestamps when possible
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_stamp is not None:
            dt = stamp_sec - self._last_stamp
            if 0.0005 < dt < 1.0:   # sanity: between 0.5 ms and 1 s
                # Reinitialise KF A matrix only when dt changes significantly
                if abs(dt - self._dt) > 1e-4:
                    self._kf.init(
                        dt=dt,
                        q_pos=self.get_parameter('q_pos').value,
                        q_vel=self.get_parameter('q_vel').value,
                        q_ang=self.get_parameter('q_ang').value,
                        q_rate=self.get_parameter('q_rate').value,
                        r_pos=self.get_parameter('r_pos').value,
                        r_ang=self.get_parameter('r_ang').value,
                    )
                    self._dt = dt
        self._last_stamp = stamp_sec

        # Seed the KF state on the first message
        if not self._initialized:
            seed = np.zeros(ViconKF.NX)
            seed[0] = p.x; seed[2] = p.y; seed[4] = p.z
            seed[6] = roll; seed[8] = pitch; seed[10] = yaw
            self._kf._x = seed
            self._initialized = True

        y = np.array([p.x, p.y, p.z, roll, pitch, yaw])
        x = self._kf.step(y)

        # Publish state vector
        state_msg = Float64MultiArray()
        state_msg.data = x.tolist()
        self._pub_state.publish(state_msg)

        # Publish smoothed pose for visualisation
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = msg.header.stamp
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = x[0]
        pose_msg.pose.position.y = x[2]
        pose_msg.pose.position.z = x[4]
        cy, sy = math.cos(x[10] * 0.5), math.sin(x[10] * 0.5)
        cr, sr = math.cos(x[6]  * 0.5), math.sin(x[6]  * 0.5)
        cp, sp = math.cos(x[8]  * 0.5), math.sin(x[8]  * 0.5)
        pose_msg.pose.orientation.w = cr * cp * cy + sr * sp * sy
        pose_msg.pose.orientation.x = sr * cp * cy - cr * sp * sy
        pose_msg.pose.orientation.y = cr * sp * cy + sr * cp * sy
        pose_msg.pose.orientation.z = cr * cp * sy - sr * sp * cy
        self._pub_pose.publish(pose_msg)

        # Accumulate path and publish for Foxglove trajectory visualisation
        self._path_poses.append(pose_msg)
        if len(self._path_poses) > self._max_path_len:
            self._path_poses.popleft()
        path_msg = Path()
        path_msg.header.stamp    = msg.header.stamp
        path_msg.header.frame_id = 'world'
        path_msg.poses = list(self._path_poses)
        self._pub_path.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ViconKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
