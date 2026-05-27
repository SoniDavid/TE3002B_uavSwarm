"""Visualisation bridge: Vicon PoseStamped → TF + Path for RViz2.

For each configured subject this node:
  - Broadcasts a TF transform  world → <subject>
  - Publishes nav_msgs/Path    /vicon/<subject>/path

RViz2 setup (Fixed Frame = world):
  Add > TF          — shows drone axes live
  Add > Path        — shows trajectory
  Add > Pose        — subscribe /vicon/<subject>/<subject> directly
"""

from collections import deque

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class ViconVizNode(Node):
    def __init__(self):
        super().__init__('vicon_viz_node')

        self.declare_parameter('subjects', ['tello_soni1', 'robomaster_soni'])
        self.declare_parameter('max_path_len', 3000)

        subjects = self.get_parameter('subjects').value
        max_path = self.get_parameter('max_path_len').value

        self._tf_br = TransformBroadcaster(self)

        self._path_pubs = {}
        self._path_poses: dict[str, deque] = {}

        for subj in subjects:
            topic = f'/vicon/{subj}/{subj}'
            pub   = self.create_publisher(Path, f'/vicon/{subj}/path', 10)
            self._path_pubs[subj]  = pub
            self._path_poses[subj] = deque(maxlen=max_path)
            self.create_subscription(
                PoseStamped, topic,
                lambda msg, s=subj: self._cb(msg, s),
                10,
            )
            self.get_logger().info(f'Subscribed to {topic}')

    def _cb(self, msg: PoseStamped, subject: str):
        # Broadcast TF: world → subject
        tf = TransformStamped()
        tf.header.stamp    = msg.header.stamp
        tf.header.frame_id = 'world'
        tf.child_frame_id  = subject
        tf.transform.translation.x = msg.pose.position.x
        tf.transform.translation.y = msg.pose.position.y
        tf.transform.translation.z = msg.pose.position.z
        tf.transform.rotation      = msg.pose.orientation
        self._tf_br.sendTransform(tf)

        # Accumulate and publish Path
        self._path_poses[subject].append(msg)
        path               = Path()
        path.header.stamp    = msg.header.stamp
        path.header.frame_id = 'world'
        path.poses           = list(self._path_poses[subject])
        self._path_pubs[subject].publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = ViconVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
