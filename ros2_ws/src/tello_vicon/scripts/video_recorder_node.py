"""video_recorder_node.py — record video from one or all drones.

Subscribes to /<ns>/image_raw for each drone and writes .mp4 files
using OpenCV VideoWriter.

Parameters
----------
drone_subjects : str   comma-separated namespace list (same as swarm_bridge)
record_mode    : str   "leader" = only leader | "all" = all drones (default "leader")
leader_ns      : str   namespace of the leader drone
output_dir     : str   directory to save videos (default /home/kfcnef/videos)
fps            : float recording frame rate (default 30.0)
codec          : str   fourcc codec string (default "mp4v")

Topics subscribed
-----------------
  /<ns>/image_raw  (sensor_msgs/Image)  one per drone

Services
--------
  /video/set_mode  (std_srvs/SetBool)
    data=true  → switch to "all"    mode
    data=false → switch to "leader" mode

Runtime mode switch (no rebuild needed)
----------------------------------------
  ros2 service call /video/set_mode std_srvs/srv/SetBool "data: true"   # all
  ros2 service call /video/set_mode std_srvs/srv/SetBool "data: false"  # leader only
"""

import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import SetBool


class VideoRecorderNode(Node):

    def __init__(self):
        super().__init__('video_recorder')

        self.declare_parameter('drone_subjects', 'tello0')
        self.declare_parameter('record_mode',    'leader')   # "leader" | "all"
        self.declare_parameter('leader_ns',      'tello0')
        self.declare_parameter('output_dir', '/home/kfcnef/homework/herman/videos')
        self.declare_parameter('fps',            30.0)
        self.declare_parameter('codec',          'mp4v')

        subjects    = [s.strip() for s in
                       self.get_parameter('drone_subjects').value.split(',') if s.strip()]
        self._mode  = self.get_parameter('record_mode').value.lower()
        leader_ns   = self.get_parameter('leader_ns').value
        output_dir  = self.get_parameter('output_dir').value
        self._fps   = self.get_parameter('fps').value
        codec_str   = self.get_parameter('codec').value

        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir

        self._fourcc   = cv2.VideoWriter_fourcc(*codec_str)
        self._leader   = leader_ns
        self._subjects = subjects
        self._lock     = threading.Lock()

        # Per-drone state
        self._writers:   dict[str, cv2.VideoWriter | None] = {s: None for s in subjects}
        self._recording: dict[str, bool]                   = {s: False for s in subjects}
        self._frame_sz:  dict[str, tuple | None]           = {s: None for s in subjects}

        # Subscribe to image_raw for every drone
        for ns in subjects:
            self.create_subscription(
                Image, f'/{ns}/image_raw',
                lambda msg, n=ns: self._cb_image(msg, n), 5)

        # Service to switch mode at runtime
        self._srv = self.create_service(
            SetBool, '/video/set_mode', self._cb_set_mode)

        # Publisher for status messages
        self._pub_status = self.create_publisher(String, '/video/status', 10)

        # Start recording based on initial mode
        self._apply_mode()

        self.get_logger().info(
            f'video_recorder ready  mode={self._mode}  '
            f'output={self._output_dir}')

    # ── Mode management ───────────────────────────────────────────

    def _active_subjects(self) -> list[str]:
        """Return the list of drones that should be recording right now."""
        if self._mode == 'all':
            return self._subjects
        return [self._leader]

    def _apply_mode(self):
        """Start/stop writers based on current mode."""
        active = set(self._active_subjects())
        with self._lock:
            for ns in self._subjects:
                should_record = ns in active
                if should_record and not self._recording[ns]:
                    self._recording[ns] = True   # writer opened on first frame
                    self.get_logger().info(f'[{ns}] recording queued')
                elif not should_record and self._recording[ns]:
                    self._stop_writer(ns)

    def _cb_set_mode(self, request: SetBool.Request,
                     response: SetBool.Response):
        new_mode = 'all' if request.data else 'leader'
        if new_mode == self._mode:
            response.success = True
            response.message = f'Already in {self._mode} mode'
            return response

        self._mode = new_mode
        self._apply_mode()
        msg = f'Switched to {self._mode} mode'
        self.get_logger().info(msg)
        self._pub_status.publish(String(data=msg))
        response.success = True
        response.message = msg
        return response

    # ── Image callback ────────────────────────────────────────────

    def _cb_image(self, msg: Image, ns: str):
        with self._lock:
            if not self._recording.get(ns):
                return

            # Convert ROS Image → numpy BGR
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, 3))

            # Open writer on first frame (we learn the resolution here)
            if self._writers[ns] is None:
                h, w = frame.shape[:2]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.join(
                    self._output_dir, f'{ns}_{timestamp}.mp4')
                writer = cv2.VideoWriter(
                    path, self._fourcc, self._fps, (w, h))
                if not writer.isOpened():
                    self.get_logger().error(
                        f'[{ns}] failed to open VideoWriter at {path}')
                    self._recording[ns] = False
                    return
                self._writers[ns] = writer
                self._frame_sz[ns] = (w, h)
                self.get_logger().info(
                    f'[{ns}] recording started → {path}')
                self._pub_status.publish(
                    String(data=f'[{ns}] recording → {path}'))

            self._writers[ns].write(frame)

    # ── Writer helpers ────────────────────────────────────────────

    def _stop_writer(self, ns: str):
        """Release writer for one drone. Must be called with self._lock held."""
        self._recording[ns] = False
        w = self._writers[ns]
        if w is not None:
            w.release()
            self._writers[ns] = None
            self.get_logger().info(f'[{ns}] recording stopped')

    def _stop_all(self):
        with self._lock:
            for ns in self._subjects:
                if self._writers[ns] is not None or self._recording[ns]:
                    self._stop_writer(ns)

    def destroy_node(self):
        self._stop_all()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()