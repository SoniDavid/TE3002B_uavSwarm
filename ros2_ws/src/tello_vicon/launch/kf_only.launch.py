"""KF node only — for offline rosbag replay and KF parameter tuning.

Usage:
    Terminal 1:  ros2 launch tello_vicon kf_only.launch.py subject:=tello1
    Terminal 2:  ros2 bag play <path-to-bag>
    Visualise:   Foxglove → subscribe /tello1/kf_state, /tello1/kf_pose
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('tello_vicon')
    params    = os.path.join(pkg_share, 'config', 'params.yaml')

    subject_arg = DeclareLaunchArgument(
        'subject', default_value='tello1',
        description='Vicon subject name (must match bag topic prefix)')

    ns = LaunchConfiguration('subject')

    vicon_kf = Node(
        package='tello_vicon',
        executable='vicon_kf_node',
        name='vicon_kf_node',
        namespace=ns,
        parameters=[params, {'subject_name': ns}],
        output='screen',
    )

    return LaunchDescription([subject_arg, vicon_kf])
