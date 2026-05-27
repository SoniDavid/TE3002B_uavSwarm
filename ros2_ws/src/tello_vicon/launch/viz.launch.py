"""Launch the Vicon visualiser node and RViz2.

Usage:
    ros2 launch tello_vicon viz.launch.py
    ros2 launch tello_vicon viz.launch.py subjects:="['tello_soni1']"
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('tello_vicon')
    rviz_cfg  = os.path.join(pkg_share, 'config', 'rviz', 'vicon_viz.rviz')

    subjects_arg = DeclareLaunchArgument(
        'subjects',
        default_value="['tello_soni1', 'robomaster_soni']",
        description='List of Vicon subject names to track',
    )

    viz_node = Node(
        package='tello_vicon',
        executable='vicon_viz_node',
        name='vicon_viz_node',
        parameters=[{'subjects': LaunchConfiguration('subjects')}],
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        output='screen',
    )

    return LaunchDescription([subjects_arg, viz_node, rviz_node])
