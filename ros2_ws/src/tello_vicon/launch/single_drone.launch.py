"""Single drone: Vicon KF + PD controller + Tello bridge, all under one namespace.

Usage:
    ros2 launch tello_vicon single_drone.launch.py subject:=tello1 drone_ip:=192.168.10.1

Topics created (example with subject:=tello1):
    /tello1/kf_state   — KF state [px,vx, py,vy, pz,vz, roll,vroll, pitch,vpitch, yaw,vyaw]
    /tello1/kf_pose    — smoothed PoseStamped for visualisation
    /tello1/reference  — setpoint input (PoseStamped)
    /tello1/rc_cmd     — RC command [lr, fb, ud, yaw_rate]
    /tello1/battery    — battery %
    /tello1/land       — publish any Bool to trigger landing
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

    subject_arg  = DeclareLaunchArgument('subject',  default_value='tello1')
    drone_ip_arg = DeclareLaunchArgument('drone_ip', default_value='192.168.10.1')
    mock_arg     = DeclareLaunchArgument('mock',     default_value='false')

    ns = LaunchConfiguration('subject')

    vicon_kf = Node(
        package='tello_vicon',
        executable='vicon_kf_node',
        name='vicon_kf_node',
        namespace=ns,
        parameters=[params, {'subject_name': ns}],
        output='screen',
    )

    controller = Node(
        package='tello_vicon',
        executable='tello_controller',
        name='tello_controller',
        namespace=ns,
        parameters=[params],
        output='screen',
    )

    bridge = Node(
        package='tello_vicon',
        executable='tello_bridge',
        name='tello_bridge',
        namespace=ns,
        parameters=[params, {
            'mock':     LaunchConfiguration('mock'),
            'drone_ip': LaunchConfiguration('drone_ip'),
        }],
        output='screen',
    )

    return LaunchDescription([subject_arg, drone_ip_arg, mock_arg,
                               vicon_kf, controller, bridge])
