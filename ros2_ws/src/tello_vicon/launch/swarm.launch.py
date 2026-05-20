"""Swarm launch: spin up N drones, each with its own KF + controller + bridge.

The drone list is configured via the `drones` launch argument as a
colon-separated string of  subject:ip  pairs.  Example:

    ros2 launch tello_vicon swarm.launch.py \\
        drones:="tello1:192.168.10.1:tello2:192.168.20.1:tello3:192.168.30.1"

Each drone gets its own ROS namespace so all topics are independent:
    /tello1/kf_state,  /tello2/kf_state,  /tello3/kf_state  ...
    /tello1/reference, /tello2/reference, /tello3/reference  ...
    /tello1/rc_cmd,    /tello2/rc_cmd,    /tello3/rc_cmd     ...

For offline / mock testing:
    ros2 launch tello_vicon swarm.launch.py mock:=true \\
        drones:="tello1:0.0.0.0:tello2:0.0.0.0:tello3:0.0.0.0"
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _spawn_drone(context, params, mock):
    """OpaqueFunction callback — parses the drones argument and creates nodes."""
    raw   = context.launch_configurations.get('drones', '')
    parts = [p.strip() for p in raw.split(':') if p.strip()]

    if len(parts) % 2 != 0:
        raise ValueError(
            "drones argument must be colon-separated subject:ip pairs, "
            f"got: {raw!r}"
        )

    actions = []
    for i in range(0, len(parts), 2):
        subject  = parts[i]
        drone_ip = parts[i + 1]

        common = dict(package='tello_vicon', output='screen', namespace=subject)

        actions.append(Node(
            **common,
            executable='vicon_kf_node',
            name='vicon_kf_node',
            parameters=[params, {'subject_name': subject}],
        ))
        actions.append(Node(
            **common,
            executable='tello_controller',
            name='tello_controller',
            parameters=[params],
        ))
        actions.append(Node(
            **common,
            executable='tello_bridge',
            name='tello_bridge',
            parameters=[params, {'mock': mock, 'drone_ip': drone_ip}],
        ))

    return actions


def generate_launch_description():
    pkg_share = get_package_share_directory('tello_vicon')
    params    = os.path.join(pkg_share, 'config', 'params.yaml')

    drones_arg = DeclareLaunchArgument(
        'drones',
        default_value='tello1:192.168.10.1:tello2:192.168.20.1:tello3:192.168.30.1',
        description='Colon-separated subject:ip pairs for each drone',
    )
    mock_arg = DeclareLaunchArgument(
        'mock', default_value='false',
        description='Mock mode — no real drones (for offline testing)',
    )

    spawn = OpaqueFunction(
        function=lambda ctx: _spawn_drone(
            ctx,
            params,
            mock=ctx.launch_configurations.get('mock', 'false').lower() == 'true',
        )
    )

    return LaunchDescription([drones_arg, mock_arg, spawn])
