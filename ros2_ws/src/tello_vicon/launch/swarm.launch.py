"""Swarm launch: spin up N drones, each with its own KF + controller + bridge,
plus one shared formation_controller.
 
    ros2 launch tello_vicon swarm.launch.py \
        drones:="tello0:192.168.1.50:tello1:192.168.1.51:tello2:192.168.1.52"
 
For offline / mock testing:
    ros2 launch tello_vicon swarm.launch.py mock:=true \
        drones:="tello0:0.0.0.0:tello1:0.0.0.0:tello2:0.0.0.0"
"""

import os
 
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
 
 
def _spawn_drone(context, params, mock):
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
 
    # One formation controller for the whole swarm (not per-drone)
    actions.append(Node(
        package='tello_vicon',
        executable='formation_controller',
        name='formation_controller',
        output='screen',
        parameters=[params],
    ))
 
    return actions
 
 
def generate_launch_description():
    pkg_share = get_package_share_directory('tello_vicon')
    params    = os.path.join(pkg_share, 'config', 'params.yaml')
 
    drones_arg = DeclareLaunchArgument(
        'drones',
        default_value='tello0:192.168.1.50:tello1:192.168.1.51:tello2:192.168.1.52',
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
