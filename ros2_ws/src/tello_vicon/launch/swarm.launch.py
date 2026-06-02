"""Swarm launch — N drones + ArUco pipeline on leader + formation controller.

Usage
-----
    ros2 launch tello_vicon swarm.launch.py \
        drones:="tello0:192.168.1.50:tello1:192.168.1.51:tello2:192.168.1.52" \
        cam_params:=/abs/path/to/camera_params.npz

Mock / offline:
    ros2 launch tello_vicon swarm.launch.py mock:=true \
        drones:="tello0:0.0.0.0:tello1:0.0.0.0:tello2:0.0.0.0"

Arguments
---------
drones      colon-separated subject:ip pairs — first pair is the leader
mock        true = no real drones (default false)
formation   V | LINE | COLUMN | PANORAMIC | RECONSTRUCTION (default V)
cam_params  absolute path to camera_params.npz
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def _spawn_all(context, params, mock, formation, cam_params):
    raw   = context.launch_configurations.get('drones', '')
    parts = [p.strip() for p in raw.split(':') if p.strip()]

    if len(parts) % 2 != 0:
        raise ValueError(
            f"drones must be colon-separated subject:ip pairs, got: {raw!r}")

    leader_ns = parts[0]   # first subject is always the leader
    actions   = []

    # ── Per-drone stack ──────────────────────────────────────────────
    for i in range(0, len(parts), 2):
        subject   = parts[i]
        drone_ip  = parts[i + 1]
        is_leader = (subject == leader_ns)

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
            parameters=[params, {
                'mock':          mock,
                'drone_ip':      drone_ip,
                # Only the leader needs to stream video
                'publish_image': is_leader,
            }],
        ))

    # ArUco node — leader namespace only 
    # Subscribes to /<leader_ns>/image_raw and /<leader_ns>/kf_state.
    # Publishes /aruco/pose (world frame) for formation_controller.
    actions.append(Node(
        package='tello_vicon',
        executable='aruco_node',
        name='aruco_node',
        namespace=leader_ns,
        output='screen',
        parameters=[params, {
            'camera_params_file': cam_params,
            'image_topic':        'image_raw',   # relative → /<leader_ns>/image_raw
        }],
    ))

    # Formation controller — one for the whole swarm 
    actions.append(Node(
        package='tello_vicon',
        executable='formation_controller',
        name='formation_controller',
        output='screen',
        parameters=[params, {'formation': formation}],
    ))

    return actions


def generate_launch_description():
    pkg_share = get_package_share_directory('tello_vicon')
    params    = os.path.join(pkg_share, 'config', 'params.yaml')
    default_cam = os.path.join(pkg_share, 'config', 'camera_params.npz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'drones',
            default_value='tello0:192.168.1.50:tello1:192.168.1.51:tello2:192.168.1.52',
            description='Colon-separated subject:ip pairs; first = leader',
        ),
        DeclareLaunchArgument(
            'mock', default_value='false',
            description='Mock mode — no real drones',
        ),
        DeclareLaunchArgument(
            'formation', default_value='V',
            description='V | LINE | COLUMN | PANORAMIC | RECONSTRUCTION',
        ),
        DeclareLaunchArgument(
            'cam_params', default_value=default_cam,
            description='Absolute path to camera_params.npz',
        ),
        OpaqueFunction(
            function=lambda ctx: _spawn_all(
                ctx,
                params,
                mock      = ctx.launch_configurations.get('mock','false').lower()=='true',
                formation = ctx.launch_configurations.get('formation', 'V'),
                cam_params= ctx.launch_configurations.get('cam_params', default_cam),
            )
        ),
    ])