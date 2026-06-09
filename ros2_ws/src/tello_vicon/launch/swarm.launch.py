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


def _spawn_all(context, params, mock, formation, cam_params, record_mode, output_dir):
    raw   = context.launch_configurations.get('drones', '')
    parts = [p.strip() for p in raw.split(':') if p.strip()]

    if len(parts) % 2 != 0:
        raise ValueError(
            f"drones must be colon-separated subject:ip pairs, got: {raw!r}")

    leader_ns = parts[0]   # first subject is always the leader
    actions   = []

    subjects = [parts[i]     for i in range(0, len(parts), 2)]
    ips      = [parts[i + 1] for i in range(0, len(parts), 2)]

    # ── ONE vicon_kf_node for ALL drones (one process) ──────────────
    # Remaps each drone's Vicon topic to what vicon_kf_node expects:
    #   /vicon/<subject>/<subject>  →  /vicon/<subject>/pose
    vicon_remaps = [
        (f'/vicon/{s}/pose', f'/vicon/{s}/{s}')
        for s in subjects
    ]
    actions.append(Node(
        package='tello_vicon',
        executable='vicon_kf_node',
        name='vicon_kf_node',
        output='screen',
        parameters=[params, {
            'drone_subjects': ','.join(subjects),
        }],
        remappings=vicon_remaps,
    ))

    # ── ONE tello_controller for ALL drones (one process) ────────────
    actions.append(Node(
        package='tello_vicon',
        executable='tello_controller',
        name='tello_controller',
        output='screen',
        parameters=[params, {
            'drone_subjects': ','.join(subjects),
        }],
    ))

    # ── Single swarm_bridge for ALL drones (one process = one UDP socket) ──
    actions.append(Node(
        package='tello_vicon',
        executable='swarm_bridge',
        name='swarm_bridge',
        output='screen',
        parameters=[params, {
            'mock':           mock,
            'drone_subjects': ','.join(subjects),
            'drone_ips':      ','.join(ips),
            'leader_ns':      leader_ns,
        }],
    ))

    # ── ArUco node — leader namespace only ───────────────────────────
    # Subscribes to /<leader_ns>/image_raw/compressed and /<leader_ns>/kf_state.
    # Publishes /aruco/pose (world frame) for formation_controller.
    actions.append(Node(
        package='tello_vicon',
        executable='aruco_node',
        name='aruco_node',
        namespace=leader_ns,
        output='screen',
        parameters=[params, {
            'camera_params_file': cam_params,
            'image_topic':        'image_raw/compressed',
        }],
    ))

    # ── Formation controller — one for the whole swarm ───────────────
    # leader_ns, s1_ns, s2_ns MUST be passed explicitly — the node
    # defaults are tello0/tello1/tello2 which don't match real subjects.
    follower_nss = [s for s in subjects if s != leader_ns]
    s1_ns = follower_nss[0] if len(follower_nss) > 0 else ''
    s2_ns = follower_nss[1] if len(follower_nss) > 1 else ''

    actions.append(Node(
        package='tello_vicon',
        executable='formation_controller',
        name='formation_controller',
        output='screen',
        parameters=[params, {
            'formation': formation,
            'leader_ns': leader_ns,
            's1_ns':     s1_ns,
            's2_ns':     s2_ns,
        }],
    ))


    # ── Video recorder — subscribes to /<ns>/image_raw for each drone ──
    # Switch mode at runtime:
    #   ros2 service call /video/set_mode std_srvs/srv/SetBool "data: true"   # all
    #   ros2 service call /video/set_mode std_srvs/srv/SetBool "data: false"  # leader only
    actions.append(Node(
        package='tello_vicon',
        executable='video_recorder',
        name='video_recorder',
        output='screen',
        parameters=[params, {
            'drone_subjects': ','.join(subjects),
            'leader_ns':      leader_ns,
            'record_mode':    record_mode,
            'output_dir':     output_dir,
        }],
    ))

    return actions


def generate_launch_description():
    pkg_share = get_package_share_directory('tello_vicon')
    params    = os.path.join(pkg_share, 'config', 'params.yaml')
    default_cam = os.path.join(pkg_share, 'config', 'camera_params.npz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'drones',
            default_value='tello_soni1:192.168.0.100:tello_soni2:192.168.0.104:tello_soni3:192.168.0.102',
            description='Colon-separated subject:ip pairs; first = leader',
        ),
        DeclareLaunchArgument(
            'mock', default_value='false',
            description='Mock mode — no real drones',
        ),
        DeclareLaunchArgument(
            'formation', default_value='LINE',
            description='V | LINE | COLUMN | PANORAMIC | RECONSTRUCTION',
        ),
        DeclareLaunchArgument(
            'cam_params', default_value=default_cam,
            description='Absolute path to camera_params.npz',
        ),
        DeclareLaunchArgument(
            'record_mode', default_value='leader',
            description='Video recording mode: leader | all',
        ),
        DeclareLaunchArgument(
            'output_dir', default_value='/home/kfcnef/videos',
            description='Directory to save recordings',
        ),
        OpaqueFunction(
            function=lambda ctx: _spawn_all(
                ctx,
                params,
                mock        = ctx.launch_configurations.get('mock','false').lower()=='true',
                formation   = ctx.launch_configurations.get('formation', 'LINE'),
                cam_params  = ctx.launch_configurations.get('cam_params', default_cam),
                record_mode = ctx.launch_configurations.get('record_mode', 'leader'),
                output_dir  = ctx.launch_configurations.get('output_dir', ''),
            )
        ),
    ])