"""Record a rosbag of raw Vicon poses — for KF parameter tuning offline.

This launch file intentionally does NOT start the KF node. Recording only
the raw sensor data means you can replay the bag as many times as needed
with different Q/R values in params.yaml, each time getting a fresh KF
estimate from the same ground-truth trajectory.

Usage (while flying a trajectory with TE3002B_UAV djitellopy scripts):
    ros2 launch tello_vicon record_bag.launch.py subject:=tello1

Bag saved to: ~/bags/tello_vicon_<subject>_<timestamp>/

Offline KF tuning workflow:
    Terminal 1:  ros2 launch tello_vicon kf_only.launch.py subject:=tello1
    Terminal 2:  ros2 bag play ~/bags/tello_vicon_tello1_<date>/ 
                               --topics /vicon/tello1/pose
    Foxglove:    compare /vicon/tello1/pose  vs  /tello1/kf_pose  and  /tello1/kf_path
    Tune:        edit q_pos, q_vel, r_pos in config/params.yaml, re-run both terminals
"""
import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction


def _make_record(context):
    subject   = context.launch_configurations.get('subject', 'tello1')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_dir   = os.path.expanduser(f'~/bags/tello_vicon_{subject}_{timestamp}')

    return [ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-o', bag_dir,
            f'/vicon/{subject}/pose',
        ],
        output='screen',
    )]


def generate_launch_description():
    subject_arg = DeclareLaunchArgument(
        'subject', default_value='tello1',
        description='Vicon subject name (must match Vicon Tracker label)')

    spawn = OpaqueFunction(function=lambda ctx: _make_record(ctx))

    return LaunchDescription([subject_arg, spawn])
