from glob import glob
import os
from setuptools import setup

package_name = 'tello_vicon'

setup(
    name=package_name,
    version='0.1.0',
    packages=['tello_vicon_scripts'],
    package_dir={'tello_vicon_scripts': 'scripts'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config', 'foxglove_layouts'), glob('config/foxglove_layouts/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='sonidavid46@gmail.com',
    description='Vicon-based closed-loop position control for DJI Tello drones.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vicon_kf_node    = tello_vicon_scripts.vicon_kf_node:main',
            'tello_controller = tello_vicon_scripts.tello_controller_node:main',
            'tello_bridge     = tello_vicon_scripts.tello_bridge_node:main',
        ],
    },
)
