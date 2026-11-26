import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'sar_mission'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', 'MAPPING.md', 'KNOWN_MAP.md', 'CARDS.md']),
        (os.path.join('share', package_name, 'cards'), glob('cards/*.png') + glob('cards/*.jpg')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world') + glob('worlds/*.json')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Group 8',
    maintainer_email='group8@example.com',
    description='50.047 Mobile Robotics search and rescue challenge: autonomous search, target '
                'classification and exit for the ZED2 RC car.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission_manager = sar_mission.nodes.mission_manager:main',
            'path_planner = sar_mission.nodes.path_planner:main',
            'local_driver = sar_mission.nodes.local_driver:main',
            'depth_scan = sar_mission.nodes.depth_scan:main',
            'target_tracker = sar_mission.nodes.target_tracker:main',
            'exit_detector = sar_mission.nodes.exit_detector:main',
            'board_detector = sar_mission.nodes.board_detector:main',
            'fake_detector = sar_mission.nodes.fake_detector:main',
            'steering_wrapper = sar_mission.nodes.steering_wrapper:main',
            'dataset_capture = sar_mission.nodes.dataset_capture:main',
            'auto_mapper = sar_mission.nodes.auto_mapper:main',
            'save_map = sar_mission.nodes.save_map:main',
            'localizer = sar_mission.nodes.localizer:main',
            'map_diff = sar_mission.nodes.map_diff:main',
        ],
    },
)
