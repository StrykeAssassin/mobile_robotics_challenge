"""The mission on the real car (Jetson, Foxy). Bring-up order from the W3 manual first:

  Terminal 1: remote_control                       (then reset the Arduino)
  Terminal 2: ros2 launch car_control car_control.launch.py
  Terminal 3: ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2   (or launch_zed:=true)
  Terminal 4: ros2 launch sar_mission car.launch.py start_corner:=SW
  Then press O on the remote to hand over to autonomous mode; the mission clock starts when the car moves.

Useful switches:
  detector:=false   run without the YOLO node (e.g. obstacle tests only)
  mission:=false    everything except mission_manager (publish /sar/route yourself for tests)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('sar_mission')
    params = os.path.join(share, 'config', 'car.yaml')
    corner = LaunchConfiguration('start_corner')
    over = [params, {'start_corner': corner}]
    nodes = [
        DeclareLaunchArgument('start_corner', default_value='SW'),
        DeclareLaunchArgument('detector', default_value='true'),
        DeclareLaunchArgument('mission', default_value='true'),
        DeclareLaunchArgument('launch_zed', default_value='false'),
        Node(package='sar_mission', executable='depth_scan', parameters=over, output='screen'),
        Node(package='sar_mission', executable='board_detector', parameters=over, output='screen',
             condition=IfCondition(LaunchConfiguration('detector'))),
        Node(package='sar_mission', executable='target_tracker', parameters=over, output='screen'),
        Node(package='sar_mission', executable='exit_detector', parameters=over, output='screen'),
        Node(package='sar_mission', executable='path_planner', parameters=over, output='screen'),
        Node(package='sar_mission', executable='local_driver', parameters=over, output='screen'),
        Node(package='sar_mission', executable='mission_manager', parameters=over, output='screen',
             condition=IfCondition(LaunchConfiguration('mission'))),
    ]
    try:
        zed = get_package_share_directory('zed_wrapper')
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(zed, 'launch', 'zed_camera.launch.py')),
            launch_arguments={'camera_model': 'zed2'}.items(),
            condition=IfCondition(LaunchConfiguration('launch_zed'))))
    except Exception:   # noqa: BLE001 - zed_wrapper not installed (e.g. on the laptop)
        pass
    return LaunchDescription(nodes)
