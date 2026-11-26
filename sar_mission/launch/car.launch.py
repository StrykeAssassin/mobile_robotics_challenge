"""The mission on the real car (Jetson, Foxy). Bring-up order from the W3 manual first:

  Terminal 1: remote_control                       (then reset the Arduino)
  Terminal 2: ros2 launch car_control car_control.launch.py
  Terminal 3: ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2   (or launch_zed:=true)
  Terminal 4: ros2 launch sar_mission car.launch.py start_corner:=SW
  Then press O on the remote to hand over to autonomous mode; the mission clock starts when the car moves.

Useful switches:
  detector:=false   run without the YOLO node (e.g. obstacle tests only)
  mission:=false    everything except mission_manager (publish /sar/route yourself for tests)

Known-map mode (KNOWN_MAP.md): give the saved course map, and ideally the tape measurements:
  ros2 launch sar_mission car.launch.py start_corner:=NE \
      prior_map:=~/maps/merged_day1.yaml course_file:=~/maps/course_measurements.yaml
  This adds the localizer (scan-to-map) and map_diff (new-object finder) nodes and switches every
  node to the corrected frame sar_map / topic /sar/odom. Leave prior_map empty for the old behaviour.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    share = get_package_share_directory('sar_mission')
    params = os.path.join(share, 'config', 'car.yaml')
    corner = LaunchConfiguration('start_corner').perform(context)
    prior = os.path.expanduser(LaunchConfiguration('prior_map').perform(context))
    course = os.path.expanduser(LaunchConfiguration('course_file').perform(context))
    over = {'start_corner': corner, 'course_file': course}
    known = bool(prior)
    if known:
        over.update({'prior_map': prior, 'odom_topic': '/sar/odom', 'odom_frame': 'sar_map'})
    p = [params, over]
    det = LaunchConfiguration('detector').perform(context).lower() in ('true', '1', 'yes')
    mis = LaunchConfiguration('mission').perform(context).lower() in ('true', '1', 'yes')
    nodes = [Node(package='sar_mission', executable='depth_scan', parameters=[params, {'start_corner': corner}],
                  output='screen')]
    if known:
        nodes.append(Node(package='sar_mission', executable='localizer',
                          parameters=[params, {'start_corner': corner, 'course_file': course, 'prior_map': prior}],
                          output='screen'))
        nodes.append(Node(package='sar_mission', executable='map_diff', parameters=p, output='screen'))
    if det:
        nodes.append(Node(package='sar_mission', executable='board_detector', parameters=p, output='screen'))
    for exe in ('target_tracker', 'exit_detector', 'path_planner', 'local_driver'):
        nodes.append(Node(package='sar_mission', executable=exe, parameters=p, output='screen'))
    if mis:
        nodes.append(Node(package='sar_mission', executable='mission_manager', parameters=p, output='screen'))
    return nodes


def generate_launch_description():
    items = [
        DeclareLaunchArgument('start_corner', default_value='SW'),
        DeclareLaunchArgument('detector', default_value='true'),
        DeclareLaunchArgument('mission', default_value='true'),
        DeclareLaunchArgument('launch_zed', default_value='false'),
        DeclareLaunchArgument('prior_map', default_value='', description='saved course map .yaml (known-map mode)'),
        DeclareLaunchArgument('course_file', default_value='', description='course_measurements.yaml'),
        OpaqueFunction(function=_nodes),
    ]
    try:
        zed = get_package_share_directory('zed_wrapper')
        items.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(zed, 'launch', 'zed_camera.launch.py')),
            launch_arguments={'camera_model': 'zed2'}.items(),
            condition=IfCondition(LaunchConfiguration('launch_zed'))))
    except Exception:   # noqa: BLE001 - zed_wrapper not installed (e.g. on the laptop)
        pass
    return LaunchDescription(items)
