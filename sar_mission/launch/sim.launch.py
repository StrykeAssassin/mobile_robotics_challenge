"""Gazebo test of the whole mission with a TurtleBot3 that drives like the RC car.

  export TURTLEBOT3_MODEL=waffle
  ros2 launch sar_mission sim.launch.py                      # world seed 3, start SW
  ros2 launch sar_mission sim.launch.py seed:=7 start_corner:=NE
  (make a new world first: python3 tools/make_world.py --seed 7, then colcon build)
"""
import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    share = get_package_share_directory('sar_mission')
    seed = LaunchConfiguration('seed').perform(context)
    corner = LaunchConfiguration('start_corner').perform(context)
    gui = LaunchConfiguration('gui').perform(context)
    world = os.path.join(share, 'worlds', 'arena_seed%s.world' % seed)
    layout = os.path.join(share, 'worlds', 'arena_seed%s.json' % seed)
    params = os.path.join(share, 'config', 'sim.yaml')

    # start pose in the world (= arena) frame, same rule as core.geometry.start_pose_from_corner
    s, inset = 20.0, 1.5
    pos = {'SW': (inset, inset), 'SE': (s - inset, inset), 'NE': (s - inset, s - inset),
           'NW': (inset, s - inset)}[corner.upper()]
    yaw = {'SW': 0.0, 'SE': math.pi / 2, 'NE': math.pi, 'NW': -math.pi / 2}[corner.upper()]

    gazebo_ros = get_package_share_directory('gazebo_ros')
    tb3 = get_package_share_directory('turtlebot3_gazebo')
    model = os.environ.get('TURTLEBOT3_MODEL', 'waffle')
    sdf = os.path.join(tb3, 'models', 'turtlebot3_%s' % model, 'model.sdf')

    over = [params, {'start_corner': corner}]
    actions = [
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gzserver.launch.py')),
                                 launch_arguments={'world': world}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(tb3, 'launch', 'robot_state_publisher.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items()),
        Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
             arguments=['-entity', 'sar_car', '-file', sdf, '-x', str(pos[0]), '-y', str(pos[1]), '-z', '0.01',
                        '-Y', str(yaw)]),
        Node(package='sar_mission', executable='steering_wrapper', parameters=over, output='screen'),
        Node(package='sar_mission', executable='fake_detector', output='screen',
             parameters=over + [{'layout_file': layout, 'seed': int(seed)}]),
        Node(package='sar_mission', executable='target_tracker', parameters=over, output='screen'),
        Node(package='sar_mission', executable='exit_detector', parameters=over, output='screen'),
        Node(package='sar_mission', executable='path_planner', parameters=over, output='screen'),
        Node(package='sar_mission', executable='local_driver', parameters=over, output='screen'),
        Node(package='sar_mission', executable='mission_manager', parameters=over, output='screen'),
    ]
    if gui.lower() == 'true':
        actions.insert(1, IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gzclient.launch.py'))))
    return actions


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', os.environ.get('TURTLEBOT3_MODEL', 'waffle')),
        DeclareLaunchArgument('seed', default_value='3'),
        DeclareLaunchArgument('start_corner', default_value='SW'),
        DeclareLaunchArgument('gui', default_value='true'),
        OpaqueFunction(function=setup),
    ])
