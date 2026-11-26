"""Auto-mapping run: ZED + RTAB-Map (Tutorial 4 settings) + depth_scan + auto_mapper.

Bring-up (W3 manual order), each in its own terminal on the car:
  1. remote_control                                   (then reset the Arduino)
  2. ros2 launch car_control car_control.launch.py
  3. ros2 launch sar_mission mapping.launch.py start_corner:=SW
     (starts the ZED and RTAB-Map too; add zed:=false / rtabmap:=false if you start them yourself)
  4. Hand the car over to autonomous control on the remote.
When it logs MAPPING RUN FINISHED: Ctrl+C this launch (RTAB-Map saves ~/.ros/rtabmap.db), then
copy the database straight away, because the next RTAB-Map start deletes it.

Arguments:
  start_corner:=SW|SE|NE|NW   which taped start mark the car starts on (squared to the edge)
  zed:=true|false             start the ZED wrapper here (zed_camera.launch.py camera_model:=zed2)
  rtabmap:=true|false         start RTAB-Map here with the Tutorial 4 arguments
  rtabmap_viz:=true|false     RTAB-Map's own GUI (heavy on the Jetson; use rviz on the laptop instead)
  bag:=true|false             record a small rosbag (odom, IMU, TF, scan, maps) for testing at home
  bag_images:=true|false      also record JPEG colour images (~2-4 MB/s; check free disk space)
  drive:=true|false           false = everything except auto_mapper (drive by hand with the remote)
  rtabmap_extra:="..."        extra RTAB-Map parameters (default: depth up to 4 m + ray tracing);
                              rtabmap_extra:="" gives exactly the Tutorial 4 settings
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Small topics only: raw ZED images are ~55 MB/s (about 100 GB for a 30 min run) and would fill
# the Jetson's disk. bag_images:=true adds the JPEG-compressed colour image (~2-4 MB/s).
BAG_TOPICS = ['/zed/zed_node/odom', '/zed/zed_node/imu/data', '/zed/zed_node/rgb/camera_info', '/tf',
              '/tf_static', '/sar/scan', '/sar/floor', '/sar/grid', '/sar/path', '/sar/mapper_status',
              '/cmd_vel', '/map', '/rtabmap/mapGraph', '/rtabmap/info']
IMAGE_TOPICS = ['/zed/zed_node/rgb/image_rect_color/compressed']
RTAB_EXTRA = '--Grid/RangeMax 4.0 --Grid/RayTracing true'


def generate_launch_description():
    share = get_package_share_directory('sar_mission')
    params = os.path.join(share, 'config', 'mapping.yaml')
    over = [params, {'start_corner': LaunchConfiguration('start_corner')}]
    actions = [
        DeclareLaunchArgument('start_corner', default_value='SW'),
        DeclareLaunchArgument('zed', default_value='true'),
        DeclareLaunchArgument('rtabmap', default_value='true'),
        DeclareLaunchArgument('rtabmap_viz', default_value='false'),
        DeclareLaunchArgument('bag', default_value='true'),
        DeclareLaunchArgument('bag_images', default_value='false'),
        DeclareLaunchArgument('drive', default_value='true'),
        # Extra RTAB-Map parameters on top of Tutorial 4's: only trust depth up to 4 m (ZED depth
        # gets noisy further out and smears obstacle edges) and ray-trace free space.
        DeclareLaunchArgument('rtabmap_extra', default_value=RTAB_EXTRA),
        Node(package='sar_mission', executable='depth_scan', parameters=over, output='screen'),
        # give the ZED and RTAB-Map time to initialise (IMU, first keyframe) before the car moves
        TimerAction(period=15.0, actions=[
            Node(package='sar_mission', executable='auto_mapper', parameters=over, output='screen',
                 condition=IfCondition(LaunchConfiguration('drive')))]),
        ExecuteProcess(cmd=['ros2', 'bag', 'record', '-o',
                            os.path.expanduser('~/bags/mapping_run')] + BAG_TOPICS,
                       output='screen', condition=IfCondition(LaunchConfiguration('bag'))),
        ExecuteProcess(cmd=['ros2', 'bag', 'record', '-o',
                            os.path.expanduser('~/bags/mapping_images')] + IMAGE_TOPICS,
                       output='screen', condition=IfCondition(LaunchConfiguration('bag_images'))),
    ]
    try:
        zed = get_package_share_directory('zed_wrapper')
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(zed, 'launch', 'zed_camera.launch.py')),
            launch_arguments={'camera_model': 'zed2'}.items(),
            condition=IfCondition(LaunchConfiguration('zed'))))
        # Tutorial 4's RTAB-Map command, as a delayed include (after the ZED is up)
        actions.append(TimerAction(period=8.0, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(zed, 'launch', 'rtabmap.launch.py')),
            launch_arguments={
                'rtabmap_args': ['--delete_db_on_start ', LaunchConfiguration('rtabmap_extra')],
                'rgb_topic': '/zed/zed_node/rgb/image_rect_color',
                'depth_topic': '/zed/zed_node/depth/depth_registered',
                'camera_info_topic': '/zed/zed_node/rgb/camera_info',
                'odom_topic': '/zed/zed_node/odom',
                'visual_odometry': 'false',
                'frame_id': 'zed_camera_link',
                'approx_sync': 'true',
                'wait_imu_to_init': 'true',
                'rgbd_sync': 'true',
                'approx_rgbd_sync': 'true',
                'imu_topic': '/zed/zed_node/imu/data',
                'qos': '0',
                'rviz': 'false',
                'rtabmapviz': LaunchConfiguration('rtabmap_viz'),
            }.items(),
            condition=IfCondition(LaunchConfiguration('rtabmap')))]))
    except Exception:   # noqa: BLE001 - zed_wrapper not installed (laptop): start the ZED/RTAB-Map yourself
        pass
    return LaunchDescription(actions)
